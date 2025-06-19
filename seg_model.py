import cv2
import streamlit as st
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (jaccard_score, f1_score, 
                           accuracy_score, precision_score, 
                           recall_score)
from scipy.spatial.distance import directed_hausdorff

# Load YOLOv8 segmentation model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov8', 'yolov8x-seg').to('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model()

def calculate_boundary_iou(mask1, mask2, boundary_width=2):
    """Calculate Boundary IoU between two masks"""
    kernel = np.ones((boundary_width, boundary_width), np.uint8)
    boundary1 = cv2.morphologyEx(mask1, cv2.MORPH_GRADIENT, kernel)
    boundary2 = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, kernel)
    return jaccard_score(boundary1.flatten(), boundary2.flatten())

def calculate_segmentation_metrics(results, img_shape):
    """Calculate all requested segmentation metrics"""
    metrics = {}
    
    if results[0].masks is None:
        return {"error": "No masks detected"}
        
    # Convert masks to numpy array
    pred_masks = torch.stack([m.data[0] for m in results[0].masks]).cpu().numpy()
    pred_masks = (pred_masks > 0.5).astype(np.uint8)  # Binarize
    
    # Mock ground truth (replace with actual GT for real applications)
    gt_masks = np.zeros_like(pred_masks)
    h, w = img_shape[:2]
    gt_masks[:, h//4:3*h//4, w//4:3*w//4] = 1  # Simple centered rectangle
    
    # Initialize metric storage
    ious, dices, pixel_accs, mean_accs, boundary_ious = [], [], [], [], []
    class_counts = np.zeros(model.names.__len__())
    
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # Flatten masks for metric calculation
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Calculate metrics per mask
        ious.append(jaccard_score(gt_flat, pred_flat))
        dices.append(f1_score(gt_flat, pred_flat))
        pixel_accs.append(accuracy_score(gt_flat, pred_flat))
        mean_accs.append(np.mean([precision_score(gt_flat, pred_flat),
                              recall_score(gt_flat, pred_flat)]))
        boundary_ious.append(calculate_boundary_iou(gt_mask, pred_mask))
        
        # For class-wise metrics (using detected class)
        cls = results[0].boxes.cls[len(ious)-1]  # Corresponding class
        class_counts[int(cls)] += 1
    
    # Aggregate metrics
    metrics.update({
        'IoU': {
            'mean': np.mean(ious),
            'per_instance': ious,
            'class_wise': {model.names[i]: c/sum(class_counts) 
                          for i, c in enumerate(class_counts) if c > 0}
        },
        'Dice': np.mean(dices),
        'Pixel_Accuracy': np.mean(pixel_accs),
        'Mean_Accuracy': np.mean(mean_accs),
        'mIoU': np.mean(ious),  # Same as mean IoU here
        'Frequency_Weighted_IoU': np.average(ious, weights=class_counts[class_counts > 0]),
        'Boundary_IoU': np.mean(boundary_ious),
        'Boundary_Fscore': f1_score(np.concatenate([m.flatten() for m in gt_masks]),
                                   np.concatenate([m.flatten() for m in pred_masks])),
        'Object_Counts': {model.names[i]: int(c) 
                         for i, c in enumerate(class_counts) if c > 0}
    })
    
    return metrics

def process_image(input_img):
    img = np.array(input_img)
    results = model(img)
    
    # Visualization panels
    left_img = img.copy()
    if results[0].masks is not None:
        for mask in results[0].masks:
            mask_points = mask.xy[0].astype(int)
            cv2.fillPoly(left_img, [mask_points], (0, 0, 255, 100))
    
    right_img = img.copy()
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(right_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(right_img, f"{model.names[int(cls)]} {conf:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    metrics = calculate_segmentation_metrics(results, img.shape)
    
    return (
        Image.fromarray(left_img),
        Image.fromarray(right_img),
        metrics
    )

# Streamlit UI
st.title("🧪 Advanced Segmentation Metrics Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(input_img, caption="Input Image", use_column_width=True)
    
    if st.button("Analyze", type="primary"):
        seg_img, det_img, metrics = process_image(input_img)
        
        tab1, tab2 = st.tabs(["Visual Results", "Detailed Metrics"])
        
        with tab1:
            st.subheader("Segmentation Mask")
            st.image(seg_img, use_column_width=True)
            
            st.subheader("Detection Result")
            st.image(det_img, use_column_width=True)
        
        with tab2:
            st.subheader("Segmentation Metrics")
            st.json(metrics)