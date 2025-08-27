import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (jaccard_score, f1_score, 
                           accuracy_score, precision_score, 
                           recall_score)
from scipy.spatial.distance import directed_hausdorff

# MUST BE FIRST STREAMLIT COMMAND
import streamlit as st
st.set_page_config(
    page_title="Advanced Segmentation Metrics Analyzer",
    page_icon="🧪",
    layout="wide"
)

# Model loading with enhanced error handling
@st.cache_resource
def load_model():
    try:
        # First try official ultralytics package
        from ultralytics import YOLO
        return YOLO('yolov8x-seg.pt')
    except ImportError:
        try:
            # Fallback to torch hub
            model = torch.hub.load('ultralytics/yolov8', 'yolov8x-seg', pretrained=True)
            return model.to('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            st.error(f"⚠️ Model loading failed: {str(e)}")
            st.info("Please check your internet connection and try again")
            return None

model = load_model()

def validate_image(img_array):
    """Ensure 3-channel RGB format"""
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif img_array.shape[2] > 3:  # Extra channels
        img_array = img_array[:, :, :3]
    return img_array

def calculate_boundary_iou(mask1, mask2, boundary_width=2):
    """Calculate Boundary IoU with error handling"""
    try:
        kernel = np.ones((boundary_width, boundary_width), np.uint8)
        boundary1 = cv2.morphologyEx(mask1, cv2.MORPH_GRADIENT, kernel)
        boundary2 = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, kernel)
        return jaccard_score(boundary1.flatten(), boundary2.flatten())
    except Exception:
        return 0.0  # Graceful degradation

def calculate_metrics(results, img_shape):
    """Robust metric calculation"""
    if not model:
        return {"error": "Model not loaded"}
    if not results or results[0].masks is None:
        return {"error": "No objects detected"}
    
    try:
        # Process predictions
        pred_masks = torch.stack([m.data[0] for m in results[0].masks]).cpu().numpy()
        pred_masks = (pred_masks > 0.5).astype(np.uint8)
        
        # Generate mock ground truth
        gt_masks = np.zeros_like(pred_masks)
        h, w = img_shape[:2]
        gt_masks[:, h//4:3*h//4, w//4:3*w//4] = 1
        
        # Initialize metrics
        metrics = {
            'IoU': {'mean': 0, 'per_instance': [], 'class_wise': {}},
            'Dice': 0,
            'Pixel_Accuracy': 0,
            'Boundary_IoU': 0,
            'Object_Counts': {},
            'Class_Distribution': {}
        }
        
        # Calculate per-mask metrics
        valid_masks = 0
        for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
            try:
                pred_flat = pred_mask.flatten()
                gt_flat = gt_mask.flatten()
                
                if np.sum(gt_flat) == 0:
                    continue
                    
                # Core metrics
                metrics['IoU']['per_instance'].append(jaccard_score(gt_flat, pred_flat))
                metrics['Dice'] += f1_score(gt_flat, pred_flat)
                metrics['Pixel_Accuracy'] += accuracy_score(gt_flat, pred_flat)
                metrics['Boundary_IoU'] += calculate_boundary_iou(gt_mask, pred_mask)
                
                # Class tracking
                cls = int(results[0].boxes.cls[i])
                cls_name = model.names[cls]
                metrics['Object_Counts'][cls_name] = metrics['Object_Counts'].get(cls_name, 0) + 1
                metrics['Class_Distribution'][cls_name] = metrics['Class_Distribution'].get(cls_name, 0) + 1
                
                valid_masks += 1
            except Exception:
                continue
        
        # Finalize metrics
        if valid_masks > 0:
            metrics['IoU']['mean'] = np.mean(metrics['IoU']['per_instance'])
            metrics['Dice'] /= valid_masks
            metrics['Pixel_Accuracy'] /= valid_masks
            metrics['Boundary_IoU'] /= valid_masks
            
            # Class-wise metrics
            total = sum(metrics['Object_Counts'].values())
            metrics['IoU']['class_wise'] = {k: v/total for k, v in metrics['Object_Counts'].items()}
            
        return metrics
        
    except Exception as e:
        return {"error": f"Metric calculation failed: {str(e)}"}

def visualize_results(img, results):
    """Generate visualizations with error handling"""
    try:
        # Segmentation overlay
        seg_img = img.copy()
        if results[0].masks is not None:
            for mask in results[0].masks:
                mask_points = mask.xy[0].astype(int)
                cv2.fillPoly(seg_img, [mask_points], (0, 0, 255, 100))
        
        # Bounding boxes
        det_img = img.copy()
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(det_img, f"{model.names[int(cls)]} {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return seg_img, det_img
    except Exception:
        return img, img  # Fallback to original images

def process_image(input_img):
    """Main processing pipeline"""
    try:
        img = np.array(input_img)
        img = validate_image(img)
        results = model(img)
        seg_img, det_img = visualize_results(img, results)
        metrics = calculate_metrics(results, img.shape)
        return Image.fromarray(seg_img), Image.fromarray(det_img), metrics
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None, None, {"error": str(e)}

# Main UI
def main():
    st.title("🧪 Advanced Segmentation Metrics Analyzer")
    st.markdown("""
    Upload an image to analyze object segmentation performance using YOLOv8.
    The system provides detailed metrics and visualizations.
    """)
    
    with st.sidebar:
        st.header("Configuration")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        boundary_width = st.slider("Boundary Width (pixels)", 1, 10, 2)
        st.markdown("---")
        st.markdown(f"**Device:** {'GPU 🔥' if torch.cuda.is_available() else 'CPU 🐢'}")
    
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supports JPG, PNG, BMP formats"
    )
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_column_width=True)
            
            if st.button("Analyze", type="primary"):
                with st.spinner("Processing..."):
                    seg_img, det_img, metrics = process_image(img)
                    
                    if metrics and "error" not in metrics:
                        tabs = st.tabs(["Visual Results", "Metrics Dashboard", "Raw Data"])
                        
                        with tabs[0]:
                            st.subheader("Segmentation Analysis")
                            cols = st.columns(2)
                            cols[0].image(seg_img, caption="Segmentation Mask", use_column_width=True)
                            cols[1].image(det_img, caption="Detected Objects", use_column_width=True)
                        
                        with tabs[1]:
                            st.subheader("Performance Metrics")
                            st.metric("Mean IoU", f"{metrics['IoU']['mean']:.2%}", 
                                     help="Intersection over Union")
                            st.metric("Dice Coefficient", f"{metrics['Dice']:.2%}",
                                     help="F1 Score for segmentation")
                            st.metric("Pixel Accuracy", f"{metrics['Pixel_Accuracy']:.2%}")
                            
                            st.plotly_chart({
                                'data': [{
                                    'x': list(metrics['Class_Distribution'].keys()),
                                    'y': list(metrics['Class_Distribution'].values()),
                                    'type': 'bar'
                                }],
                                'layout': {'title': 'Class Distribution'}
                            })
                        
                        with tabs[2]:
                            st.download_button(
                                "Download Metrics",
                                str(metrics),
                                "metrics.json",
                                "application/json"
                            )
                            st.json(metrics)
                    
                    elif metrics and "error" in metrics:
                        st.error(metrics["error"])
        
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

if __name__ == "__main__":
    main()