import torch
import torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained models once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
maskrcnn.eval()
deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
deeplab.eval()

def process_image(image_path, save_path=None):
    pil_image = Image.open(image_path).convert("RGB")
    img_cv = np.array(pil_image)
    # Caption
    inputs = processor(pil_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    # Instance Segmentation
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = transform(pil_image).to(device)
    pred_instance = maskrcnn([image])[0]
    instance_mask_display = np.zeros_like(img_cv, dtype=np.uint8)
    num_instances = len(pred_instance["masks"])
    colors = [
        [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 0, 128], [128, 128, 0], [0, 128, 128], [128, 128, 128]
    ]
    img_with_boundaries = img_cv.copy()
    object_features = []
    for j in range(num_instances):
        score = pred_instance["scores"][j].item()
        if score < 0.7:
            continue
        mask = pred_instance["masks"][j, 0].detach().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)
        label_id = pred_instance["labels"][j].item()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_with_boundaries, contours, -1, (255, 0, 0), 2)
        area = int(np.sum(mask))
        y_indices, x_indices = np.where(mask)
        bbox = [0, 0, 0, 0]
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            bbox = [x_min, y_min, x_max, y_max]
        object_features.append({
            'label_id': label_id,
            'score': float(score),
            'area': area,
            'bbox': bbox
        })
        for c in range(3):
            instance_mask_display[:, :, c] = np.where(mask == 1, colors[j % len(colors)][c], instance_mask_display[:, :, c])
    # Save result image
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_with_boundaries, cv2.COLOR_RGB2BGR))
    return caption, object_features, save_path

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, frame)
        caption, features, _ = process_image(temp_path)
        processed_frame = cv2.imread(temp_path)
        if out is None:
            h, w, _ = processed_frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        out.write(processed_frame)
        frame_count += 1
    cap.release()
    out.release()
    os.remove('temp_frame.jpg')
    return output_path, frame_count

def process_webcam_frame(frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_path = 'temp_webcam.jpg'
    cv2.imwrite(temp_path, frame)
    caption, features, _ = process_image(temp_path)
    processed_frame = cv2.imread(temp_path)
    _, buffer = cv2.imencode('.jpg', processed_frame)
    os.remove(temp_path)
    return buffer.tobytes(), caption, features