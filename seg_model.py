import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

# Load YOLOv8 segmentation model
model = torch.hub.load('ultralytics/yolov8', 'yolov8x-seg')

def format_detections(results):
    """Format detections in your specified style"""
    detections = []
    for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
        class_name = model.names[int(cls)]
        detections.append(f"{class_name} {conf:.2f}")
    
    # Format to match your example layout
    formatted = ""
    for i in range(0, len(detections), 3):
        line = "  ".join(detections[i:i+3])
        formatted += line + "\n"
    return formatted.strip()

def analyze_image(input_img):
    img = np.array(input_img)
    results = model(img)
    
    # Left Panel: Instance Segmentation
    seg_plot = results[0].plot(conf=True, labels=True, boxes=True)
    seg_plot = cv2.cvtColor(seg_plot, cv2.COLOR_BGR2RGB)
    
    # Right Panel: Formatted detections
    detections_text = format_detections(results)
    
    # Metrics
    metrics = {
        "Total Objects": len(results[0].boxes),
        "Average Confidence": f"{results[0].boxes.conf.mean():.2f}",
        "Class Distribution": {model.names[int(cls)]: int(sum(results[0].boxes.cls == cls)) 
                             for cls in torch.unique(results[0].boxes.cls)}
    }
    
    return (
        Image.fromarray(seg_plot),  # Left
        detections_text,            # Right-top
        metrics                     # Right-bottom
    )

with gr.Blocks() as demo:
    gr.Markdown("# Instance Segmentation Analyzer")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil")
        
        with gr.Column():
            with gr.Row():
                left_output = gr.Image(label="Segmentation Result")
                right_output = gr.Textbox(label="Detections", lines=6)
            metrics_output = gr.Json(label="Segmentation Metrics")

    img_input.change(
        fn=analyze_image,
        inputs=img_input,
        outputs=[left_output, right_output, metrics_output]
    )

demo.launch()