# Perceptra AI: Advanced Multi-Modal Vision System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A comprehensive multi-modal AI system integrating computer vision and natural language processing through state-of-the-art deep learning architectures.**

Perceptra AI combines advanced image captioning, instance segmentation, and real-time video processing capabilities into a unified system with enterprise-grade security features. Built with custom neural architectures and production-ready deployment considerations.

## 🎥 Project Demo: Image Captioning + Segmentation

https://github.com/user-attachments/assets/d5d4e279-842e-4d98-a7b3-b6760679eeb4

*Complete walkthrough demonstrating image captioning, instance segmentation and biometric authentication features*


## Project Demo: Video Captioning
https://github.com/Varsha-1605/Real-Time-Video-Captioning-Project


## 🌟 Key Features

### 🖼️ **Intelligent Image Captioning**
- Custom ResNet50-LSTM encoder-decoder architecture with Bahdanau attention mechanism
- Trained on MS COCO dataset (200,000 images) achieving competitive BLEU-4 scores (0.105)
- Advanced attention visualization showing model focus areas
- Support for batch processing and API integration

### 🎥 **Real-Time Video Captioning**
- BLIP (Bootstrapping Language-Image Pre-training) model integration
- Sub-second processing capabilities with continuous caption generation
- WebSocket-based real-time communication for live video streams
- Intelligent scene change detection for optimized caption updates
- Frame buffering and efficient tensor operations for smooth performance

### 🎯 **Precision Instance Segmentation**
- YOLOv8x-seg integration for state-of-the-art object detection and segmentation
- Support for 80 COCO object categories with high-quality mask generation
- Real-time inference with confidence scoring and class labeling
- Interactive visualization with overlay masks and bounding boxes

### 🔒 **Advanced Security & Authentication**
- **Dual Authentication System**: Traditional login + Facial Recognition
- Face encoding with 128-dimensional embeddings using dlib
- Secure session management with encrypted face data storage
- 97% traditional login accuracy, 94% facial recognition accuracy

### 🌐 **Production-Ready Web Application**
- Responsive Flask web application with modern UI/UX
- Cross-device compatibility with mobile optimization
- Drag-and-drop file upload with progress indicators
- Real-time processing feedback and error handling
- WCAG 2.1 accessibility compliance

## 📊 Performance Metrics

Our models achieve competitive performance on industry-standard benchmarks:

| Metric | Score | Industry Benchmark | Assessment |
|--------|-------|-------------------|------------|
| **BLEU-4** | 0.1049 | 0.08-0.12 | ✅ Meets Expectations |
| **METEOR** | 0.3158 | 0.28-0.32 | ⭐ Excellent |
| **CIDEr** | 1.0275 | 0.9-1.1 | 🔥 Strong |
| **ROUGE-L** | 0.3836 | 0.35-0.40 | ✅ Good |
| **Perplexity** | 12.43 | 10-15 | ✅ Acceptable |

### Training Performance
- **Total Training Time**: ~42 hours (30 epochs)
- **Convergence**: Achieved by epoch 21 with early stopping
- **Loss Reduction**: 44.8% improvement from initial training
- **GPU Utilization**: 85-95% efficiency

### Real-Time Processing
- **Video Frame Rate**: 5-15 FPS (content-dependent)
- **Caption Latency**: 2-second intervals
- **Memory Usage**: ~800MB RAM, ~2GB GPU
- **System Uptime**: >99% during extended operation

## 🏗️ System Architecture

```mermaid
graph TD
    A[Input Sources] --> B[Core Processing Layer]
    A --> C[Static Images]
    A --> D[Live Video Stream]
    A --> E[User Authentication]
    
    B --> F[Image Captioning Module]
    B --> G[Instance Segmentation Module]
    B --> H[Real-time Video Module]
    
    F --> I[ResNet50 Encoder]
    F --> J[LSTM Decoder + Attention]
    
    G --> K[YOLOv8x-seg Model]
    
    H --> L[BLIP Model]
    H --> M[WebSocket Communication]
    
    I --> N[Output Layer]
    J --> N
    K --> N
    L --> N
    M --> N
    
    N --> O[Generated Captions]
    N --> P[Segmentation Masks]
    N --> Q[Real-time Video Feed]
    N --> R[Web Dashboard]
```

### Component Details

**1. Image Captioning Pipeline**
- **Encoder**: Pre-trained ResNet50 with 14×14×2048 feature maps
- **Decoder**: Single-layer LSTM (256 hidden units) with visual attention
- **Attention**: Bahdanau-style additive attention with visual sentinel gating
- **Vocabulary**: 14,030 unique words with frequency filtering

**2. Real-Time Video Processing**
- **Model**: BLIP architecture with optimized inference pipeline
- **Communication**: Socket.IO for bi-directional real-time updates
- **Optimization**: Automatic Mixed Precision (AMP) for efficient GPU usage
- **Scene Detection**: Intelligent threshold-based caption update triggers

**3. Instance Segmentation**
- **Architecture**: YOLOv8x-seg single-stage detector
- **Post-processing**: NMS with configurable confidence/IoU thresholds
- **Visualization**: Interactive mask overlays with class labels

## 📁 Project Structure

```
IMAGE-CAPTIONING-AND-SEGMENTATION/
├── 📁 data/                          # Dataset storage
│   ├── 📁 images/
│   │   ├── 📁 annotations/           # COCO annotation files
│   │   ├── 📁 train2017/            # Training images
│   │   └── 📁 val2017/              # Validation images
│   └── 📁 processed/                # Preprocessed data
├── 📁 models/                       # Trained model weights
│   ├── 📄 best_model.pth           # Best captioning model
│   ├── 📄 yolov8x-seg.pt          # Segmentation model
│   └── 📄 encoder_state.pth        # Encoder weights
├── 📁 src/                         # Core source code
│   ├── 📄 __init__.py
│   ├── 📄 app.py                   # Main Flask application
│   ├── 📄 model.py                 # Neural network architectures
│   ├── 📄 train.py                 # Training pipeline
│   ├── 📄 evaluation.py            # Evaluation metrics
│   ├── 📄 inference_api.py         # API endpoints
│   ├── 📄 data_preprocessing.py    # Data processing utilities
│   ├── 📄 utils.py                 # Helper functions
│   └── 📄 config.py                # Configuration settings
├── 📁 face_auth_app/               # Facial authentication system
├── 📁 static/                      # Web assets (CSS, JS, images)
├── 📁 templates/                   # HTML templates
│   ├── 📄 index.html              # Main dashboard
│   ├── 📄 auth.html               # Authentication page
│   └── 📄 video.html              # Real-time video interface
├── 📁 output/                      # Generated outputs
│   ├── 📄 captions.json           # Generated captions
│   ├── 📄 metrics.json            # Performance metrics
│   ├── 📄 training.log            # Training logs
│   └── 📄 vocabulary.pkl          # Vocabulary mappings
├── 📄 web_app.py                   # Web application entry point
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # This file
└── 📄 .gitignore                   # Git ignore patterns
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **CUDA-capable GPU** (recommended for training/inference)
- **Git** for repository cloning
- **Webcam** (for real-time video captioning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Varsha-1605/Image-Captioning-and-Segmentation.git
   cd Image-Captioning-and-Segmentation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download MS COCO 2017 Dataset**
   ```bash
   # Create data directory structure
   mkdir -p data/images/annotations
   mkdir -p data/images/train2017
   mkdir -p data/images/val2017
   
   # Download dataset (you can use wget or download manually)
   # Training images: ~13GB
   # Validation images: ~1GB
   # Annotations: ~240MB
   ```

   Place the downloaded files as follows:
   ```
   data/images/
   ├── annotations/
   │   ├── captions_train2017.json
   │   └── captions_val2017.json
   ├── train2017/          # 118,287 training images
   └── val2017/            # 5,000 validation images
   ```

### Training (Optional)

To train your own model from scratch:

```bash
# Configure training parameters in src/config.py
python -m src.train

# Monitor training progress
tail -f output/training.log
```

**Training Configuration:**
- **Batch Size**: 64 (adjust based on GPU memory)
- **Learning Rate**: 4×10⁻⁴ (decoder), 1×10⁻⁵ (encoder)
- **Expected Training Time**: ~42 hours on single GPU
- **Memory Requirements**: 10-12GB GPU RAM

### Running the Application

1. **Start the web server**
   ```bash
   python web_app.py
   ```

2. **Access the application**
   - Open browser to `http://127.0.0.1:5000`
   - Create account or login with facial recognition
   - Upload images for captioning and segmentation
   - Access real-time video captioning at `/video`

## 🖥️ Usage Examples

### Image Captioning & Segmentation

1. **Upload Image**: Navigate to the main dashboard and upload an image
2. **Generate Caption**: The system automatically generates a descriptive caption
3. **View Segmentation**: Object masks and labels are displayed with confidence scores
4. **Download Results**: Export captions and segmentation data in JSON format

### Real-Time Video Captioning

1. **Access Video Mode**: Navigate to `/video` endpoint
2. **Allow Camera**: Grant webcam permissions when prompted
3. **Live Captions**: View real-time captions updating every 2 seconds
4. **Scene Analysis**: Captions adapt to scene changes automatically

### API Usage

```python
import requests

# Image captioning API
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/caption', 
                           files={'image': f})
    caption = response.json()['caption']

# Segmentation API
response = requests.post('http://localhost:5000/api/segment',
                       files={'image': f})
results = response.json()
```

## 🔧 Configuration

Key configuration parameters in `src/config.py`:

```python
# Model Configuration
IMAGE_SIZE = 224                    # Input image resolution
EMBED_SIZE = 256                    # Word embedding dimension  
HIDDEN_SIZE = 256                   # LSTM hidden units
VOCAB_SIZE = 14030                  # Vocabulary size
MAX_CAPTION_LENGTH = 20             # Maximum caption words

# Training Configuration
BATCH_SIZE = 64                     # Training batch size
LEARNING_RATE = 4e-4               # Base learning rate
EPOCHS = 30                        # Maximum training epochs
GRAD_CLIP = 5.0                    # Gradient clipping threshold

# Real-time Video Configuration
FRAME_SKIP = 2                     # Process every Nth frame
SCENE_CHANGE_THRESHOLD = 0.05      # Scene change sensitivity
USE_AMP = True                     # Automatic Mixed Precision
```

## 🧪 Evaluation & Testing

### Running Evaluations

```bash
# Evaluate trained model on validation set
python -m src.evaluation --model_path models/best_model.pth

# Generate attention visualizations
python -m src.utils --visualize_attention

# Test API endpoints
python -m pytest tests/ -v
```

### Performance Monitoring

The system includes comprehensive logging and metrics collection:

- **Training Metrics**: Loss curves, BLEU scores, convergence analysis
- **Inference Metrics**: Processing time, memory usage, accuracy scores
- **System Metrics**: API response times, error rates, user sessions

## 🛠️ Technical Details

### Neural Architecture Innovations

**Custom Attention Mechanism:**
```python
# Visual sentinel gating for dynamic attention control
e_ij = f_att(s_{i-1}, h_j)          # Attention energy
α_ij = softmax(e_ij)                # Attention weights  
c_i = Σ(α_ij × h_j)                 # Context vector
```

**Optimization Strategies:**
- Differential learning rates for encoder/decoder
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling with plateau detection
- Early stopping with validation monitoring

### Real-Time Processing Optimizations

- **Frame Buffering**: Efficient sliding window for temporal consistency
- **Async Processing**: Non-blocking inference with thread pools  
- **Memory Management**: Automatic cleanup and garbage collection
- **Error Recovery**: Graceful degradation and reconnection handling

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure backwards compatibility

## 🙏 Acknowledgments

- **MS COCO Dataset**: For providing comprehensive image-caption pairs
- **PyTorch Team**: For the excellent deep learning framework
- **Ultralytics**: For the YOLOv8 implementation
- **Hugging Face**: For BLIP model and transformers library
- **Flask Community**: For the robust web framework

## 📧 Contact & Support

**Varsha Dewangan**
- 📧 Email: varshadewangan1605@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/varsha-dewangan-197983256/
- 🐱 GitHub: [Varsha-1605](https://github.com/Varsha-1605)

### Get Help

- 🐛 **Bug Reports**: Open an issue with detailed reproduction steps
- 💡 **Feature Requests**: Describe your use case and proposed solution

---

<div align="center">
  <p><strong>⭐ Star this repository if you find it helpful!</strong></p>
  <p>Built with ❤️ using PyTorch, Flask, and cutting-edge AI research</p>
</div>
