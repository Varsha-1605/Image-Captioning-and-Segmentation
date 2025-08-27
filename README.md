##Project Demo: Image Captioning + Segmentation


## Project Demo: Video Captioning
https://github.com/Varsha-1605/Real-Time-Video-Captioning-Project




Image Captioning with Attention
This project implements an image captioning model using a ResNet50-based CNN Encoder and an LSTM Decoder with Bahdanau-style Attention. The model is trained on the COCO 2017 dataset to generate descriptive captions for images.

Features
Modular Architecture: Separated concerns into distinct Python files for clarity and maintainability.

CNN Encoder: Utilizes a pre-trained ResNet50 for robust feature extraction.

Attention Mechanism: Allows the model to focus on salient regions of an image when generating specific words.

LSTM Decoder: Generates captions word by word, conditioned on image features and previously generated words.

Beam Search: Implemented for high-quality caption generation during inference.

Comprehensive Evaluation: Calculates standard metrics like BLEU (1-4), METEOR, ROUGE-L, and CIDEr.

Training Resumption: Ability to resume training from the latest checkpoint.

Logging: Detailed logging for training progress and evaluation results.

Project Structure
ImageCaptioning/
├── data/
│   ├── coco/
│   │   ├── train2017/              # COCO 2017 training images
│   │   ├── val2017/                # COCO 2017 validation images
│   │   └── annotations/
│   │       ├── captions_train2017.json # Training captions JSON
│   │       └── captions_val2017.json   # Validation captions JSON
├── models/                         # Directory for saving trained model checkpoints
│   └── (e.g., best_model_bleu0.1037.pth)
├── src/
│   ├── app.py                      # Main script for running inference
│   ├── config.py                   # Configuration parameters for training, eval, inference
│   ├── data_preprocessing.py       # Classes for vocabulary and dataset loading
│   ├── model.py                    # Defines the Encoder, Attention, and Decoder modules
│   ├── train.py                    # Contains the training loop and validation functions
│   ├── evaluation.py               # Functions for calculating various evaluation metrics
│   └── utils.py                    # General utility functions (e.g., logging setup, attention visualization)
├── output/                         # Output directory for logs, saved vocabulary, and evaluation results
├── requirements.txt                # List of Python dependencies
└── README.md                       # This file

Setup
Follow these steps to set up the project locally.

1. Clone the Repository
git clone https://github.com/YourUsername/ImageCaptioning.git # Replace with your repo URL
cd ImageCaptioning

2. Create a Python Virtual Environment (Recommended)
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
Install the required Python packages. Note that pycocoevalcap needs to be installed directly from its GitHub repository as it's not on PyPI.

pip install -r requirements.txt

4. Download NLTK Data
The evaluation metrics (BLEU, METEOR) require NLTK's punkt and wordnet data.

import nltk
nltk.download('punkt')
nltk.download('wordnet')

5. Download COCO 2017 Dataset
You'll need the COCO 2017 dataset.

Images:

train2017.zip (around 18GB): http://images.cocodataset.org/zips/train2017.zip

val2017.zip (around 1GB): http://images.cocodataset.org/zips/val2017.zip

Annotations:

annotations_trainval2017.zip (around 250MB): http://images.cocodataset.org/annotations/annotations_trainval2017.zip

After downloading:

Create the data/coco directory if it doesn't exist.

Extract train2017.zip into data/coco/train2017/.

Extract val2017.zip into data/coco/val2017/.

Extract annotations_trainval2017.zip into data/coco/annotations/. This will create captions_train2017.json and captions_val2017.json (among others).

Your data/coco directory should then look like this:

data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
    └── ... (other annotation files)

Usage
1. Configure Parameters
All configurable parameters are located in src/config.py. Open this file and adjust paths, hyperparameters, and other settings as needed.

data_folder: Ensure this points to the data/coco directory.

model_path: Update this to the path of your trained model (e.g., models/best_model_bleu0.1037.pth if you've already trained one or downloaded a pre-trained model).

2. Train the Model
To start or resume training, run the train.py script:

python src/train.py

Training logs, model checkpoints, and a saved vocabulary file will be stored in the output/ directory. The best model (based on BLEU-4 score) will be saved in the models/ directory.

3. Evaluate the Model
After training, you can evaluate the model's performance on the validation set:

python src/evaluation.py

This will generate captions for the test set and calculate various metrics. The detailed results will be saved in the output/evaluation_results/ directory.

4. Run Inference (Generate Caption for a Single Image)
To generate a caption for a specific image, run the app.py script. Make sure src/config.py has the correct model_path and example_image_path set.

python src/app.py

The generated caption will be printed to the console. You can modify inference_config in src/config.py to change the beam_size or max_caption_length.

Pre-trained Models
(Optional: If you plan to provide pre-trained models, you would include instructions here on how users can download and place them in the models/ directory.)

Future Improvements
Implement a web interface (e.g., using Flask or FastAPI) for interactive captioning.

Explore different CNN backbones (e.g., EfficientNet, Vision Transformers).

Integrate advanced attention mechanisms or transformer architectures.

Add support for more datasets.

Quantization or pruning for model optimization.

License
(Add your chosen license, e.g., MIT, Apache 2.0)
