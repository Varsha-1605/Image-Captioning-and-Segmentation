import os

# Base data folder, assuming COCO 2017 is extracted here
# Adjust this path if your data is located elsewhere
_BASE_DATA_FOLDER = 'data/images'

# Output directory for logs, saved vocabulary, and temporary files
# and where the best model checkpoint will be saved for easy access
_OUTPUT_DIR = 'output'
_MODELS_DIR = 'models' # Where the final best model will be saved

# Ensure output and models directories exist
os.makedirs(_OUTPUT_DIR, exist_ok=True)
os.makedirs(_MODELS_DIR, exist_ok=True)


# --- Configuration for Training ---
TRAINING_CONFIG = {
    'data_folder': _BASE_DATA_FOLDER,
    'train_image_folder': 'train2017',
    'val_image_folder': 'val2017',
    'train_caption_file': os.path.join(_BASE_DATA_FOLDER, 'annotations', 'captions_train2017.json'),
    'val_caption_file': os.path.join(_BASE_DATA_FOLDER, 'annotations', 'captions_val2017.json'),

    # Subset sizes for quicker testing during development. Set to None for full dataset.
    'vocab_subset_size': None,
    'train_subset_size': None, # e.g., 200000 for a large subset, None for full
    'val_subset_size': None,   # e.g., 10000 for a subset, None for full

    # Model Hyperparameters
    'embed_dim': 256,
    'attention_dim': 256,
    'decoder_dim': 256,
    'dropout': 0.5,
    'fine_tune_encoder': True, # Set to False to freeze ResNet weights during training

    # Training Parameters
    'batch_size': 64,
    'num_workers': 4, # Adjust based on your CPU cores and RAM (e.g., 2, 4, 8)
    'learning_rate': 4e-4,
    'encoder_learning_rate': 1e-5, # Lower LR for encoder if fine_tune_encoder is True
    'lr_reduce_factor': 0.5,
    'lr_patience': 5,
    'num_epochs': 30, # Total number of epochs to run
    'log_step': 100,  # Print loss every N steps
    'grad_clip': 5.0, # Gradient clipping value

    'max_caption_length': 30, # Max length of captions, including <START> and <END>
    'val_beam_size': 3,       # Beam size used for validation inference during training
    'val_inference_batches': None, # None to generate captions for all validation batches, or an int for a subset

    'output_dir': _OUTPUT_DIR, # Where training logs and vocabulary will be saved
    'models_dir': _MODELS_DIR  # Where the best model checkpoint will be saved
}


# --- Configuration for Evaluation ---
# This uses the validation set for evaluation, as is common practice.
EVALUATION_CONFIG = {
    'data_folder': _BASE_DATA_FOLDER,
    'test_image_folder': 'val2017', # Typically evaluate on the validation set for final metrics
    'test_caption_file': os.path.join(_BASE_DATA_FOLDER, 'annotations', 'captions_val2017.json'),
    'test_subset_size': None, # Evaluate on a subset for faster testing, or None for full validation set
    'eval_batch_size': 1,     # Must be 1 for accurate beam search evaluation
    'beam_size': 5,           # Beam size for caption generation during evaluation
    'max_caption_length': 30,
    'num_workers': 4,
    'eval_output_dir': os.path.join(_OUTPUT_DIR, 'evaluation_results'), # Directory to save evaluation results JSONs
    'output_dir': _OUTPUT_DIR,

    # Placeholder for model path. This will be updated dynamically after training or
    # can be set manually if you have a pre-trained model.
    'model_path': os.path.join(_MODELS_DIR, 'best_model_bleu0.1058.pth') # Placeholder, update after training
}


# --- Configuration for Inference Example ---
INFERENCE_CONFIG = {
    'beam_size': 5,
    'max_caption_length': 30,
    # Placeholder for model path. This will be updated dynamically after training or
    # can be set manually if you have a pre-trained model.
    'model_path': os.path.join(_MODELS_DIR, 'models/best_model_bleu0.1058.pth'), # Placeholder, update after training

    # Path to an example image for quick inference demonstration
    'example_image_path': os.path.join(_BASE_DATA_FOLDER, 'new_one.jpg') # Example image from COCO val2017
}


# --- Utility Functions for updating config with latest trained model ---
def update_config_with_latest_model(config_dict):
    """
    Finds the latest best model checkpoint in the models directory and updates
    the given configuration dictionary's 'model_path'.
    """
    saved_models = [f for f in os.listdir(_MODELS_DIR) if f.startswith('best_model_bleu') and f.endswith('.pth')]
    if saved_models:
        # Get the one with the highest BLEU score in its name
        latest_model_name = max(saved_models, key=lambda f: float(f.split('bleu')[1].replace('.pth', '')))
        latest_model_path = os.path.join(_MODELS_DIR, latest_model_name)
        config_dict['model_path'] = latest_model_path
        print(f"Updated config with latest model: {latest_model_path}")
    else:
        print(f"Warning: No best model found in '{_MODELS_DIR}'. Inference/Evaluation might fail.")

# Update inference and evaluation configs to point to the latest model if available
update_config_with_latest_model(EVALUATION_CONFIG)
update_config_with_latest_model(INFERENCE_CONFIG)
