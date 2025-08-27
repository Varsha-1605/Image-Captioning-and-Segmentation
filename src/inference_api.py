import os
import torch
from PIL import Image

# Import core model components and utilities using ABSOLUTE IMPORTS
# Since 'src' is added to sys.path, we refer to modules directly under 'src'.
from src.model import ImageCaptioningModel
from src.data_preprocessing import COCOVocabulary
from src.utils import get_logger, get_eval_transform
from src.config import INFERENCE_CONFIG, update_config_with_latest_model # Import global config

logger = get_logger(__name__)

# --- Global variables to store the loaded model and vocabulary ---
# These will be loaded once when this module is first imported.
_model = None
_vocabulary = None
_device = None
_transform = None

def _load_model_and_vocabulary():
    """
    Loads the image captioning model and vocabulary.
    This function should be called only once during application startup.
    """
    global _model, _vocabulary, _device, _transform

    if _model is not None:
        logger.info("Model and vocabulary already loaded.")
        return

    logger.info("Initializing model and vocabulary for web inference...")

    # Update INFERENCE_CONFIG with the path to the latest best model
    # This ensures the web app uses the correct trained model.
    update_config_with_latest_model(INFERENCE_CONFIG)
    model_path = INFERENCE_CONFIG['model_path']
    example_image_path = INFERENCE_CONFIG['example_image_path'] # Not directly used for inference, but useful for context

    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found at {model_path}. "
                     "Please ensure the model is trained and saved.")
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    try:
        # Load the complete checkpoint (model state, vocabulary, config)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Extract configuration and vocabulary from the checkpoint
        model_config_from_checkpoint = checkpoint.get('config', {})
        _vocabulary = checkpoint['vocabulary']

        # Initialize the model structure with parameters saved in the checkpoint
        _model = ImageCaptioningModel(
            vocab_size=_vocabulary.vocab_size,
            embed_dim=model_config_from_checkpoint.get('embed_dim', 256),
            attention_dim=model_config_from_checkpoint.get('attention_dim', 256),
            decoder_dim=model_config_from_checkpoint.get('decoder_dim', 256),
            dropout=0.0, # Dropout should be off during inference
            fine_tune_encoder=False, # Encoder should not be fine-tuned during inference
            max_caption_length=INFERENCE_CONFIG.get('max_caption_length', 20)
        )
        # Load the trained weights into the model
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval() # Set the model to evaluation mode (important for batch norm, dropout)

        # Determine the device to run inference on
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model = _model.to(_device) # Move model to GPU if available
        logger.info(f"Model loaded successfully on device: {_device}")

        # Get the image transformation pipeline for evaluation/inference
        _transform = get_eval_transform()

        logger.info("Model and vocabulary are ready for inference.")

    except Exception as e:
        logger.critical(f"Failed to load model or vocabulary: {e}", exc_info=True)
        # Reraise the exception to prevent the Flask app from starting without the model
        raise

# Call the loading function immediately when this module is imported
# This ensures the model is loaded only once when the Flask app starts
_load_model_and_vocabulary()


def generate_caption_for_image(image_path: str) -> str:
    """
    Generates a caption for a given image path using the pre-loaded model.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        str: The generated caption.
    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: For errors during image loading or caption generation.
    """
    if _model is None or _vocabulary is None or _transform is None or _device is None:
        logger.error("Model or vocabulary not loaded. Cannot generate caption.")
        raise RuntimeError("Image captioning model is not initialized.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}.")

    logger.info(f"Processing image: {image_path}")

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = _transform(image).to(_device)
    except Exception as e:
        raise Exception(f"Error loading or transforming image {image_path}: {e}")

    # Generate the caption using the model's integrated method
    generated_caption = _model.generate_caption(
        image_tensor,
        _vocabulary,
        _device,
        beam_size=INFERENCE_CONFIG.get('beam_size', 5),
        max_length=INFERENCE_CONFIG.get('max_caption_length', 20)
    )
    logger.info(f"Generated caption: {generated_caption}")
    return generated_caption
