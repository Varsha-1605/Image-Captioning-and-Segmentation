import os
import torch
from PIL import Image
import sys # Import sys for flushing stdout

# Import modules from your project structure
# Ensure utils is imported early to set up logging
from .utils import get_logger, get_eval_transform, visualize_attention
from .model import ImageCaptioningModel
from .data_preprocessing import COCOVocabulary
from .config import INFERENCE_CONFIG, update_config_with_latest_model # config is imported here
from .evaluation import calculate_bleu_scores_detailed # evaluation is imported here

# Get the module-specific logger. This logger will inherit from the root logger
# which is configured when `utils` is imported.
logger = get_logger(__name__)

def run_inference_example(model_path, image_path, config=None):
    """
    Function to run inference on a single image and generate a caption.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth file).
        image_path (str): Path to the image file for captioning.
        config (dict, optional): Configuration dictionary for inference parameters
                                 (e.g., beam_size, max_caption_length).
    Returns:
        str: The generated caption for the image.
    Raises:
        FileNotFoundError: If the model checkpoint or image file is not found.
        Exception: For other unexpected errors during inference.
    """
    logger.info("Loading model for inference...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. "
                                "Please train the model first or provide a valid path.")

    # Load the complete checkpoint (model state, optimizer state, vocabulary, config)
    # map_location='cpu' ensures it loads to CPU even if trained on GPU
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract configuration and vocabulary from the checkpoint
    model_config_from_checkpoint = checkpoint.get('config', {})
    vocabulary = checkpoint['vocabulary']

    # Initialize the model structure with parameters saved in the checkpoint
    # Ensure dropout is set to 0.0 for inference and fine_tune_encoder is False
    model = ImageCaptioningModel(
        vocab_size=vocabulary.vocab_size,
        embed_dim=model_config_from_checkpoint.get('embed_dim', 256),
        attention_dim=model_config_from_checkpoint.get('attention_dim', 256),
        decoder_dim=model_config_from_checkpoint.get('decoder_dim', 256),
        dropout=0.0, # Dropout should be off during inference
        fine_tune_encoder=False, # Encoder should not be fine-tuned during inference
        max_caption_length=config.get('max_caption_length', 20) if config else 20 # Use config's max length for inference
    )
    # Load the trained weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set the model to evaluation mode (important for batch norm, dropout)

    # Determine the device to run inference on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) # Move model to GPU if available
    logger.info(f"Model loaded successfully on device: {device}")

    # Get the image transformation pipeline for evaluation/inference
    transform = get_eval_transform()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the image path.")

    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).to(device)
    except Exception as e:
        raise Exception(f"Error loading or transforming image {image_path}: {e}")

    logger.info(f"Generating caption for {image_path} using beam search (beam_size="
                f"{config.get('beam_size', 5) if config else 5})...")

    # Generate the caption using the model's integrated method
    generated_caption = model.generate_caption(
        image_tensor,
        vocabulary,
        device,
        beam_size=config.get('beam_size', 5) if config else 5,
        max_length=config.get('max_caption_length', 20) if config else 20
    )

    # Optional: Visualize attention weights
    visualize_attention(model, image_path, vocabulary, device,
                        save_path=os.path.join('output', 'attention_visualization.png'))

    # These logs are now placed AFTER the point where the logger is definitely active.
    logger.info("\n" + "="*50)
    logger.info("             GENERATED CAPTION")
    logger.info("="*50)
    logger.info(f"Image: {image_path}")
    logger.info(f"Caption: {generated_caption}")
    logger.info("="*50 + "\n")
    sys.stdout.flush() # Explicitly flush the standard output buffer

    return generated_caption


if __name__ == '__main__':
    # When `app.py` is run directly, it will run the inference example.
    # Update INFERENCE_CONFIG with the latest model path if available
    update_config_with_latest_model(INFERENCE_CONFIG)

    # --- User Input/Configuration for Inference ---
    # These values are now primarily controlled via INFERENCE_CONFIG in config.py
    # You can override them here if you need to test specific scenarios immediately.
    my_image_path = INFERENCE_CONFIG['example_image_path']
    my_model_path = INFERENCE_CONFIG['model_path']
    # You can also set a reference caption here if you know it for comparison
    my_reference_caption = "Two riders on horses are performing a reining maneuver on a green grassy field surrounded by trees" # Example reference, replace or leave empty

    # Use a copy of INFERENCE_CONFIG to avoid modifying the global config directly
    inference_params = INFERENCE_CONFIG.copy()

    logger.info("--- Running Inference Example ---")
    try:
        generated_caption = run_inference_example(my_model_path, my_image_path, config=inference_params)

        # You can add evaluation of this single generated caption against its reference here if desired
        if my_reference_caption:
            # calculate_bleu_scores_detailed is already imported from evaluation
            bleu_scores = calculate_bleu_scores_detailed([my_reference_caption], [generated_caption])
            logger.info("\n--- Single Image Evaluation ---")
            logger.info(f"Reference: {my_reference_caption}")
            logger.info(f"Generated: {generated_caption}")
            logger.info(f"BLEU-4 Score: {bleu_scores['BLEU-4']:.4f}")
            logger.info("-------------------------------\n")
            sys.stdout.flush() # Explicitly flush after single image evaluation too

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please ensure your model, vocabulary, and image paths are correct "
                     "and data is downloaded as per README.md.")
        sys.stdout.flush() # Flush errors too
    except Exception as e:
        logger.critical(f"An unexpected error occurred during inference: {e}", exc_info=True)
        sys.stdout.flush() # Flush critical errors too
