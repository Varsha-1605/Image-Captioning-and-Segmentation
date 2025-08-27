import os
import time
import math
import pickle
import gc # For memory optimization

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from .model import ImageCaptioningModel # Import the model
from .data_preprocessing import COCODataset, COCOVocabulary # Import data handling classes
from .evaluation import calculate_bleu_scores_detailed # Import evaluation metric
from .utils import get_logger, get_train_transform, get_eval_transform # Import utilities

logger = get_logger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """
    Performs a single training epoch.

    Args:
        model (nn.Module): The image captioning model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on (cpu/cuda).
        epoch (int): Current epoch number (0-indexed).
        config (dict): Configuration dictionary.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train() # Set model to training mode
    running_loss = 0.0
    start_time = time.time()
    total_batches = len(train_loader)

    # Use tqdm for a progress bar
    for i, (images, captions, lengths, _) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        # Forward pass
        # scores: (batch_size, max_decode_length_in_batch, vocab_size)
        # caps_sorted: (batch_size, max_padded_length_from_dataset)
        # decode_lengths: list of actual lengths for current batch (after sorting)
        scores, caps_sorted, decode_lengths, _, _ = model(images, captions, lengths)

        # Prepare targets for loss calculation
        # Pack scores to remove padding and ensure correct length for loss calculation.
        # This matches the dynamic lengths of the sequences.
        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data

        # Slice targets to match the length of scores_packed, removing the <START> token.
        # The target sequence is `caption[1:]` because the model predicts the word
        # at `t+1` given `caption[t]`.
        targets = caps_sorted[:, 1:] # Remove <START> token from targets
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores_packed, targets_packed)

        # Backward pass and optimize
        optimizer.zero_grad() # Clear gradients from previous step
        loss.backward() # Compute gradients
        
        # Gradient clipping to prevent exploding gradients, especially common in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 5.0))
        
        optimizer.step() # Update model parameters

        running_loss += loss.item()

        # Log training progress periodically
        if (i + 1) % config.get('log_step', 100) == 0:
            current_loss = loss.item()
            perplexity = math.exp(current_loss) if current_loss < float('inf') else float('inf')
            logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{i+1}/{total_batches}], "
                        f"Loss: {current_loss:.4f}, Perplexity: {perplexity:.4f}")

    epoch_loss = running_loss / total_batches
    epoch_time = time.time() - start_time
    logger.info(f"Epoch {epoch+1} Training finished. Avg Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
    return epoch_loss


def validate_epoch(model, val_loader, criterion, vocabulary, device, config):
    """
    Performs a single validation epoch.
    Generates captions for a subset of the validation set to calculate BLEU scores.

    Args:
        model (nn.Module): The image captioning model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (used for validation loss).
        vocabulary (COCOVocabulary): Vocabulary object, used for converting indices to words.
        device (torch.device): Device to run validation on (cpu/cuda).
        config (dict): Configuration dictionary.

    Returns:
        tuple: (Average validation loss, list of generated captions, list of reference captions)
    """
    model.eval() # Set model to evaluation mode
    val_running_loss = 0.0
    val_generated_captions = []
    val_reference_captions = []

    with torch.no_grad(): # Disable gradient calculations for validation
        total_batches = len(val_loader)
        # Iterate through the validation loader for loss calculation and caption generation
        for i, (images, captions, lengths, _) in enumerate(val_loader):
            images = images.to(device)
            val_captions_for_loss = captions.to(device)
            val_lengths_for_loss = lengths.to(device)

            # Forward pass for loss calculation (similar to training)
            val_scores, val_caps_sorted, val_decode_lengths, _, _ = model(images, val_captions_for_loss, val_lengths_for_loss)

            val_scores_packed = pack_padded_sequence(val_scores, val_decode_lengths, batch_first=True).data
            val_targets = val_caps_sorted[:, 1:] # Remove <START>
            val_targets_packed = pack_padded_sequence(val_targets, val_decode_lengths, batch_first=True).data

            val_loss = criterion(val_scores_packed, val_targets_packed)
            val_running_loss += val_loss.item()

            # Generate captions using beam search for a subset of batches or all
            # The `val_inference_batches` config parameter controls how many batches to run inference on.
            val_inference_batches_limit = config.get('val_inference_batches')
            if val_inference_batches_limit is None or i < val_inference_batches_limit:
                # Iterate through each image in the current batch to generate captions
                for j in range(images.size(0)):
                    image_tensor_single = images[j] # Get a single image tensor (C, H, W)
                    generated_caption = model.generate_caption(
                        image_tensor_single, vocabulary, device,
                        beam_size=config.get('val_beam_size', 3), # Use beam search for validation
                        max_length=config.get('max_caption_length', 20)
                    )
                    # Convert reference caption indices back to string for metric calculation
                    reference_caption_str = vocabulary.indices_to_caption(captions[j].cpu().numpy())
                    val_generated_captions.append(generated_caption)
                    val_reference_captions.append(reference_caption_str)

    val_avg_loss = val_running_loss / total_batches
    perplexity = math.exp(val_avg_loss) if val_avg_loss < float('inf') else float('inf')
    logger.info(f"Validation Avg Loss: {val_avg_loss:.4f}, Perplexity: {perplexity:.4f}")

    return val_avg_loss, val_generated_captions, val_reference_captions


def train_model(config):
    """
    Main training function. Orchestrates training and validation epochs.

    Args:
        config (dict): Configuration dictionary containing all training parameters.

    Returns:
        tuple: (Trained model, optimizer, scheduler, vocabulary)
    """
    logger.info("Starting training process...")

    # Set device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data paths from configuration
    data_folder = config['data_folder']
    train_image_folder = config['train_image_folder']
    val_image_folder = config['val_image_folder']
    train_caption_file = config['train_caption_file']
    val_caption_file = config['val_caption_file']

    # Check if caption files exist
    if not os.path.exists(train_caption_file):
        raise FileNotFoundError(f"Training caption file not found: {train_caption_file}")
    if not os.path.exists(val_caption_file):
        raise FileNotFoundError(f"Validation caption file not found: {val_caption_file}")

    # Image transformations for training and validation
    train_transform = get_train_transform()
    val_transform = get_eval_transform() # Use eval transform for validation images

    # ======================== VOCABULARY HANDLING ========================
    # Define paths for loading/saving vocabulary
    # First, try to load from a pre-saved vocabulary file in the output directory
    VOCABULARY_FILE_PATH = os.path.join(config['output_dir'], 'vocabulary.pkl')

    vocabulary = None # Initialize vocabulary to None

    # Try to LOAD vocabulary
    if os.path.exists(VOCABULARY_FILE_PATH):
        try:
            with open(VOCABULARY_FILE_PATH, 'rb') as f:
                vocabulary = pickle.load(f)
            logger.info(f"Loaded vocabulary from {VOCABULARY_FILE_PATH}")
        except Exception as e:
            logger.warning(f"Could not load vocabulary from {VOCABULARY_FILE_PATH}: {e}. Will attempt to build new vocabulary.")
            vocabulary = None # Ensure it's None if loading fails
    else:
        logger.info(f"Vocabulary file not found at {VOCABULARY_FILE_PATH}. Will build new vocabulary.")

    # If vocabulary is still None (meaning it couldn't be loaded), then BUILD a new one
    if vocabulary is None:
        logger.info("Building new vocabulary from training dataset...")
        # Create a temporary dataset to build the vocabulary.
        # No image transforms are needed for vocabulary building.
        temp_train_dataset_for_vocab = COCODataset(
            image_dir=os.path.join(data_folder, train_image_folder), # Image dir is still needed for dataset init
            caption_file=train_caption_file,
            subset_size=config.get('vocab_subset_size'), # Use subset if specified for vocab building
            transform=None,
            vocabulary=None # Explicitly tell it to build a new vocabulary
        )
        vocabulary = temp_train_dataset_for_vocab.vocabulary
        del temp_train_dataset_for_vocab # Free up memory
        gc.collect() # Force garbage collection
        logger.info("New vocabulary built.")

        # Save the newly built vocabulary
        try:
            os.makedirs(os.path.dirname(VOCABULARY_FILE_PATH), exist_ok=True)
            with open(VOCABULARY_FILE_PATH, 'wb') as f:
                pickle.dump(vocabulary, f)
            logger.info(f"Saved newly built vocabulary to {VOCABULARY_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error saving newly built vocabulary to {VOCABULARY_FILE_PATH}: {e}")
    # ===========================================================================


    # Create datasets for training and validation using the determined vocabulary
    train_dataset = COCODataset(
        image_dir=os.path.join(data_folder, train_image_folder),
        caption_file=train_caption_file,
        vocabulary=vocabulary, # Pass the vocabulary
        max_caption_length=config.get('max_caption_length', 20),
        subset_size=config.get('train_subset_size'),
        transform=train_transform
    )

    val_dataset = COCODataset(
        image_dir=os.path.join(data_folder, val_image_folder),
        caption_file=val_caption_file,
        vocabulary=vocabulary, # Pass the same vocabulary
        max_caption_length=config.get('max_caption_length', 20),
        subset_size=config.get('val_subset_size'),
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True, # Shuffle training data
        num_workers=config.get('num_workers', 2),
        pin_memory=True # Pin memory for faster data transfer to GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False, # Do not shuffle validation data
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model
    model = ImageCaptioningModel(
        vocab_size=vocabulary.vocab_size,
        embed_dim=config.get('embed_dim', 256),
        attention_dim=config.get('attention_dim', 256),
        decoder_dim=config.get('decoder_dim', 256),
        dropout=config.get('dropout', 0.5),
        fine_tune_encoder=config.get('fine_tune_encoder', True),
        max_caption_length=config.get('max_caption_length', 20) # Pass for model's generate_caption
    ).to(device) # Move model to specified device

    # Loss function and optimizer
    # CrossEntropyLoss ignores the <PAD> token in target labels
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.word2idx['<PAD>']).to(device)

    # Separate optimizer for encoder and decoder if fine_tune_encoder is True.
    # This allows setting different learning rates.
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': config.get('encoder_learning_rate', 1e-5) if config.get('fine_tune_encoder', True) else 0.0},
        {'params': decoder_params, 'lr': config.get('learning_rate', 4e-4)}
    ])

    # Learning rate scheduler: Reduces learning rate when a metric (BLEU-4) stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max', # Monitor validation metric (e.g., BLEU-4, which we want to maximize)
        factor=config.get('lr_reduce_factor', 0.5), # Factor by which the learning rate will be reduced
        patience=config.get('lr_patience', 5), # Number of epochs with no improvement after which learning rate will be reduced
        verbose=True, # Print messages when LR is updated
        min_lr=1e-7 # Minimum learning rate
    )

    # ======================== RESUMPTION LOGIC ========================
    start_epoch = 0
    # Initialize best_val_score to a very low value for 'max' mode, so any improvement is noted
    best_val_score = 0.0
    output_dir = config['output_dir']
    models_dir = config['models_dir']

    # Try to find and load the latest checkpoint to resume training
    latest_checkpoint_path = None
    # Look for best_model_bleu*.pth first, then model_epoch_*.pth
    saved_models = [f for f in os.listdir(models_dir) if f.startswith('best_model_bleu') and f.endswith('.pth')]
    if not saved_models:
        saved_models = [f for f in os.listdir(output_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]

    if saved_models:
        if 'best_model_bleu' in saved_models[0]:
            # Sort by BLEU score extracted from filename for best_model_bleu naming
            latest_checkpoint_name = max(saved_models, key=lambda f: float(f.split('bleu')[1].replace('.pth', '')))
        else: # For 'model_epoch_X.pth' or similar, sort by epoch number
            latest_checkpoint_name = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

        # Determine the full path of the latest checkpoint
        if latest_checkpoint_name.startswith('best_model_bleu'):
            latest_checkpoint_path = os.path.join(models_dir, latest_checkpoint_name)
        else:
            latest_checkpoint_path = os.path.join(output_dir, latest_checkpoint_name)

        logger.info(f"Attempting to resume training from checkpoint: {latest_checkpoint_path}")
        try:
            # Load checkpoint without strict=False unless there are known key mismatches
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state if it exists in the checkpoint (important for correct LR adjustment)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                logger.warning("Scheduler state not found in checkpoint. Scheduler will restart its state.")

            start_epoch = checkpoint['epoch']
            # Safely get best_val_score, default to 0.0 if not found
            best_val_score = checkpoint.get('best_val_score', 0.0)
            logger.info(f"Resumed training from epoch {start_epoch}. Best validation score so far: {best_val_score:.4f}")
        except Exception as e:
            logger.error(f"Could not load checkpoint from {latest_checkpoint_path}: {e}. Starting training from scratch.")
            # Reset start_epoch and best_val_score if loading fails
            start_epoch = 0
            best_val_score = 0.0
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
    # ===========================================================================


    # Training loop
    num_epochs = config.get('num_epochs', 10)

    for epoch in range(start_epoch, num_epochs): # Start from 'start_epoch' for resuming
        # Train for one epoch
        epoch_train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)

        # Validate after each training epoch
        val_avg_loss, val_generated_captions, val_reference_captions = validate_epoch(
            model, val_loader, criterion, vocabulary, device, config
        )

        # Calculate BLEU scores on validation set for tracking and scheduler stepping
        if val_generated_captions and val_reference_captions:
            val_bleu_scores = calculate_bleu_scores_detailed(val_reference_captions, val_generated_captions)
            current_val_score_for_scheduler = val_bleu_scores['BLEU-4'] # Use BLEU-4 for scheduler
            logger.info(f"Epoch {epoch+1} Validation BLEU-4: {current_val_score_for_scheduler:.4f}")

            # Step the scheduler based on validation BLEU-4.
            # This will reduce the learning rate if BLEU-4 does not improve for 'patience' epochs.
            scheduler.step(current_val_score_for_scheduler)

            # Save the best model based on BLEU-4 score on the validation set
            if current_val_score_for_scheduler > best_val_score:
                best_val_score = current_val_score_for_scheduler
                # Save best model to the 'models' directory
                model_path = os.path.join(models_dir, f"best_model_bleu{best_val_score:.4f}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), # IMPORTANT: Save scheduler state!
                    'loss': epoch_train_loss,
                    'vocabulary': vocabulary,
                    'config': config, # Save config for easy loading later
                    'best_val_score': best_val_score # Save the best score achieved
                }, model_path)
                logger.info(f"Saved best model checkpoint to {model_path}")
        else:
            logger.warning("No captions generated during validation for metric calculation. Scheduler stepped with 0.0.")
            scheduler.step(0.0) # Step with a low value if no metrics

        # Save checkpoint periodically (optional)
        # This is good practice for resuming training even if it's not the "best" model yet.
        if (epoch + 1) % config.get('save_interval', 5) == 0:
            model_path_periodic = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # IMPORTANT: Save scheduler state!
                'loss': epoch_train_loss,
                'vocabulary': vocabulary,
                'config': config,
                'best_val_score': best_val_score # Also save current best score here
            }, model_path_periodic)
            logger.info(f"Saved periodic model checkpoint to {model_path_periodic}")


        # ======================== MEMORY OPTIMIZATION AFTER EACH EPOCH ========================
        logger.info("Performing memory optimization after epoch...")
        # Clear PyTorch's CUDA cache (if using GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache emptied.")

        # Force Python's garbage collector to run to free up unreferenced objects
        gc.collect()
        logger.info("Python garbage collector run.")
        # ======================================================================================

    logger.info("Training complete.")
    return model, optimizer, scheduler, vocabulary # Return trained components for potential further use


if __name__ == '__main__':
    # When `train.py` is run directly, it will initiate the training process.
    from config import TRAINING_CONFIG, update_config_with_latest_model, _MODELS_DIR, _OUTPUT_DIR

    # Update config to ensure it looks for latest model in 'models' dir
    # This specifically helps if you copy pre-trained models into 'models' folder for initial load.
    # If starting from scratch, it will still default to 0.0000.
    update_config_with_latest_model(TRAINING_CONFIG)

    logger.info("Starting model training process...")
    try:
        trained_model, optimizer, scheduler, vocabulary = train_model(TRAINING_CONFIG)
        logger.info("Model Training Complete!")

        # Optional: You might want to save the final model explicitly if it's not the best one.
        # This ensures you have the model from the last epoch.
        final_model_path = os.path.join(_MODELS_DIR, f"final_model_epoch_{TRAINING_CONFIG['num_epochs']}.pth")
        torch.save({
            'epoch': TRAINING_CONFIG['num_epochs'],
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocabulary': vocabulary,
            'config': TRAINING_CONFIG,
            'best_val_score': 0 # Placeholder, retrieve from scheduler if needed
        }, final_model_path)
        logger.info(f"Saved final model checkpoint to {final_model_path}")


    except FileNotFoundError as e:
        logger.error(f"Critical data file missing: {e}")
        logger.error("Please ensure the COCO dataset and annotation files are correctly placed as described in README.md.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during training: {e}", exc_info=True)
        # exc_info=True prints the full traceback
