# import logging
# import sys
# import os
# import shutil
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import torch
# import torchvision.transforms as transforms

# # --- Logging Configuration ---
# # Configure logging to output to console and a file.
# # This logger will be used across all modules.
# logger = logging.getLogger(__name__) # Get a logger specific to this module
# logger.setLevel(logging.INFO) # Set the minimum level of messages to be logged

# # Ensure handlers are not duplicated if script is run multiple times in same session
# if not logger.handlers:
#     # Console handler
#     c_handler = logging.StreamHandler(sys.stdout)
#     c_handler.setLevel(logging.INFO)

#     # File handler - logs to 'training.log' in the 'output' directory
#     # Ensure 'output' directory exists before creating the log file
#     log_dir = 'output'
#     os.makedirs(log_dir, exist_ok=True)
#     f_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
#     f_handler.setLevel(logging.INFO)

#     # Formatters
#     c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     c_handler.setFormatter(c_format)
#     f_handler.setFormatter(f_format)

#     # Add handlers to the logger
#     logger.addHandler(c_handler)
#     logger.addHandler(f_handler)

# # Get a top-level logger to ensure all modules use the same setup
# def get_logger(name=__name__):
#     """Returns a logger instance with predefined settings."""
#     return logging.getLogger(name)


# # --- Image Transformation Utilities ---
# def get_train_transform():
#     """Get image transform for training (resize, horizontal flip, normalize)"""
#     return transforms.Compose([
#         transforms.Resize((224, 224)), # Resize images to 224x224 pixels
#         transforms.RandomHorizontalFlip(), # Randomly flip images horizontally for data augmentation
#         transforms.ToTensor(), # Convert PIL Image or numpy.ndarray to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize pixel values
#                              std=[0.229, 0.224, 0.225])
#     ])

# def get_eval_transform():
#     """Get image transform for evaluation/inference (resize, normalize)"""
#     return transforms.Compose([
#         transforms.Resize((224, 224)), # Resize images to 224x224 pixels
#         transforms.ToTensor(), # Convert PIL Image or numpy.ndarray to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize pixel values
#                              std=[0.229, 0.224, 0.225])
#     ])


# # --- Dataset Analysis Utility (moved from original script for modularity) ---
# class DatasetAnalyzer:
#     """Utility class to analyze COCO dataset statistics."""

#     @staticmethod
#     def analyze_captions(caption_file, max_samples=None):
#         """
#         Analyzes caption statistics from a COCO-format JSON file.
#         Args:
#             caption_file (str): Path to the COCO captions JSON file.
#             max_samples (int, optional): Maximum number of captions to analyze.
#                                          Useful for large datasets. Defaults to None (all).
#         Returns:
#             dict: A dictionary containing various caption statistics.
#         """
#         try:
#             with open(caption_file, 'r') as f:
#                 data = json.load(f)
#         except FileNotFoundError:
#             logger.error(f"Caption file not found for analysis: {caption_file}")
#             return {}
#         except json.JSONDecodeError:
#             logger.error(f"Error decoding JSON from {caption_file}. Ensure it's valid.")
#             return {}

#         captions = [ann['caption'] for ann in data['annotations']]
#         if max_samples:
#             captions = captions[:max_samples]

#         # Basic statistics
#         lengths = [len(caption.split()) for caption in captions]

#         stats = {
#             'total_captions': len(captions),
#             'avg_length': np.mean(lengths) if lengths else 0,
#             'std_length': np.std(lengths) if lengths else 0,
#             'min_length': min(lengths) if lengths else 0,
#             'max_length': max(lengths) if lengths else 0,
#             'median_length': np.median(lengths) if lengths else 0
#         }

#         # Word frequency
#         all_words = []
#         from collections import Counter # Import here to avoid circular dependency issues if Counter is used elsewhere
#         for caption in captions:
#             words = caption.lower().split()
#             all_words.extend(words)

#         word_freq = Counter(all_words)
#         stats['unique_words'] = len(word_freq)
#         stats['most_common_words'] = word_freq.most_common(20)

#         return stats

#     @staticmethod
#     def plot_length_distribution(caption_file, max_samples=None, save_path=None):
#         """
#         Plots the distribution of caption lengths.
#         Args:
#             caption_file (str): Path to the COCO captions JSON file.
#             max_samples (int, optional): Maximum number of captions to plot. Defaults to None (all).
#             save_path (str, optional): Path to save the plot. If None, displays the plot.
#         """
#         try:
#             with open(caption_file, 'r') as f:
#                 data = json.load(f)
#         except FileNotFoundError:
#             logger.error(f"Caption file not found for plotting: {caption_file}")
#             return
#         except json.JSONDecodeError:
#             logger.error(f"Error decoding JSON from {caption_file}. Ensure it's valid.")
#             return

#         captions = [ann['caption'] for ann in data['annotations']]
#         if max_samples:
#             captions = captions[:max_samples]

#         lengths = [len(caption.split()) for caption in captions]

#         plt.figure(figsize=(10, 6))
#         plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
#         plt.xlabel('Caption Length (words)')
#         plt.ylabel('Frequency')
#         plt.title('Distribution of Caption Lengths')
#         plt.grid(True, alpha=0.3)

#         if save_path:
#             plt.savefig(save_path, bbox_inches='tight', dpi=150)
#             logger.info(f"Caption length distribution plot saved to {save_path}")
#         else:
#             plt.show()

# # Import json here as it's used by DatasetAnalyzer
# import json

# # --- Attention Visualization Utility ---
# def visualize_attention(model, image_path, vocabulary, device, save_path=None, max_words_to_show=10):
#     """
#     Visualizes attention weights on an image for a generated caption.
#     This function requires the model to have a `generate_caption` method
#     and access to the encoder and decoder components to extract attention.

#     Args:
#         model (ImageCaptioningModel): The trained image captioning model.
#         image_path (str): Path to the image file for visualization.
#         vocabulary (COCOVocabulary): The vocabulary object.
#         device (torch.device): Device to run the model on (cpu/cuda).
#         save_path (str, optional): Path to save the visualization plot. If None, displays the plot.
#         max_words_to_show (int): Maximum number of words to visualize attention for.
#     """
#     logger = get_logger(__name__) # Get logger for this function

#     model.eval() # Set model to evaluation mode

#     # Load and preprocess image
#     transform = get_eval_transform()

#     try:
#         image = Image.open(image_path).convert('RGB')
#     except FileNotFoundError:
#         logger.error(f"Image not found at {image_path} for attention visualization.")
#         return
#     except Exception as e:
#         logger.error(f"Error loading image {image_path} for attention visualization: {e}")
#         return

#     image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

#     with torch.no_grad():
#         # Get encoder output
#         # (1, encoder_dim, encoded_image_size, encoded_image_size)
#         encoder_out = model.encoder(image_tensor)

#         # Reshape for attention: (1, num_pixels, encoder_dim)
#         encoder_out_reshaped = encoder_out.permute(0, 2, 3, 1).contiguous()
#         encoder_out_reshaped = encoder_out_reshaped.view(1, -1, model.encoder_dim)

#         # Initialize decoder states
#         h, c = model.decoder.init_hidden_state(encoder_out_reshaped)

#         # Start of sentence token
#         word_idx = vocabulary.word2idx['<START>']
#         caption_words = []
#         attention_weights = []

#         # Generate caption word by word and collect attention weights
#         # Iterate up to max_words_to_show or until <END> token is generated
#         for _ in range(model.decoder.max_caption_length_for_inference): # Use model's max_length
#             if word_idx == vocabulary.word2idx['<END>'] or len(caption_words) >= max_words_to_show:
#                 break

#             # Get embeddings for current word
#             # (1, embed_dim)
#             embeddings = model.decoder.embedding(torch.LongTensor([word_idx]).to(device))

#             # Get attention-weighted encoding and alpha
#             # alpha: (1, num_pixels)
#             awe, alpha = model.decoder.attention(encoder_out_reshaped, h)
#             attention_weights.append(alpha.cpu().numpy())

#             # Apply gate to attention-weighted encoding
#             gate = model.decoder.sigmoid(model.decoder.f_beta(h))
#             awe = gate * awe

#             # Perform one step of LSTM decoding
#             h, c = model.decoder.decode_step(
#                 torch.cat([embeddings, awe], dim=1),
#                 (h, c)
#             )

#             # Predict next word
#             scores = model.decoder.fc(h) # (1, vocab_size)
#             word_idx = scores.argmax(dim=1).item() # Get the index of the predicted word

#             word = vocabulary.idx2word[word_idx]
#             caption_words.append(word)

#     # Visualize the attention maps
#     num_plots = len(caption_words)
#     if num_plots == 0:
#         logger.warning("No words generated for attention visualization. Cannot create plot.")
#         return

#     # Adjust figure size dynamically based on number of plots
#     fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))
#     if num_plots == 1: # Ensure axes is iterable even for single plot
#         axes = [axes]

#     for i, (word, alpha) in enumerate(zip(caption_words, attention_weights)):
#         # Reshape attention to encoder's spatial size (e.g., 14x14 for ResNet50)
#         # Assuming encoded_image_size is available in model.encoder
#         enc_img_size = model.encoder.encoded_image_size
#         alpha_img = alpha.reshape(enc_img_size, enc_img_size)

#         # Resize attention map to original image size for overlay
#         alpha_img_resized = Image.fromarray(alpha_img * 255).resize(image.size, Image.LANCZOS)
#         alpha_img_np = np.array(alpha_img_resized) / 255.0 # Normalize back to 0-1

#         axes[i].imshow(image)
#         axes[i].imshow(alpha_img_np, alpha=0.6, cmap='jet') # Overlay attention map
#         axes[i].set_title(f'Word: {word}')
#         axes[i].axis('off')

#     plt.suptitle(f"Generated Caption (Attention Visualization): {' '.join(caption_words)}")
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, bbox_inches='tight', dpi=150)
#         logger.info(f"Attention visualization saved to {save_path}")
#     else:
#         plt.show()

#     return ' '.join(caption_words)





import logging
import sys
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# --- Logging Configuration ---
# Get the root logger
root_logger = logging.getLogger()
# Set the minimum level for the root logger. This ensures all messages at or above
# this level from any logger (including child loggers like those in app.py) are processed.
root_logger.setLevel(logging.INFO)

# Ensure handlers are not duplicated if script is run multiple times in same session
# This check is crucial and applies to the root logger's handlers.
if not root_logger.handlers:
    # Console handler: directs log messages to standard output (console)
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO) # Set level for console output

    # File handler: directs log messages to a file
    # Ensure the 'output' directory exists before creating the log file
    log_dir = 'output'
    os.makedirs(log_dir, exist_ok=True)
    f_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    f_handler.setLevel(logging.INFO) # Set level for file output

    # Formatters define the layout of log records
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the ROOT logger. This is the critical change.
    # Any logger instance obtained later will inherit these handlers by default.
    root_logger.addHandler(c_handler)
    root_logger.addHandler(f_handler)

def get_logger(name=__name__):
    """
    Returns a logger instance with predefined settings.
    When called with a specific name (e.g., __name__), it retrieves
    a child logger that inherits settings (like handlers) from the root logger.
    """
    return logging.getLogger(name)


# --- Image Transformation Utilities ---
def get_train_transform():
    """Get image transform for training (resize, horizontal flip, normalize)"""
    return transforms.Compose([
        transforms.Resize((224, 224)), # Resize images to 224x224 pixels
        transforms.RandomHorizontalFlip(), # Randomly flip images horizontally for data augmentation
        transforms.ToTensor(), # Convert PIL Image or numpy.ndarray to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize pixel values
                             std=[0.229, 0.224, 0.225])
    ])

def get_eval_transform():
    """Get image transform for evaluation/inference (resize, normalize)"""
    return transforms.Compose([
        transforms.Resize((224, 224)), # Resize images to 224x224 pixels
        transforms.ToTensor(), # Convert PIL Image or numpy.ndarray to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize pixel values
                             std=[0.229, 0.224, 0.225])
    ])


# --- Dataset Analysis Utility (moved from original script for modularity) ---
class DatasetAnalyzer:
    """Utility class to analyze COCO dataset statistics."""

    @staticmethod
    def analyze_captions(caption_file, max_samples=None):
        """
        Analyzes caption statistics from a COCO-format JSON file.
        Args:
            caption_file (str): Path to the COCO captions JSON file.
            max_samples (int, optional): Maximum number of captions to analyze.
                                         Useful for large datasets. Defaults to None (all).
        Returns:
            dict: A dictionary containing various caption statistics.
        """
        try:
            with open(caption_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            root_logger.error(f"Caption file not found for analysis: {caption_file}")
            return {}
        except json.JSONDecodeError:
            root_logger.error(f"Error decoding JSON from {caption_file}. Ensure it's valid.")
            return {}

        captions = [ann['caption'] for ann in data['annotations']]
        if max_samples:
            captions = captions[:max_samples]

        # Basic statistics
        lengths = [len(caption.split()) for caption in captions]

        stats = {
            'total_captions': len(captions),
            'avg_length': np.mean(lengths) if lengths else 0,
            'std_length': np.std(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'median_length': np.median(lengths) if lengths else 0
        }

        # Word frequency
        all_words = []
        from collections import Counter # Import here to avoid circular dependency issues if Counter is used elsewhere
        for caption in captions:
            words = caption.lower().split()
            all_words.extend(words)

        word_freq = Counter(all_words)
        stats['unique_words'] = len(word_freq)
        stats['most_common_words'] = word_freq.most_common(20)

        return stats

    @staticmethod
    def plot_length_distribution(caption_file, max_samples=None, save_path=None):
        """
        Plots the distribution of caption lengths.
        Args:
            caption_file (str): Path to the COCO captions JSON file.
            max_samples (int, optional): Maximum number of captions to plot. Defaults to None (all).
            save_path (str, optional): Path to save the plot. If None, displays the plot.
        """
        try:
            with open(caption_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            root_logger.error(f"Caption file not found for plotting: {caption_file}")
            return
        except json.JSONDecodeError:
            root_logger.error(f"Error decoding JSON from {caption_file}. Ensure it's valid.")
            return

        captions = [ann['caption'] for ann in data['annotations']]
        if max_samples:
            captions = captions[:max_samples]

        lengths = [len(caption.split()) for caption in captions]

        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Caption Length (words)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Caption Lengths')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            root_logger.info(f"Caption length distribution plot saved to {save_path}")
        else:
            plt.show()

# Import json here as it's used by DatasetAnalyzer
import json

# --- Attention Visualization Utility ---
def visualize_attention(model, image_path, vocabulary, device, save_path=None, max_words_to_show=10):
    """
    Visualizes attention weights on an image for a generated caption.
    This function requires the model to have a `generate_caption` method
    and access to the encoder and decoder components to extract attention.

    Args:
        model (ImageCaptioningModel): The trained image captioning model.
        image_path (str): Path to the image file for visualization.
        vocabulary (COCOVocabulary): The vocabulary object.
        device (torch.device): Device to run the model on (cpu/cuda).
        save_path (str, optional): Path to save the visualization plot. If None, displays the plot.
        max_words_to_show (int): Maximum number of words to visualize attention for.
    """
    logger = get_logger(__name__) # Get logger for this function

    model.eval() # Set model to evaluation mode

    # Load and preprocess image
    transform = get_eval_transform()

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        logger.error(f"Image not found at {image_path} for attention visualization.")
        return
    except Exception as e:
        logger.error(f"Error loading image {image_path} for attention visualization: {e}")
        return

    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    with torch.no_grad():
        # Get encoder output
        # (1, encoder_dim, encoded_image_size, encoded_image_size)
        encoder_out = model.encoder(image_tensor)

        # Reshape for attention: (1, num_pixels, encoder_dim)
        encoder_out_reshaped = encoder_out.permute(0, 2, 3, 1).contiguous()
        encoder_out_reshaped = encoder_out_reshaped.view(1, -1, model.encoder_dim)

        # Initialize decoder states
        h, c = model.decoder.init_hidden_state(encoder_out_reshaped)

        # Start of sentence token
        word_idx = vocabulary.word2idx['<START>']
        caption_words = []
        attention_weights = []

        # Generate caption word by word and collect attention weights
        # Iterate up to max_words_to_show or until <END> token is generated
        for _ in range(model.decoder.max_caption_length_for_inference): # Use model's max_length
            if word_idx == vocabulary.word2idx['<END>'] or len(caption_words) >= max_words_to_show:
                break

            # Get embeddings for current word
            # (1, embed_dim)
            embeddings = model.decoder.embedding(torch.LongTensor([word_idx]).to(device))

            # Get attention-weighted encoding and alpha
            # alpha: (1, num_pixels)
            awe, alpha = model.decoder.attention(encoder_out_reshaped, h)
            attention_weights.append(alpha.cpu().numpy())

            # Apply gate to attention-weighted encoding
            gate = model.decoder.sigmoid(model.decoder.f_beta(h))
            awe = gate * awe

            # Perform one step of LSTM decoding
            h, c = model.decoder.decode_step(
                torch.cat([embeddings, awe], dim=1),
                (h, c)
            )

            # Predict next word
            scores = model.decoder.fc(h) # (1, vocab_size)
            word_idx = scores.argmax(dim=1).item() # Get the index of the predicted word

            word = vocabulary.idx2word[word_idx]
            caption_words.append(word)

    # Visualize the attention maps
    num_plots = len(caption_words)
    if num_plots == 0:
        logger.warning("No words generated for attention visualization. Cannot create plot.")
        return

    # Adjust figure size dynamically based on number of plots
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))
    if num_plots == 1: # Ensure axes is iterable even for single plot
        axes = [axes]

    for i, (word, alpha) in enumerate(zip(caption_words, attention_weights)):
        # Reshape attention to encoder's spatial size (e.g., 14x14 for ResNet50)
        # Assuming encoded_image_size is available in model.encoder
        enc_img_size = model.encoder.encoded_image_size
        alpha_img = alpha.reshape(enc_img_size, enc_img_size)

        # Resize attention map to original image size for overlay
        alpha_img_resized = Image.fromarray(alpha_img * 255).resize(image.size, Image.LANCZOS)
        alpha_img_np = np.array(alpha_img_resized) / 255.0 # Normalize back to 0-1

        axes[i].imshow(image)
        axes[i].imshow(alpha_img_np, alpha=0.6, cmap='jet') # Overlay attention map
        axes[i].set_title(f'Word: {word}')
        axes[i].axis('off')

    plt.suptitle(f"Generated Caption (Attention Visualization): {' '.join(caption_words)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Attention visualization saved to {save_path}")
    else:
        plt.show()

    return ' '.join(caption_words)
