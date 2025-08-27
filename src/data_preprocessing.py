import os
import json
import pickle
import random
from collections import Counter
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import get_logger, get_eval_transform # Import logger and transforms from utils

logger = get_logger(__name__)


class COCOVocabulary:
    """
    Vocabulary builder for COCO captions.
    Handles tokenization, building word-to-index and index-to-word mappings,
    and converting captions to numerical indices.
    """
    def __init__(self, min_word_freq=5):
        """
        Initializes the COCOVocabulary.
        Args:
            min_word_freq (int): Minimum frequency for a word to be included in the vocabulary.
                                 Words less frequent than this will be replaced by <UNK>.
        """
        self.min_word_freq = min_word_freq
        self.word2idx = {} # Maps words to their numerical indices
        self.idx2word = {} # Maps numerical indices back to words
        self.word_freq = Counter() # Counts frequency of each word
        self.vocab_size = 0 # Total number of unique words in the vocabulary

    def build_vocabulary(self, captions):
        """
        Builds the vocabulary from a list of captions.
        Args:
            captions (list): A list of strings, where each string is a caption.
        """
        logger.info("Building vocabulary...")

        # 1. Count word frequencies
        for caption in tqdm(captions, desc="Counting word frequencies"):
            tokens = self.tokenize(caption)
            self.word_freq.update(tokens)

        # 2. Add special tokens
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        for token in special_tokens:
            if token not in self.word2idx: # Avoid re-adding if already present
                self.word2idx[token] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = token

        # 3. Add words that meet the minimum frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= self.min_word_freq:
                if word not in self.word2idx: # Avoid re-adding words if they are special tokens
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.idx2word)] = word

        self.vocab_size = len(self.word2idx)
        logger.info(f"Vocabulary built successfully. Size: {self.vocab_size}")

    def tokenize(self, caption):
        """
        Simple tokenization: convert to lowercase, strip leading/trailing spaces,
        and split by space. Normalizes multiple spaces.
        Args:
            caption (str): The input caption string.
        Returns:
            list: A list of tokenized words.
        """
        caption = caption.lower().strip()
        # Normalize multiple spaces into a single space
        caption = ' '.join(caption.split())
        tokens = caption.split()
        return tokens

    def caption_to_indices(self, caption, max_length=20):
        """
        Converts a caption string into a list of numerical indices.
        Adds <START> and <END> tokens and pads with <PAD> up to max_length.
        Args:
            caption (str): The input caption string.
            max_length (int): The maximum desired length for the indexed caption.
        Returns:
            list: A list of integer indices representing the caption.
        """
        tokens = self.tokenize(caption)
        indices = [self.word2idx['<START>']] # Start with the <START> token

        for token in tokens:
            if len(indices) >= max_length - 1: # Reserve space for <END>
                break
            idx = self.word2idx.get(token, self.word2idx['<UNK>']) # Use <UNK> for unknown words
            indices.append(idx)

        indices.append(self.word2idx['<END>']) # End with the <END> token

        # Pad with <PAD> tokens if the caption is shorter than max_length
        while len(indices) < max_length:
            indices.append(self.word2idx['<PAD>'])

        return indices[:max_length] # Ensure the caption does not exceed max_length

    def indices_to_caption(self, indices):
        """
        Converts a list of numerical indices back into a human-readable caption string.
        Stops at <END> token and ignores <PAD> and <START> tokens.
        Args:
            indices (list or numpy.ndarray): A list or array of integer indices.
        Returns:
            str: The reconstructed caption string.
        """
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>') # Get word, default to <UNK>
            if word == '<END>':
                break # Stop decoding when <END> token is encountered
            if word not in ['<PAD>', '<START>']: # Ignore special tokens
                words.append(word)
        return ' '.join(words)


class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO Image Captioning.
    Loads image paths and their corresponding captions,
    and returns preprocessed image tensors and indexed caption tensors.
    """
    def __init__(self, image_dir, caption_file, vocabulary=None,
                 max_caption_length=20, subset_size=None, transform=None):
        """
        Initializes the COCODataset.
        Args:
            image_dir (str): Path to the directory containing COCO images (e.g., 'train2017', 'val2017').
            caption_file (str): Path to the COCO captions JSON file (e.g., 'captions_train2017.json').
            vocabulary (COCOVocabulary, optional): A pre-built COCOVocabulary object. If None,
                                                   a new vocabulary will be built from the captions.
            max_caption_length (int): Maximum length for indexed captions.
            subset_size (int, optional): If specified, uses a random subset of this size from the dataset.
            transform (torchvision.transforms.Compose, optional): Image transformations to apply.
        """
        self.image_dir = image_dir
        self.max_caption_length = max_caption_length
        self.transform = transform if transform is not None else get_eval_transform() # Default transform

        try:
            with open(caption_file, 'r') as f:
                self.coco_data = json.load(f)
            logger.info(f"Successfully loaded captions from {caption_file}")
        except FileNotFoundError:
            logger.error(f"Caption file not found at {caption_file}. Please check the path.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {caption_file}. Ensure it's a valid JSON file.")
            raise

        # Create a mapping from image ID to its filename for quick lookup
        self.id_to_filename = {img_info['id']: img_info['file_name'] for img_info in self.coco_data['images']}

        self.data = [] # Stores (image_path, caption, image_id) tuples
        missing_image_files = 0

        # Process annotations to pair image paths with captions
        for ann in tqdm(self.coco_data['annotations'], desc="Processing annotations"):
            image_id = ann['image_id']
            if image_id in self.id_to_filename:
                caption = ann['caption']
                filename = self.id_to_filename[image_id]
                image_full_path = os.path.join(image_dir, filename)

                if os.path.exists(image_full_path):
                    self.data.append({
                        'image_path': image_full_path,
                        'caption': caption,
                        'image_id': image_id # Store original image_id for evaluation
                    })
                else:
                    missing_image_files += 1
                    # logger.warning(f"Image file not found: {image_full_path}. Skipping this annotation.")
            else:
                logger.warning(f"Image ID {image_id} not found in images list. Skipping annotation.")

        if missing_image_files > 0:
            logger.warning(f"Skipped {missing_image_files} annotations due to missing image files. "
                           "Please ensure all images are in the specified directory.")

        # If subset_size is specified, take a random sample
        if subset_size and subset_size < len(self.data):
            self.data = random.sample(self.data, subset_size)
            logger.info(f"Using subset of {subset_size} samples for the dataset.")

        logger.info(f"Dataset size after filtering: {len(self.data)} samples.")

        # Build vocabulary if not provided
        if vocabulary is None:
            self.vocabulary = COCOVocabulary()
            captions_for_vocab = [item['caption'] for item in self.data]
            self.vocabulary.build_vocabulary(captions_for_vocab)
        else:
            self.vocabulary = vocabulary

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the given index.
        Returns:
            tuple: (image_tensor, caption_tensor, caption_length, image_id)
        """
        item = self.data[idx]

        # Load and transform image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {item['image_path']}: {e}. Returning a black image as fallback.")
            # Return a black image tensor of expected size (3, 224, 224) if image loading fails
            image = torch.zeros(3, 224, 224)

        # Convert caption to indices
        caption_indices = self.vocabulary.caption_to_indices(
            item['caption'], self.max_caption_length
        )
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        # Calculate actual length of the caption (excluding padding, including START/END)
        try:
            # Find the index of <END> token, length is (index + 1)
            end_idx = caption_indices.index(self.vocabulary.word2idx['<END>'])
            caption_length = end_idx + 1
        except ValueError:
            # If <END> not found (shouldn't happen with proper max_caption_length),
            # count non-PAD tokens.
            caption_length = len([idx for idx in caption_indices if idx != self.vocabulary.word2idx['<PAD>']])

        caption_length = torch.tensor(caption_length, dtype=torch.long)

        # Return image tensor, caption tensor, actual caption length, and original image ID
        return image, caption_tensor, caption_length, item['image_id']
