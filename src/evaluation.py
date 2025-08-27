import os
import time
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import math # For perplexity
import random
from .config import EVALUATION_CONFIG, update_config_with_latest_model
from .data_preprocessing import COCOVocabulary

# Import necessary NLTK components for BLEU, METEOR
try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    # Suppress NLTK download messages if already downloaded
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("NLTK not installed or data not downloaded. BLEU/METEOR scores will be skipped.")
    print("Please install NLTK (`pip install nltk`) and download data (`python -c \"import nltk; nltk.download('punkt'); nltk.download('wordnet')\"`)")
    corpus_bleu = None
    meteor_score = None
    word_tokenize = None
    SmoothingFunction = None

# Import ROUGE
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("rouge-score not installed. ROUGE-L score will be skipped.")
    print("Please install it: `pip install rouge-score`")
    rouge_scorer = None

# Import pycocotools and pycocoevalcap for CIDEr
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    import tempfile
except ImportError:
    print("pycocotools or pycocoevalcap not installed. CIDEr score will be skipped.")
    print("Please install: `pip install pycocotools` and `pip install git+https://github.com/salaniz/pycocoevalcap.git`")
    COCO = None
    COCOEvalCap = None
    tempfile = None


from .model import ImageCaptioningModel # Import the model
from .data_preprocessing import COCODataset # Import dataset
from .utils import get_logger, get_eval_transform # Import utilities

logger = get_logger(__name__)


def calculate_bleu_scores_detailed(references, hypotheses):
    """
    Calculates detailed BLEU scores (BLEU-1 to BLEU-4) for a corpus.
    Args:
        references (list of str): List of reference captions. Each reference is a single string.
        hypotheses (list of str): List of hypothesis (generated) captions. Each hypothesis is a single string.
    Returns:
        dict: A dictionary containing BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
              Returns zeros if NLTK is not available or an error occurs.
    """
    if corpus_bleu is None or word_tokenize is None or SmoothingFunction is None:
        logger.error("NLTK requirements for BLEU not met. Returning 0 for BLEU scores.")
        return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}

    try:
        # Each reference is a list of ONE tokenized sentence (as `corpus_bleu` expects multiple references per hypothesis)
        # We assume one reference per image for simplicity.
        ref_tokens = [[word_tokenize(ref.lower())] for ref in references]
        hyp_tokens = [word_tokenize(hyp.lower()) for hyp in hypotheses]

        # Use smoothing function for better BLEU calculation, especially for short sentences or small test sets
        smooth = SmoothingFunction().method1

        # Calculate corpus-level BLEU scores for different n-grams
        bleu_1 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu_2 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
        bleu_3 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
        bleu_4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

        return {
            'BLEU-1': bleu_1,
            'BLEU-2': bleu_2,
            'BLEU-3': bleu_3,
            'BLEU-4': bleu_4
        }
    except Exception as e:
        logger.error(f"Error calculating BLEU scores: {e}")
        return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}


def calculate_meteor_score(references, hypotheses):
    """
    Calculates the METEOR score for a corpus.
    Args:
        references (list of str): List of reference captions.
        hypotheses (list of str): List of hypothesis (generated) captions.
    Returns:
        float: Average METEOR score, or None if NLTK/WordNet not available.
    """
    if meteor_score is None or word_tokenize is None:
        logger.error("NLTK requirements for METEOR (wordnet) not met. Returning None for METEOR score.")
        return None

    scores = []
    try:
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = word_tokenize(ref.lower())
            hyp_tokens = word_tokenize(hyp.lower())
            # meteor_score expects a list of reference sentences (even if only one)
            score = meteor_score([ref_tokens], hyp_tokens)
            scores.append(score)

        return np.mean(scores) if scores else 0.0
    except Exception as e:
        logger.error(f"Error calculating METEOR score: {e}")
        return None


def calculate_rouge_l_score(references, hypotheses):
    """
    Calculates the ROUGE-L F-measure score for a corpus.
    Args:
        references (list of str): List of reference captions.
        hypotheses (list of str): List of hypothesis (generated) captions.
    Returns:
        float: Average ROUGE-L score, or None if rouge-score library not available.
    """
    if rouge_scorer is None:
        logger.error("rouge-score library not available. Returning None for ROUGE-L score.")
        return None

    try:
        # Use 'rougeL' for Longest Common Subsequence based ROUGE
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []

        for ref, hyp in zip(references, hypotheses):
            score = scorer.score(ref, hyp)
            scores.append(score['rougeL'].fmeasure) # We are interested in the F-measure

        return np.mean(scores) if scores else 0.0
    except Exception as e:
        logger.error(f"Error calculating ROUGE-L score: {e}")
        return None


def calculate_cider_score(references, hypotheses):
    """
    Calculates the CIDEr score using pycocoevalcap library.
    Requires pycocotools and pycocoevalcap to be installed.
    Args:
        references (list of str): List of reference captions.
        hypotheses (list of str): List of hypothesis (generated) captions.
    Returns:
        float: CIDEr score, or None if pycocotools/pycocoevalcap not available.
    """
    if COCO is None or COCOEvalCap is None or tempfile is None:
        logger.error("pycocotools or pycocoevalcap not available. Returning None for CIDEr score.")
        return None

    try:
        # pycocoevalcap requires data in a specific COCO format
        # Create dummy image IDs for the COCO objects
        dummy_image_ids = list(range(len(references)))

        # Format references for COCO
        refs_coco_format = []
        for i, ref_str in enumerate(references):
            refs_coco_format.append({"image_id": dummy_image_ids[i], "id": i, "caption": ref_str})

        # Format hypotheses for COCO
        hyps_coco_format = []
        for i, hyp_str in enumerate(hypotheses):
            hyps_coco_format.append({"image_id": dummy_image_ids[i], "id": i, "caption": hyp_str})

        # Save to temporary JSON files as required by COCO/COCOEvalCap
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f_ref:
            # Need to create a minimal COCO-like structure for references
            json.dump({"annotations": refs_coco_format, "images": [{"id": i} for i in dummy_image_ids]}, f_ref)
            ref_path = f_ref.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f_hyp:
            json.dump(hyps_coco_format, f_hyp)
            hyp_path = f_hyp.name

        # Initialize COCO and COCOEvalCap objects
        coco = COCO(ref_path)
        cocoRes = coco.loadRes(hyp_path)

        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds() # Specify images to evaluate
        cocoEval.evaluate() # Perform evaluation

        # Clean up temporary files
        os.remove(ref_path)
        os.remove(hyp_path)

        return cocoEval.eval['CIDEr'] # CIDEr score is directly available
    except Exception as e:
        logger.error(f"Error calculating CIDEr score: {e}")
        return None


def calculate_length_statistics(generated_captions, reference_captions):
    """
    Calculates statistics related to caption lengths.
    Args:
        generated_captions (list of str): List of generated captions.
        reference_captions (list of str): List of reference captions.
    Returns:
        dict: Dictionary containing average, std dev, and difference in lengths.
    """
    gen_lengths = [len(cap.split()) for cap in generated_captions]
    ref_lengths = [len(cap.split()) for cap in reference_captions]

    return {
        'avg_generated_length': np.mean(gen_lengths) if gen_lengths else 0,
        'avg_reference_length': np.mean(ref_lengths) if ref_lengths else 0,
        'length_difference': (np.mean(gen_lengths) - np.mean(ref_lengths)) if gen_lengths and ref_lengths else 0,
        'length_std_generated': np.std(gen_lengths) if gen_lengths else 0,
        'length_std_reference': np.std(ref_lengths) if ref_lengths else 0
    }


def calculate_vocabulary_statistics(generated_captions, vocabulary):
    """
    Calculates vocabulary usage statistics for generated captions.
    Args:
        generated_captions (list of str): List of generated captions.
        vocabulary (COCOVocabulary): The vocabulary object used by the model.
    Returns:
        dict: Dictionary with unique word count, vocabulary coverage, etc.
    """
    all_words = []
    from collections import Counter # Import here to avoid circular dependency issues
    for caption in generated_captions:
        words = caption.lower().split()
        all_words.extend(words)

    unique_words = set(all_words)
    word_freq = Counter(all_words)

    return {
        'unique_words_used': len(unique_words),
        'total_vocabulary_size': vocabulary.vocab_size,
        'vocabulary_coverage': len(unique_words) / vocabulary.vocab_size if vocabulary.vocab_size > 0 else 0,
        'avg_word_frequency': np.mean(list(word_freq.values())) if word_freq else 0,
        'most_common_generated_words': word_freq.most_common(10)
    }


def calculate_diversity_metrics(generated_captions):
    """
    Calculates diversity metrics for generated captions, such as Type-Token Ratio (TTR),
    Self-BLEU, and caption uniqueness.
    Args:
        generated_captions (list of str): List of generated captions.
    Returns:
        dict: Dictionary containing diversity metrics.
    """
    # Type-Token Ratio (TTR)
    all_words = []
    from collections import Counter
    for caption in generated_captions:
        words = caption.lower().split()
        all_words.extend(words)

    unique_words = set(all_words)
    ttr = len(unique_words) / len(all_words) if all_words else 0

    # Self-BLEU (diversity measure) - calculate on a subset for efficiency
    self_bleu = 0
    try:
        if corpus_bleu and word_tokenize and SmoothingFunction:
            smooth = SmoothingFunction().method1
            self_bleu_scores = []

            # Sample a subset of generated captions for Self-BLEU to avoid long computation
            sample_size = min(1000, len(generated_captions))
            sampled_captions = random.sample(generated_captions, sample_size) if len(generated_captions) > sample_size else generated_captions

            for i, caption in enumerate(sampled_captions):
                # References are all other captions in the sample
                references_for_self_bleu = [[word_tokenize(other_cap.lower())]
                                            for j, other_cap in enumerate(sampled_captions) if i != j]
                hypothesis = word_tokenize(caption.lower())

                if references_for_self_bleu and hypothesis: # Ensure there are references and hypothesis tokens
                    # Calculate sentence BLEU with other captions as references
                    score = corpus_bleu(references_for_self_bleu, [hypothesis], smoothing_function=smooth)
                    self_bleu_scores.append(score)

            self_bleu = np.mean(self_bleu_scores) if self_bleu_scores else 0
        else:
            logger.warning("NLTK not fully available for Self-BLEU calculation. Skipping.")
    except Exception as e:
        logger.error(f"Error calculating Self-BLEU: {e}")
        self_bleu = 0

    # Caption uniqueness
    unique_captions = len(set(generated_captions))
    uniqueness_ratio = unique_captions / len(generated_captions) if len(generated_captions) > 0 else 0

    return {
        'type_token_ratio': ttr,
        'self_bleu': self_bleu,
        'unique_captions_count': unique_captions,
        'caption_uniqueness_ratio': uniqueness_ratio
    }


def calculate_perplexity(model, data_loader, vocabulary, device):
    """
    Calculates the perplexity of the model on a given dataset.
    Perplexity measures how well a probability model predicts a sample. Lower is better.
    Args:
        model (nn.Module): The trained image captioning model.
        data_loader (DataLoader): DataLoader for the dataset.
        vocabulary (COCOVocabulary): The vocabulary object.
        device (torch.device): Device to run calculation on.
    Returns:
        float: Perplexity score, or infinity if calculation fails.
    """
    model.eval()
    total_loss = 0
    total_words = 0

    # Use CrossEntropyLoss with sum reduction to get the sum of losses over all tokens
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocabulary.word2idx['<PAD>'], reduction='sum')

    with torch.no_grad():
        for images, captions_from_loader, caption_lengths_from_loader, _ in tqdm(data_loader, desc="Calculating Perplexity"):
            images = images.to(device)
            captions_for_model = captions_from_loader.to(device)
            caption_lengths_for_model = caption_lengths_from_loader.to(device)

            # Forward pass to get scores
            scores, caps_sorted, decode_lengths, _, _ = model(images, captions_for_model, caption_lengths_for_model)

            # Prepare targets: remove <START> token and slice to match the sequence length of 'scores'.
            # scores are (batch_size, max_decode_len_in_batch, vocab_size)
            # targets should be (batch_size, max_decode_len_in_batch)
            targets = caps_sorted[:, 1:scores.size(1) + 1].contiguous().view(-1) # Flatten targets
            scores_flat = scores.view(-1, scores.size(-1)) # Flatten scores

            loss = criterion(scores_flat, targets) # Calculate loss for all tokens
            total_loss += loss.item()

            # Count non-padded words in the targets that were actually used for loss.
            num_valid_targets_in_batch = targets.ne(vocabulary.word2idx['<PAD>']).sum().item()
            total_words += num_valid_targets_in_batch

    if total_words == 0:
        logger.warning("No valid words found to calculate perplexity (total_words is 0). Returning inf.")
        return float('inf')

    avg_loss_per_word = total_loss / total_words

    # Perplexity is exp(average negative log-likelihood)
    try:
        perplexity = math.exp(avg_loss_per_word)
    except OverflowError: # Handle cases where avg_loss_per_word is very large, leading to overflow
        perplexity = float('inf')

    return perplexity


def print_evaluation_results(metrics):
    """
    Prints the evaluation results in a formatted way.
    Args:
        metrics (dict): Dictionary containing all evaluation metrics.
    """
    logger.info("\n"+"="*60)
    logger.info("                      EVALUATION RESULTS")
    logger.info("="*60)

    # BLEU Scores
    if 'BLEU-1' in metrics:
        logger.info(f"\nBLEU Scores:")
        logger.info(f"  BLEU-1: {metrics['BLEU-1']:.4f}")
        logger.info(f"  BLEU-2: {metrics['BLEU-2']:.4f}")
        logger.info(f"  BLEU-3: {metrics['BLEU-3']:.4f}")
        logger.info(f"  BLEU-4: {metrics['BLEU-4']:.4f}")

    # Other metrics
    if 'METEOR' in metrics and metrics['METEOR'] is not None:
        logger.info(f"\nMETEOR Score: {metrics['METEOR']:.4f}")

    if 'ROUGE-L' in metrics and metrics['ROUGE-L'] is not None:
        logger.info(f"ROUGE-L Score: {metrics['ROUGE-L']:.4f}")

    if 'CIDEr' in metrics and metrics['CIDEr'] is not None:
        logger.info(f"CIDEr Score: {metrics['CIDEr']:.4f}")

    if 'Perplexity' in metrics and metrics['Perplexity'] is not None:
        logger.info(f"Perplexity: {metrics['Perplexity']:.2f}")

    # Length Statistics
    logger.info(f"\nLength Statistics:")
    logger.info(f"  Avg Generated Length: {metrics.get('avg_generated_length', 0):.2f}")
    logger.info(f"  Avg Reference Length: {metrics.get('avg_reference_length', 0):.2f}")
    logger.info(f"  Length Difference (Gen - Ref): {metrics.get('length_difference', 0):.2f}")
    logger.info(f"  Std Dev Generated Length: {metrics.get('length_std_generated', 0):.2f}")
    logger.info(f"  Std Dev Reference Length: {metrics.get('length_std_reference', 0):.2f}")

    # Diversity Metrics
    logger.info(f"\nDiversity Metrics:")
    logger.info(f"  Type-Token Ratio: {metrics.get('type_token_ratio', 0):.4f}")
    logger.info(f"  Caption Uniqueness Ratio: {metrics.get('caption_uniqueness_ratio', 0):.4f}")
    logger.info(f"  Self-BLEU (Higher is lower diversity): {metrics.get('self_bleu', 0):.4f}")
    logger.info(f"  Unique Captions Count: {metrics.get('unique_captions_count', 0)}")

    # Vocabulary Usage
    logger.info(f"\nVocabulary Usage:")
    logger.info(f"  Unique Words Used in Generated Captions: {metrics.get('unique_words_used', 0)}")
    logger.info(f"  Vocabulary Coverage (Used / Total): {metrics.get('vocabulary_coverage', 0):.4f}")
    if 'most_common_generated_words' in metrics:
        logger.info(f"  Most Common Generated Words: {metrics['most_common_generated_words']}")

    logger.info(f"\nEvaluation Info:")
    eval_info = metrics.get('evaluation_info', {})
    logger.info(f"  Total Samples Evaluated: {eval_info.get('total_samples', 0)}")
    logger.info(f"  Evaluation Time: {eval_info.get('evaluation_time_seconds', 0):.2f}s")
    logger.info(f"  Test Data Path: {eval_info.get('test_data_path', 'N/A')}")
    logger.info(f"  Image Directory Used: {eval_info.get('image_dir_used', 'N/A')}")
    logger.info(f"  Device: {eval_info.get('device', 'unknown')}")
    logger.info(f"  Model Architecture: {eval_info.get('model_architecture', 'N/A')}")

    logger.info("="*60)


def save_evaluation_results(metrics, generated_captions, reference_captions, image_ids, output_dir='evaluation_results'):
    """
    Saves detailed evaluation results to JSON files.
    Args:
        metrics (dict): Dictionary containing all evaluation metrics.
        generated_captions (list of str): List of generated captions.
        reference_captions (list of str): List of reference captions.
        image_ids (list): List of original image IDs corresponding to captions.
        output_dir (str): Directory to save the results.
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            serializable_metrics[key] = int(value)
        else:
            serializable_metrics[key] = value

    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    # Save generated captions and their references along with image_ids
    captions_data = []
    for img_id, gen_cap, ref_cap in zip(image_ids, generated_captions, reference_captions):
        captions_data.append({
            'image_id': img_id,
            'generated_caption': gen_cap,
            'reference_caption': ref_cap
        })

    captions_path = os.path.join(output_dir, 'captions.json')
    with open(captions_path, 'w') as f:
        json.dump(captions_data, f, indent=2)

    logger.info(f"\nDetailed evaluation results saved to: {output_dir}/")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Captions saved to: {captions_path}")


def perform_evaluation(model, vocabulary, test_config):
    """
    Performs comprehensive evaluation of the image captioning model on a test set.

    Args:
        model (nn.Module): The trained image captioning model.
        vocabulary (COCOVocabulary): The vocabulary object used by the model.
        test_config (dict): Configuration dictionary for evaluation.

    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    logger.info("Starting comprehensive model evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info(f"Model set to evaluation mode on device: {device}")

    # Data paths for evaluation from config
    data_folder = test_config['data_folder']
    test_image_folder = test_config['test_image_folder']
    test_caption_file = test_config['test_caption_file']

    if not os.path.exists(test_caption_file):
        raise FileNotFoundError(f"Test caption file not found: {test_caption_file}")

    # Construct the correct image directory path for evaluation
    image_dir_for_eval = os.path.join(data_folder, test_image_folder)
    if not os.path.exists(image_dir_for_eval):
        logger.error(f"Image directory for evaluation not found: {image_dir_for_eval}")
        logger.error("Please ensure COCO images are extracted to the correct path.")
        return {'error': f'Image directory not found: {image_dir_for_eval}'}

    logger.info(f"Attempting to load evaluation images from directory: {image_dir_for_eval}")

    # Create test dataset
    test_dataset = COCODataset(
        image_dir=image_dir_for_eval,
        caption_file=test_caption_file,
        vocabulary=vocabulary, # Use the vocabulary from training
        max_caption_length=test_config.get('max_caption_length', 20),
        subset_size=test_config.get('test_subset_size'),
        transform=get_eval_transform() # Use standard eval transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_config.get('eval_batch_size', 1), # Batch size 1 is crucial for beam search
        shuffle=False, # Do not shuffle test data
        num_workers=test_config.get('num_workers', 2),
        pin_memory=True
    )

    logger.info(f"Test dataset size: {len(test_dataset)}")

    if len(test_dataset) == 0:
        logger.warning("Test dataset is empty. Evaluation will not produce meaningful results.")
        return {'error': 'Test dataset is empty', 'image_dir_checked': image_dir_for_eval}

    # Generate captions for all test images
    logger.info("Generating captions for evaluation set...")
    generated_captions_list = []
    reference_captions_list = []
    image_ids_list = [] # To store original image IDs for mapping

    eval_start_time = time.time()

    with torch.no_grad(): # Disable gradient calculations
        for i, (images, caption_indices_batch, _, original_image_ids_batch) in enumerate(tqdm(test_loader, desc="Generating captions")):
            images = images.to(device)

            for j in range(images.size(0)): # Iterate through batch (should be size 1 if eval_batch_size=1)
                image_tensor_single = images[j] # Get single image tensor from batch

                # Generate caption using the model's beam search method
                generated_caption = model.generate_caption(
                    image_tensor_single, vocabulary, device,
                    beam_size=test_config.get('beam_size', 5),
                    max_length=test_config.get('max_caption_length', 20)
                )

                # Convert reference caption indices back to string
                reference_caption_str = vocabulary.indices_to_caption(caption_indices_batch[j].cpu().numpy())

                generated_captions_list.append(generated_caption)
                reference_captions_list.append(reference_caption_str)
                # Ensure image_id is a string or compatible type for JSON serialization
                image_ids_list.append(str(original_image_ids_batch[j].item()))

    eval_time = time.time() - eval_start_time
    logger.info(f"Caption generation completed in {eval_time:.2f} seconds for {len(generated_captions_list)} images.")

    if not generated_captions_list or not reference_captions_list:
        logger.error("No captions were generated or no reference captions were loaded. Cannot calculate metrics.")
        return {'error': 'No generated or reference captions available for metric calculation.'}

    # Calculate evaluation metrics
    logger.info("Calculating evaluation metrics...")
    metrics = {}

    # Calculate standard metrics
    bleu_scores = calculate_bleu_scores_detailed(reference_captions_list, generated_captions_list)
    metrics.update(bleu_scores)

    meteor_score_val = calculate_meteor_score(reference_captions_list, generated_captions_list)
    if meteor_score_val is not None:
        metrics['METEOR'] = meteor_score_val

    rouge_score_val = calculate_rouge_l_score(reference_captions_list, generated_captions_list)
    if rouge_score_val is not None:
        metrics['ROUGE-L'] = rouge_score_val

    cider_score_val = calculate_cider_score(reference_captions_list, generated_captions_list)
    if cider_score_val is not None:
        metrics['CIDEr'] = cider_score_val

    # Calculate length and diversity statistics
    length_stats = calculate_length_statistics(generated_captions_list, reference_captions_list)
    metrics.update(length_stats)

    vocab_stats = calculate_vocabulary_statistics(generated_captions_list, vocabulary)
    metrics.update(vocab_stats)

    diversity_stats = calculate_diversity_metrics(generated_captions_list)
    metrics.update(diversity_stats)

    # Calculate perplexity
    try:
        perplexity = calculate_perplexity(model, test_loader, vocabulary, device)
        metrics['Perplexity'] = perplexity
    except Exception as e:
        logger.error(f"Could not calculate perplexity: {e}")

    # Add meta information about the evaluation run
    metrics['evaluation_info'] = {
        'total_samples': len(generated_captions_list),
        'evaluation_time_seconds': eval_time,
        'test_data_path': test_caption_file,
        'image_dir_used': image_dir_for_eval,
        'device': str(device),
        'model_architecture': 'ResNet50 Encoder + LSTM Decoder with Attention',
        'beam_size_for_inference': test_config.get('beam_size', 5),
        'max_caption_length_for_inference': test_config.get('max_caption_length', 20)
    }

    # Print and save results
    print_evaluation_results(metrics)
    save_evaluation_results(metrics, generated_captions_list, reference_captions_list, image_ids_list, output_dir=test_config.get('eval_output_dir', 'output/evaluation_results'))

    return metrics


if __name__ == '__main__':
    # When `evaluation.py` is run directly, it will perform evaluation.
    from .config import EVALUATION_CONFIG, update_config_with_latest_model
    import pickle # For loading vocabulary

    logger.info("Starting model evaluation process...")

    # Load the vocabulary first
    VOCABULARY_FILE_PATH = 'output/vocabulary.pkl' # Path to the vocabulary file
    if not os.path.exists(VOCABULARY_FILE_PATH):
        logger.error(f"Vocabulary not found at {VOCABULARY_FILE_PATH}. Please train the model first or provide a pre-trained vocabulary.")
        exit() # Exit if vocabulary is not found
    try:
        with open(VOCABULARY_FILE_PATH, 'rb') as f:
            vocabulary = pickle.load(f)
        logger.info(f"Loaded vocabulary from {VOCABULARY_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error loading vocabulary from {VOCABULARY_FILE_PATH}: {e}")
        exit()

    # Update evaluation config to point to the latest trained model
    update_config_with_latest_model(EVALUATION_CONFIG)
    model_path = EVALUATION_CONFIG.get('model_path')

    if not model_path or not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found at {model_path}. Please train the model or specify a valid model_path in config.py.")
        exit()

    try:
        # Load the model state dict and config from the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config_from_checkpoint = checkpoint.get('config', {})

        # Initialize model with parameters from checkpoint config (or defaults if missing)
        eval_model = ImageCaptioningModel(
            vocab_size=vocabulary.vocab_size, # Use the loaded vocabulary's size
            embed_dim=model_config_from_checkpoint.get('embed_dim', 256),
            attention_dim=model_config_from_checkpoint.get('attention_dim', 256),
            decoder_dim=model_config_from_checkpoint.get('decoder_dim', 256),
            dropout=0.0, # No dropout during evaluation
            fine_tune_encoder=False, # No fine-tuning during evaluation
            max_caption_length=model_config_from_checkpoint.get('max_caption_length', 20)
        )
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded successfully from {model_path} for evaluation.")

        # Perform the comprehensive evaluation
        eval_metrics = perform_evaluation(eval_model, vocabulary, EVALUATION_CONFIG)
        logger.info("Model Evaluation Complete!")

    except FileNotFoundError as e:
        logger.error(f"Error during evaluation setup: {e}")
        logger.error("Please ensure the model path and data paths are correct.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
