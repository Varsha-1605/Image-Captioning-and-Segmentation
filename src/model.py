import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models # Used for ResNet50

from .utils import get_logger # Import logger

logger = get_logger(__name__)


class EncoderCNN(nn.Module):
    """
    Encoder using a pre-trained ResNet50 model.
    The output feature maps are adaptively pooled to a fixed size
    and then reshaped for the attention mechanism in the decoder.
    """
    def __init__(self, encoded_image_size=14, fine_tune=True):
        """
        Initializes the EncoderCNN.
        Args:
            encoded_image_size (int): The spatial dimension (e.g., 14x14) to which
                                      the feature maps will be adaptively pooled.
            fine_tune (bool): If True, allows the parameters of the pre-trained
                              ResNet to be updated during training. If False, they are frozen.
        """
        super(EncoderCNN, self).__init__()
        self.encoded_image_size = encoded_image_size

        # Load pre-trained ResNet50 and remove the final classification layer
        # We use the default recommended weights for ResNet50.
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove the average pooling layer and the fully connected layer at the end.
        # We want the feature maps before these layers for spatial attention.
        # The `modules` list will contain layers up to `layer4` (the last convolutional block).
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Freeze parameters of the pre-trained ResNet if fine_tune is False.
        # This prevents updating their weights during training.
        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
            logger.info("ResNet encoder base layers are frozen.")
        else:
            logger.info("ResNet encoder base layers are fine-tuning enabled.")

        # Adaptive pool to a fixed size (e.g., 14x14).
        # This ensures a consistent spatial dimension for the feature maps,
        # regardless of the input image size, useful for attention.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # The output feature dimension from ResNet50 before the last avg pool/fc is 2048.
        self.encoder_dim = 2048

    def forward(self, images):
        """
        Forward pass through the ResNet encoder.
        Args:
            images (torch.Tensor): Input images, shape (batch_size, 3, H, W).
        Returns:
            torch.Tensor: Encoded image features,
                          shape (batch_size, encoder_dim, encoded_image_size, encoded_image_size).
        """
        # Pass images through the ResNet feature extractor
        out = self.resnet(images)
        # Apply adaptive pooling to get a fixed spatial size (e.g., 14x14)
        out = self.adaptive_pool(out)

        # The output shape is (batch_size, encoder_dim, encoded_image_size, encoded_image_size)
        return out


class Attention(nn.Module):
    """
    Additive Attention Mechanism (Bahdanau style).
    Calculates attention weights based on encoded image features and decoder's hidden state.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Initializes the Attention module.
        Args:
            encoder_dim (int): Feature size of encoded images (e.g., 2048 for ResNet50).
            decoder_dim (int): Hidden state size of the decoder LSTM.
            attention_dim (int): Size of the linear layers within the attention mechanism.
        """
        super(Attention, self).__init__()
        # Linear layer to transform encoder output for attention calculation
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # Linear layer to transform decoder hidden state for attention calculation
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # Linear layer to calculate attention "score" (or energy)
        # This layer projects the combined features to a single scalar per pixel.
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        # Softmax over the "num_pixels" dimension to get attention weights that sum to 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass through the attention mechanism.
        Args:
            encoder_out (torch.Tensor): Encoded images, shape (batch_size, num_pixels, encoder_dim).
            decoder_hidden (torch.Tensor): Previous decoder hidden state, shape (batch_size, decoder_dim).
        Returns:
            tuple:
                - attention_weighted_encoding (torch.Tensor): Context vector,
                                                             shape (batch_size, encoder_dim).
                - alpha (torch.Tensor): Attention weights (probability distribution over pixels),
                                        shape (batch_size, num_pixels).
        """
        # Transform encoder output: (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # Transform decoder hidden state, then unsqueeze to (batch_size, 1, attention_dim)
        # for broadcasting during addition with att1
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)

        # Calculate attention scores: (batch_size, num_pixels)
        # Sum of transformed encoder output and transformed decoder hidden state,
        # passed through ReLU and then a linear layer to get a single score per pixel.
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)

        # Apply softmax to get attention weights (alpha): (batch_size, num_pixels)
        alpha = self.softmax(att)

        # Calculate attention-weighted encoding: (batch_size, encoder_dim)
        # This is the context vector, a weighted sum of the encoder features.
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Attention mechanism.
    Generates captions word by word, using the attention-weighted image features
    and previously generated words.
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=2048, dropout=0.5):
        """
        Initializes the DecoderWithAttention.
        Args:
            attention_dim (int): Size of the attention linear layer.
            embed_dim (int): Dimension of word embeddings.
            decoder_dim (int): Hidden state size of the decoder LSTM.
            vocab_size (int): Total size of the vocabulary.
            encoder_dim (int): Feature size of encoded images (default 2048 for ResNet50).
            dropout (float): Dropout rate for regularization.
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout

        # Attention network
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding_dropout = nn.Dropout(self.dropout_rate)

        # LSTMCell for decoding
        # Input to LSTMCell is the concatenation of word embedding and attention-weighted encoding
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # Linear layers to initialize the LSTM's hidden and cell states from the encoder output
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Linear layer to create a "gate" for the attention-weighted encoding (Visual Sentinel)
        # This f_beta gate allows the model to decide how much of the attention-weighted
        # context to use for generating the next word, enabling it to ignore irrelevant visual information.
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid() # Activation for the gate

        # Linear layer to project decoder output to vocabulary size (scores for each word)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        # Initialize some weights
        self.init_weights()

        # A placeholder for max caption length during inference/generation
        # This will typically be set by the calling model or config
        self.max_caption_length_for_inference = 20

    def init_weights(self):
        """Initializes some parameters with values from the uniform distribution."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads pre-trained embeddings into the embedding layer.
        Args:
            embeddings (torch.Tensor): A tensor of pre-trained word embeddings.
        """
        self.embedding.weight = nn.Parameter(embeddings)
        # Optionally, freeze embeddings if they are pre-trained and you don't want to fine-tune them
        # self.embedding.weight.requires_grad = False

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allows or disallows fine-tuning of the embedding layer.
        Args:
            fine_tune (bool): If True, embedding weights are trainable. If False, they are frozen.
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates initial hidden and cell states for the LSTM from the encoded image.
        Uses the mean of the encoder output features to initialize the LSTM states.
        Args:
            encoder_out (torch.Tensor): Encoded images, shape (batch_size, num_pixels, encoder_dim).
        Returns:
            tuple: (hidden state (h), cell state (c)), each of shape (batch_size, decoder_dim).
        """
        # Calculate mean of encoder output across pixels
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out) # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out) # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass through the decoder during training.
        Args:
            encoder_out (torch.Tensor): Encoded images from CNN,
                                        shape (batch_size, encoder_dim, enc_image_size_H, enc_image_size_W).
            encoded_captions (torch.Tensor): Captions, shape (batch_size, max_caption_length).
            caption_lengths (torch.Tensor): Actual lengths of captions (before padding), shape (batch_size,).
        Returns:
            tuple:
                - predictions (torch.Tensor): Predicted word scores,
                                              shape (batch_size, max_decode_length_in_batch, vocab_size).
                - encoded_captions (torch.Tensor): Captions sorted by length.
                - decode_lengths (list): Actual decoding lengths for each caption in the batch.
                - alphas (torch.Tensor): Attention weights,
                                         shape (batch_size, max_decode_length_in_batch, num_pixels).
                - sort_ind (torch.Tensor): Indices used to sort the batch.
        """
        batch_size = encoder_out.size(0)
        enc_image_h = encoder_out.size(2)
        enc_image_w = encoder_out.size(3)
        num_pixels = enc_image_h * enc_image_w

        # Reshape encoder_out for attention: (batch_size, num_pixels, encoder_dim)
        # Permute from (N, C, H, W) to (N, H, W, C) then flatten H*W
        encoder_out = encoder_out.permute(0, 2, 3, 1).contiguous()
        encoder_out = encoder_out.view(batch_size, num_pixels, self.encoder_dim)

        # Sort input data by decreasing lengths for packed sequences.
        # This is crucial for efficient processing with `pack_padded_sequence`.
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind] # Apply sorting to encoder output
        encoded_captions = encoded_captions[sort_ind] # Apply sorting to captions

        # Embedding: (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)
        embeddings = self.embedding_dropout(embeddings)

        # Initialize LSTM state (h, c) from the mean of encoder output
        h, c = self.init_hidden_state(encoder_out)

        # Create tensors to hold word predictions and attention weights.
        # We predict up to (max_caption_length - 1) words (excluding the <START> token).
        decode_lengths = (caption_lengths - 1).tolist() # Lengths of sequences to decode
        max_decode_length = max(decode_lengths) # Max length in the current batch

        predictions = torch.zeros(batch_size, max_decode_length, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_length, num_pixels).to(encoder_out.device)

        # For each time step in the decoding process
        for t in range(max_decode_length):
            # Get batch size for current time step.
            # Sequences are padded, so some might finish early.
            batch_size_t = sum([l > t for l in decode_lengths])

            # Apply attention mechanism to the active sequences in the batch
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            # Apply sigmoid gate to attention-weighted encoding (Visual Sentinel)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # Perform one step of LSTM decoding
            # Input to LSTM: (current_word_embedding + attention_weighted_encoding)
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            # Predict next word using the fully connected layer
            preds = self.fc(self.dropout_layer(h))

            # Store predictions and attention weights for the current time step
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class ImageCaptioningModel(nn.Module):
    """
    Complete Image Captioning Model integrating EncoderCNN and DecoderWithAttention.
    Provides methods for both training (forward pass) and inference (caption generation).
    """
    def __init__(self, vocab_size, embed_dim=256, attention_dim=256, encoder_dim=2048,
                 decoder_dim=256, dropout=0.5, fine_tune_encoder=True, max_caption_length=20):
        """
        Initializes the ImageCaptioningModel.
        Args:
            vocab_size (int): Total size of the vocabulary.
            embed_dim (int): Dimension of word embeddings.
            attention_dim (int): Size of the attention linear layer.
            encoder_dim (int): Feature size of encoded images (default 2048 for ResNet50).
            decoder_dim (int): Hidden state size of the decoder LSTM.
            dropout (float): Dropout rate for regularization.
            fine_tune_encoder (bool): If True, allows the encoder parameters to be updated.
            max_caption_length (int): Maximum length of generated captions during inference.
        """
        super(ImageCaptioningModel, self).__init__()

        # Initialize the Encoder (ResNet50-based)
        self.encoder = EncoderCNN(encoded_image_size=14, fine_tune=fine_tune_encoder)
        # Ensure encoder_dim matches ResNet50 output dimension
        self.encoder_dim = self.encoder.encoder_dim # This will be 2048

        # Initialize the Decoder with Attention
        self.decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=self.encoder_dim, # Pass the correct encoder_dim
            dropout=dropout
        )
        self.decoder.max_caption_length_for_inference = max_caption_length # Set max length for inference

    def forward(self, images, captions, caption_lengths):
        """
        Forward pass through the complete model for training.
        Args:
            images (torch.Tensor): Input images.
            captions (torch.Tensor): Target captions.
            caption_lengths (torch.Tensor): Actual lengths of captions.
        Returns:
            tuple: (predictions, encoded_captions, decode_lengths, alphas, sort_ind)
                   as returned by the decoder's forward pass.
        """
        encoder_out = self.encoder(images) # Encode images
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.decoder(
            encoder_out, captions, caption_lengths # Decode captions
        )
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def generate_caption(self, image_tensor, vocabulary, device, beam_size=5, max_length=None):
        """
        Performs beam search to generate a caption for a single image.
        This method is now part of the ImageCaptioningModel class.
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor (3, H, W). NOT batched.
            vocabulary (COCOVocabulary): Vocabulary object.
            device (torch.device): Device to run the model on (cpu/cuda).
            beam_size (int): Size of beam for beam search.
            max_length (int, optional): Maximum length of the generated caption.
                                        If None, uses self.decoder.max_caption_length_for_inference.
        Returns:
            str: Generated caption string.
        """
        self.eval() # Set model to evaluation mode
        # Use the max_length from config if provided, otherwise fallback to model's default
        current_max_length = max_length if max_length is not None else self.decoder.max_caption_length_for_inference

        with torch.no_grad():
            # Add batch dimension and move to device for the encoder
            # image_tensor goes from (C, H, W) to (1, C, H, W)
            image_tensor_batched = image_tensor.unsqueeze(0).to(device)

            # Get encoder output: (1, encoder_dim, encoded_image_size, encoded_image_size)
            encoder_output_from_cnn = self.encoder(image_tensor_batched)

            # Reshape encoder_output_from_cnn to (1, num_pixels, encoder_dim) for attention
            # Permute from (N, C, H, W) to (N, H, W, C) then flatten H*W
            encoder_out = encoder_output_from_cnn.permute(0, 2, 3, 1).contiguous()
            encoder_out = encoder_out.view(1, -1, self.encoder_dim) # (1, num_pixels, encoder_dim)

            # Expand for beam search: (beam_size, num_pixels, encoder_dim)
            encoder_out = encoder_out.expand(beam_size, encoder_out.size(1), encoder_out.size(2))

            # Tensor to store top k previous words at each step; initialized with <START> token for all beams
            k_prev_words = torch.LongTensor([[vocabulary.word2idx['<START>']]] * beam_size).to(device)

            # Tensor to store top k sequences; initially just the <START> token
            seqs = k_prev_words

            # Tensor to store top k sequences' scores (log probabilities); initially all zeros
            top_k_scores = torch.zeros(beam_size, 1).to(device)

            # Lists to store completed captions and their scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Initialize hidden state and cell state for LSTM
            # encoder_out is already expanded for beam_size, so init_hidden_state will work
            h, c = self.decoder.init_hidden_state(encoder_out)

            # Start decoding loop
            step = 1
            while True:
                # Get embeddings for the previously predicted words
                embeddings = self.decoder.embedding(k_prev_words).squeeze(1) # (beam_size, embed_dim)

                # Calculate attention-weighted encoding and attention weights
                attention_weighted_encoding, alpha = self.decoder.attention(encoder_out, h)
                # Apply visual sentinel gate
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                # Perform one step of LSTM decoding
                h, c = self.decoder.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1),
                    (h, c)
                ) # (beam_size, decoder_dim)

                # Get scores for the next word
                scores = self.decoder.fc(h) # (beam_size, vocab_size)
                scores = F.log_softmax(scores, dim=1) # Convert to log-probabilities

                # Add current scores to previous scores for beam search
                scores = top_k_scores.expand_as(scores) + scores # (beam_size, vocab_size)

                # For the first step, all k generated words are from the same parent (<START>).
                # For subsequent steps, they are from different parents.
                if step == 1:
                    # For the first step, select top 'beam_size' words from the first beam's scores
                    top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)  # (beam_size)
                else:
                    # Flatten scores to find the top 'beam_size' overall (from all current beams)
                    top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)  # (beam_size)

                # Convert flattened indices to actual row (previous word's beam index)
                # and column (next word's vocabulary index) indices
                prev_word_inds = top_k_words // vocabulary.vocab_size  # (beam_size)
                next_word_inds = top_k_words % vocabulary.vocab_size  # (beam_size)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (beam_size, step + 1)

                # Identify completed sequences (where <END> is generated)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds)
                                   if next_word != vocabulary.word2idx['<END>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Add complete sequences to their lists
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])

                # Update beam_size: number of active beams for the next step
                beam_size -= len(complete_inds)
                if beam_size == 0: # If all beams complete, break
                    break

                # Filter seqs, hidden states, cell states, scores, and previous words for incomplete sequences
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1) # Reshape for next step
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]] # Propagate encoder_out for active beams

                # Break if maximum caption length is exceeded
                if step > current_max_length:
                    break
                step += 1

            # If no complete captions were found (e.g., all beams exceeded max_length before <END>),
            # pick the best incomplete sequence found so far.
            if not complete_seqs:
                # Take the best sequence among the currently active (incomplete) beams
                final_seqs = seqs.tolist()
                final_scores = top_k_scores.squeeze(1).tolist()
                if not final_seqs: # Fallback if even no incomplete sequences are available (shouldn't happen)
                    return ""
                i = final_scores.index(max(final_scores))
                best_seq = final_seqs[i]
            else:
                # Find the best caption among all completed sequences based on score
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                best_seq = complete_seqs[i]

            # Convert the best sequence of indices back to a human-readable caption
            return vocabulary.indices_to_caption(best_seq)
