import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchcrf

# This script contains the implementation of the second-level encoder networks including the custom classificaiton head for the sentence-level models
# Refer to Section 5.2 of the report for more details

# Base class implemented with functions shared across different encoder networks
class BaseSentenceTaggingModel(nn.Module):
    def __init__(self
                 , num_labels
                 , label_to_index
                 , embedding_dim
                 , num_detailed_labels_per_level=[]
                 , detailed_label_weights=None
                 , alpha=0.5
                 , loss_function="CrossEntropyLoss"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_index = label_to_index
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.num_detailed_labels_per_level = num_detailed_labels_per_level
        self.alpha = alpha

        # Here other loss functions are conceivable as well
        if loss_function == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # Initialise detailed label classifiers to be added on top of encoder network
        # Note: We do not initialise the classifier for the main objective here, as this might be dependent on the architecture of the encoder network
        self.detailed_label_classifiers = nn.ModuleList()
        if num_detailed_labels_per_level:
            for num_labels in num_detailed_labels_per_level:
                self.detailed_label_classifiers.append(nn.Linear(self.embedding_dim, num_labels))
            
            # Normalise detailed label weights (typically these values should already be provided in a sensible manner by the user)
            if detailed_label_weights is None:
                self.detailed_label_weights = [1.0 / len(num_detailed_labels_per_level)] * len(num_detailed_labels_per_level)
            else:
                total_weight = sum(detailed_label_weights)
                self.detailed_label_weights = [weight / total_weight for weight in detailed_label_weights]
                
    # To be implemented in the respective encoder networks
    def forward(self, *inputs):
        raise NotImplementedError("Encoder network should implement this!")

    # Calculate loss for the model
    def calculate_loss(self, logits, labels, detailed_logits, detailed_labels):
        
        # Loss for main objective
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        # Loss for auxiliary objective
        if detailed_labels is not None:
            detailed_loss = 0
            bio_predictions = torch.argmax(logits, dim=-1) # get bio predictions (i.e. main objective predictions) to inform detailed label loss
            
            # Iterate over detailed label classifiers and calculate loss for each (i.e. each level of the ontology label hierarchy)
            for i, (level_logits, weight) in enumerate(zip(detailed_logits, self.detailed_label_weights)):
                level_labels = detailed_labels[:, :, i].reshape(-1) # get labels for current level
                # Idea: if outisde of region, we do not need to calculate detailed labels, because there won't be any
                detailed_labels_masked = torch.where(bio_predictions.view(-1) == self.label_to_index.get("O"),  # get index that corresponds to "O"
                                                    torch.full_like(level_labels, -1),  # Set to -1
                                                    level_labels)
                
                # Use mask to only consider valid labels
                valid_idx = detailed_labels_masked != -1
                
                # Calculate weighted loss as per detailed_label_weights
                if valid_idx.any():
                    valid_logits = level_logits.view(-1, level_logits.size(-1))[valid_idx]
                    valid_labels = detailed_labels_masked[valid_idx]
                    level_loss = self.loss_fn(valid_logits, valid_labels) * weight
                    detailed_loss += level_loss
                    
            # Calculate total loss
            loss = self.alpha * loss + (1 - self.alpha) * detailed_loss
        return loss


# First two options for encoder networks: Bi-LSTM and Bi-LSTM with CRF (we call this architecture SentenceTaggingBiLSTM)
# For more details refer to Appendix A.1 in the report

# Inheritance from BaseSentenceTaggingModel (defined above)
class SentenceTaggingBiLSTM(BaseSentenceTaggingModel):
    def __init__(self
                 , embedding_dim
                 , hidden_dim
                 , nlayers
                 , num_labels
                 , label_to_index
                 , bidirectional=True
                 , dropout=0.5
                 , use_crf=False
                 , **kwargs):
        
        # Calling constructor of parent class above
        super(SentenceTaggingBiLSTM, self).__init__(num_labels=num_labels
                                                    , label_to_index=label_to_index
                                                    , embedding_dim=embedding_dim
                                                    , **kwargs)
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.use_crf = use_crf
        
        # Setup LSTM layer using PyTorch implementation
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, nlayers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Classifier for the main labels
        self.fc = nn.Linear(output_dim, num_labels)

        # In case of Bi-LSTM with CRF, we use the torchcrf implementation to add on top of the classifier
        if use_crf:
            self.crf = torchcrf.CRF(num_labels)

        self.dropout = nn.Dropout(dropout)

        # Ensure detailed_label_classifiers are correctly dimensioned based on LSTM output
        if self.num_detailed_labels_per_level:
            for i, num_labels in enumerate(self.num_detailed_labels_per_level):
                # Adjusting each classifier to match LSTM output dimensions
                self.detailed_label_classifiers[i] = nn.Linear(output_dim, num_labels)

    def forward(self, src, labels=None, detailed_labels=None, src_key_padding_mask=None):
        
        # Make sure all relevant tensors are on the same device
        src = src.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if detailed_labels is not None:
            detailed_labels = detailed_labels.to(self.device)
            
        # Apply dropout to the input
        src = self.dropout(src)
        
        # Get LSTM output
        lstm_out, hidden_state = self.lstm(src)
        lstm_out = lstm_out.to(self.device)
        logits = self.fc(lstm_out)
        outputs = (logits,)
        
        # Get detailed logits if detailed_label_classifiers are used (auxiliary objective)
        detailed_logits = []
        if self.num_detailed_labels_per_level:
            for classifier in self.detailed_label_classifiers:
                detailed_logits.append(classifier(lstm_out))
            if detailed_logits:
                outputs = outputs + (detailed_logits,)

        # if model is Bi-LSTM+CRF, use CRF to calculate loss
        if self.use_crf:
            mask = src_key_padding_mask if src_key_padding_mask is not None else torch.ones_like(labels, dtype=torch.bool) # mask for padding
            mask = mask.to(logits.device)
            if labels is not None:
                labels = labels.to(logits.device)
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')  # Use CRF to compute loss
                return (loss,) + outputs
            else:
                # decode most liekely sequence of labels using CRF
                predicted_labels = self.crf.decode(logits, mask=mask)
                return (torch.tensor(predicted_labels, device=self.device),) # not used in training, only for inference, if relevant
        elif labels is not None:
            # Calculate loss without CRF
            loss = self.calculate_loss(logits, labels, detailed_logits, detailed_labels)
            return (loss,) + outputs
        return outputs  # (logits, detailed_logits) or (logits,) if no detailed classifiers are used

    
# Third option for encoder network: Transformer-based model (we call this architecture STATO, which stands for Sentence Transformer with Auxiliary Training Objective)
# STATO also inherits from BaseSentenceTaggingModel
class STATO(BaseSentenceTaggingModel):
    def __init__(self
                , embedding_dim
                , nhead
                , hidden_dim
                , nlayers
                , dropout
                , num_labels
                , label_to_index
                , num_detailed_labels_per_level
                , detailed_label_weights=None
                , alpha=0.5
                , loss_function="CrossEntropyLoss"
                , positional_encoding=False):
        super(STATO, self).__init__(num_labels=num_labels
                                    , label_to_index=label_to_index
                                    , num_detailed_labels_per_level=num_detailed_labels_per_level
                                    , detailed_label_weights=detailed_label_weights
                                    , alpha=alpha
                                    , loss_function=loss_function
                                    , embedding_dim=embedding_dim)
        self.d_model = embedding_dim
        self.positional_encoding = positional_encoding
        
        # Here, we test positional encoding as an option
        if self.positional_encoding:
            self.layer_norm = nn.LayerNorm(embedding_dim) # layer normalisation
            self.positional_encoder = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        
        # We build the transformer encoder layer and the transformer encoder using the PyTorch implementation
        self.encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, nlayers)

        # Classifier for the main objective
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, src, labels=None, detailed_labels=None, src_key_padding_mask=None):
        
        if self.positional_encoding:
            # Normalise embeddings and add positional encoding
            src = self.layer_norm(src)
            src = self.positional_encoder(src)

        # Get transformer encoder output
        encoded_sentences = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        logits = self.classifier(encoded_sentences)
        outputs = (logits,)

        # Get detailed logits if detailed_label_classifiers are used (auxiliary objective)
        detailed_logits = []
        if self.num_detailed_labels_per_level is not None:
            for classifier in self.detailed_label_classifiers:
                detailed_logits.append(classifier(encoded_sentences))
            outputs = outputs + (detailed_logits,)

        # Calculate loss
        if labels is not None:
            loss = self.calculate_loss(logits, labels, detailed_logits, detailed_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss, logits, detailed_logits) or (logits,) if no detailed classifiers are used or detailed_labels are None

# adjusted based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    '''
    Implementing positional encoding as described in Vaswani et al. (2017)
    '''
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)