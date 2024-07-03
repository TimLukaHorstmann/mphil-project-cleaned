import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import AutoModel

# Token-level FRI discovery and classification model as described in Section 5.3 of the report
# Built based on PreTrainedModel class from Hugging Face (https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/model#transformers.PreTrainedModel)
# Based on implementation for RobertaForTokenClassification (https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/roberta/modeling_roberta.py#L1341)

class TokenClassificationWithDetailedLabels(PreTrainedModel):
    def __init__(self, model_name_or_path, num_labels, num_detailed_labels_per_level=None, detailed_label_weights=None, alpha=0.5):
        print(f"Loading pretrained model {model_name_or_path}:")
        model = AutoModel.from_pretrained(model_name_or_path, force_download=True, add_pooling_layer=False) # Load pretrained model from Hugging Face (force download), no pooling layer necessary
        config = model.config # extract config from pre-trained model
        
        # Initialisations
        super().__init__(config)
        self.num_labels = num_labels
        self.model = model
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)

        # Linear classifier for BIO(E) labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Linear classifiers for detailed labels
        # We initialise a classifier for each specified level of the ontology that we aim to predict
        # Information about the corresponding number of output classes for each level is provided in num_detailed_labels_per_level
        if num_detailed_labels_per_level:
            self.detailed_label_classifiers = nn.ModuleList()
            for num_detailed_labels in num_detailed_labels_per_level:
                self.detailed_label_classifiers.append(nn.Linear(config.hidden_size, num_detailed_labels))
        
        self.num_detailed_labels_per_level = num_detailed_labels_per_level
        
        self.alpha = alpha

        # Loss function (simple cross-entropy loss with ignore index for BIO labels (e.g. ignore padding))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialise detailed label weights and/or normalise them
        if num_detailed_labels_per_level:
            if detailed_label_weights is None:
                self.detailed_label_weights = [1.0 / len(num_detailed_labels_per_level)] * len(num_detailed_labels_per_level)
            else:
                total_weight = sum(detailed_label_weights)
                self.detailed_label_weights = [weight / total_weight for weight in detailed_label_weights]
                
        self.post_init() # all post_init method for proper initialisation

    def forward(self, input_ids, bio_labels=None, detailed_labels=None, attention_mask=None):
        
        # Obtain outputs from pre-trained base model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        
        # Get logits for main objective
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        
        # Calculate and collect logits for each set of detailed labels
        detailed_logits_list = []
        if self.num_detailed_labels_per_level: # only if auxiliary objective should be considered
            for detailed_classifier in self.detailed_label_classifiers:
                detailed_logits = detailed_classifier(sequence_output)
                detailed_logits_list.append(detailed_logits)
            outputs = outputs + (detailed_logits_list,)
        
        # Calculate loss based on above logits:
        loss = None
        if bio_labels is not None:
            
            # Calculate loss for BIO(E) labels
            loss = self.loss_fn(logits.view(-1, self.num_labels), bio_labels.view(-1))
            
            # Calculate loss for detailed labels if they are provided
            detailed_loss = 0.0
            if detailed_labels is not None:
                for i, detailed_logits in enumerate(detailed_logits_list):
                    detailed_label = detailed_labels[:, :, i].view(-1)
                    detailed_weight = self.detailed_label_weights[i]
                    # Only calculate loss where detailed labels are provided
                    valid_idx = detailed_label != -1  # no need to calculate loss for detailed labels that are -1 (i.e. outside of regions elements)
                    if valid_idx.any():
                        # Calculate loss for detailed labels as sum of weighted cross-entropy losses
                        detailed_loss += self.loss_fn(
                            detailed_logits.view(-1, self.num_detailed_labels_per_level[i])[valid_idx],
                            detailed_label[valid_idx]
                        ) * detailed_weight
            # Combine BIO and detailed label losses with weighting
            loss = self.alpha * loss + (1 - self.alpha) * detailed_loss
            outputs = (loss,) + outputs
        return outputs  # (loss, logits, detailed_logits) or (logits, detailed_logits)

