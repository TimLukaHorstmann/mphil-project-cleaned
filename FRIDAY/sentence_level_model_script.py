from datetime import datetime
import functools
import itertools
import os
import argparse

import sys
from multiprocessing import cpu_count
import pickle
import gc
from datasets import Dataset, DatasetDict, load_from_disk
import json

# Preprocessing
sys.path.append('../DataPreprocessing')
# sys.path.append('SnippetAnalytics')
from read_into_dicts import DocReader
from custom_sentence_tokenizer import Sentenizer
from detailed_labels_handler import extract_and_analyze_tags

# Model
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
import sentence_level_models as SLM
import evaluator
import optuna
from transformers import TrainerCallback

# WANDB
import wandb
import random
import string
import yaml


# This class is nearly identical to the token_level_model_script.py
# both classes could be merged into one, but for now, they are kept separate

# Custom compute_loss function for the trainer class for sentence-level models
# Implemented as an extension of the Hugging Face Trainer class (https://huggingface.co/docs/transformers/en/main_classes/trainer)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # extract labels for main and auxiliary objectives from inputs ("bio_labels" and "detailed_labels" respectively)
        labels = inputs.pop("labels", None)
        detailed_labels = inputs.pop("detailed_labels", None)
        
        # get outputs from model
        outputs = model(**inputs, labels=labels, detailed_labels=detailed_labels if detailed_labels is not None else None)
        loss = outputs[0] # extract loss as first component of outputs
        
        return (loss, outputs) if return_outputs else loss

# Custom callback for Optuna pruning
# Implemented as an extension of the Hugging Face TrainerCallback class (https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.TrainerCallback)
# Developed to allow for Optuna pruning during training while ensuring integration with Hugging Face Trainer class
class OptunaPruningCallback(TrainerCallback):

    def __init__(self, trial, monitor):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, **kwargs):
        # get best score for metric to be monitored
        objective = kwargs['metrics'][self.monitor]
        
        # report score to Optuna to let libray know if trial should be pruned
        self.trial.report(objective, step=state.epoch)
        if self.trial.should_prune():
            message = f"Trial was pruned at epoch {state.epoch}."
            raise optuna.exceptions.TrialPruned(message)

# Token-level model main class
# Corresponds to Section 5.2 in report
# This class can be used to train and optimise a sentence-level model for FRI discovery
class SentenceLevelModel():
    def __init__(self, config):
        
        # READ CONFIGURATIONS INTO VARIABLES
        self.config_file = config
        
        # random tag of 6 characters, used to identify the model (generated if not provided)
        self.unique_tag = config["general"]["UNIQUE_TAG"] if config["general"]["UNIQUE_TAG"] else ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        
        # GENERAL:
        self.DEBUG = config["general"]["DEBUG"]
        self.MODEL = config["general"]["MODEL"]
        self.DATA_SIZE = config["general"]["DATA_SIZE"]
        self.DOMAIN = config["general"]['DATA_DICT_FILE_PATH']['DOMAIN']
        self.DATA_DICT_FILE_PATH = config["general"]['DATA_DICT_FILE_PATH'][self.DOMAIN]
        self.DATA_PATH = config["general"]["DATA_PATH"]
        self.SPLITS_JSON = config["general"]["SPLITS_JSON"]
        self.TRAIN_SIZE = config["general"]["TRAIN_SIZE"]
        self.VAL_SIZE = config["general"]["VAL_SIZE"]
        self.USE_MULTIPROCESSING = config["general"]["USE_MULTIPROCESSING"]
        self.SPLIT_BASE = config["general"]["SPLIT_BASE"]
        self.SENTENCE_SPLITTER_MODEL = config["general"]["SENTENCE_SPLITTER_MODEL"]
        
        # HPO options
        self.IS_HPO_RUN = config["hpo"]["IS_HPO_RUN"]
        self.N_TRIALS = config["hpo"]["N_TRIALS"]
        self.STUDY_NAME = config["hpo"]["STUDY_NAME"]
        self.DB_FILE = config["hpo"]["DB_FILE"]
        self.HPO_DESC = config["hpo"]["HPO_DESC"]
        self.HIER_LABELS_LEVELS_SEARCH_SPACE = config["hpo"]["HIER_LABELS_LEVELS_SEARCH_SPACE"]
        self.LOSS_TO_OPTIMISE = config["hpo"]["LOSS_TO_OPTIMISE"]

        # SENTENCE LEVEL MODEL
        self.DATASET = config["sentence_level_model"]["DATASET"]
        self.PRETRAINED_MODEL = config["sentence_level_model"]["PRETRAINED_MODEL"]
        self.CHECKPOINT = config["sentence_level_model"]["CHECKPOINT"]
        self.EPOCHS = config["sentence_level_model"]["EPOCHS"]
        self.EARLY_STOPPING_PATIENCE = config["sentence_level_model"]["EARLY_STOPPING_PATIENCE"]
        self.BATCH_SIZE = config["sentence_level_model"]["BATCH_SIZE"]
        self.USE_POS_ENCODING = config["sentence_level_model"]["USE_POS_ENCODING"]
        self.ALPHA = config["sentence_level_model"]["ALPHA"]
        self.HIER_LABELS_LEVELS = config["sentence_level_model"]["HIER_LABELS_LEVELS"]  # also acts as flag if detailed models shall be used (i.e. model with auxiliary objective trained)
        self.DETAILED_LABEL_WEIGHTS = config["sentence_level_model"]["DETAILED_LABEL_WEIGHTS"]
        self.NUMBER_OF_LEVELS = len(self.HIER_LABELS_LEVELS)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.sentence_model = SentenceTransformer(config["sentence_level_model"]["SENTENCE_TRANSFORMER_MODEL"])   # test different models here?!
        self.EMBEDDING_DIMENSIONS = self.sentence_model.get_sentence_embedding_dimension()
        self.SENTENCE_MODEL_IS_BINARY = config["sentence_level_model"]["SENTENCE_MODEL_IS_BINARY"]
        self.SENTENCE_MODEL_ARCHITECTURE = config["sentence_level_model"]["SENTENCE_MODEL_ARCHITECTURE"]

        self.label_to_index = {"O": 0, "B": 1, "I": 2}
        self.index_to_label = {v: k for k, v in self.label_to_index.items()} # reverse of label_to_index
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sentenizer = Sentenizer(self.SENTENCE_SPLITTER_MODEL, f"{self.DATA_PATH}/Model/punkt_tokenizer.pkl")
        
        os.environ['WANDB_DIR'] = f"{self.DATA_PATH}/Model"
        os.environ['WANDB_CACHE_DIR'] = f"{self.DATA_PATH}/Model"
        

    def configure_wandb(self, config):
        '''
        Setup tracking run in wandb project
        '''
        
        # Define wandb init parameters
        wandb_init = {
            "project": config["wandb"]["WANDB_PROJECT"],
            "tags": [self.unique_tag, f"MODEL={self.MODEL}", f"DATA_SIZE={self.DATA_SIZE}"],
            "group": config["wandb"]["WANDB_GROUP"],
            "name": f'{config["wandb"]["WANDB_GROUP"]}-{str(self.MODEL).replace("/", "-")}-{self.HIER_LABELS_LEVELS}-{self.SENTENCE_SPLITTER_MODEL}'
        }

        if "RUN_ID" in config["wandb"]:
            wandb_init["id"] = config["wandb"]["RUN_ID"]
            wandb_init["resume"] = "allow"

        # Login and initalise
        wandb.login()
        run = wandb.init(**wandb_init)
        
        # Collect relevant information to save to wandb
        run_id = run.id
        config_dict = {
            "model": self.MODEL,
            "data_size": self.DATA_SIZE,
            "train_size": self.TRAIN_SIZE,
            "val_size": self.VAL_SIZE,
            "test_size": 1 - self.TRAIN_SIZE - self.VAL_SIZE,
            "epochs": self.EPOCHS,
            "batch_size": self.BATCH_SIZE,
            "hierarchical_labels_levels": self.HIER_LABELS_LEVELS,
            "run_id": run_id
        }
        wandb.config.update(config_dict, allow_val_change=True)
        
    def get_data_dict(self):
        '''
        Load data_dict file or create new one using the DocReader class
        (i.e. pull raw data)
        '''
        docReader = DocReader(self.MODEL, self.tokenizer)
        create_original_snippets = True
        add_full_page = False

        # Load data_dict if available, otherwise create new one
        if self.DATA_DICT_FILE_PATH:
            with open(f'{self.DATA_PATH}/{self.DATA_DICT_FILE_PATH}', 'rb') as handle:
                data_dict = pickle.load(handle)
        else:
            data_dict = docReader.preprocess_folder(preprocess=False, folder_path=f'{self.DATA_PATH}/28245V231219', num_workers=min(70, cpu_count()), data_size=self.DATA_SIZE, chunksize=1
                                                , extract_title=True, extract_doc_long_id=True, refine_regions=True, create_original_snippets=create_original_snippets, add_full_page=add_full_page
                                                , data_dict_folder=f"{self.DATA_PATH}/preprocessing/data_dicts", file_name_additional_suffix="-22-02")
        
        # Save data_dict to single pickle file if newly created
        if not self.DATA_DICT_FILE_PATH:
            with open(f'{self.DATA_PATH}/preprocessing/data_dicts/data_dict_{self.MODEL}-19-03.pkl', 'wb') as handle:
                pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        return data_dict
    
    def get_detailed_labels_info(self):
        '''
        Collect detailed labels information from data_dict (relevant for model architecture) and setup label numbers and mapping dict.
        '''
        # Extract and analyze tags

        self.tag_statistics, self.mapping_dicts = extract_and_analyze_tags(data_dict, self.HIER_LABELS_LEVELS)

        # Calculate unique tag counts
        self.unique_level_tags = {level: len(tags) for level, tags in self.tag_statistics.items()}
        self.total_unique_tags = sum(len(tags) for tags in self.tag_statistics.values())

        # Output detailed labels info
        if not self.HIER_LABELS_LEVELS:
            print("Note: no HIER_LABELS_LEVELS provided.")
        # Print statistics and unique counts
        for level, tags in self.tag_statistics.items():
            print(f"{level.capitalize()} Tags:", tags)
            print(f"Unique {level.capitalize()} Tags:", self.unique_level_tags[level])

        # Print full tag count if available
        if 'full_tags' in self.tag_statistics:
            print("Full Tags Count:", self.tag_statistics['full_tags'])
        print("Total Unique Tags:", self.total_unique_tags)

        # Print mapping dictionaries (relevant for encoding/decoding operations!)
        for level, mapping_dict in self.mapping_dicts.items():
            print(f"{level.capitalize()} Mapping Dictionary:", mapping_dict)
            
    def load_dataset(self):
        '''
        Load dataset for model from disk.
        '''
        print(f"Load Datadict from {self.DATA_PATH}/{self.DATASET}.")
        dataset_dict = load_from_disk(f"{self.DATA_PATH}/{self.DATASET}")
        print("Datadict successfully loaded.")
        
        return dataset_dict
    
    def assert_dataset_correctness(self, dataset_dict):
        '''
        Check if dataset is correctly split into train, validation and test sets according to the centralised JSON splits file.
        '''
        # Load the JSON splits
        with open(self.SPLITS_JSON, 'r') as file:
            document_splits = json.load(file)

        # Helper to extract document IDs from dataset samples
        def extract_doc_ids_from_dataset(dataset):
            return {sample['metadata']['doc_id'] for sample in dataset}

        # Get document IDs from each dataset in the dataset_dict
        print("Extract document IDs from train.")
        train_doc_ids = extract_doc_ids_from_dataset(dataset_dict["train"])
        print("Extract document IDs from validation.")
        val_doc_ids = extract_doc_ids_from_dataset(dataset_dict["validation"])
        print("Extract document IDs from test.")
        test_doc_ids = extract_doc_ids_from_dataset(dataset_dict["test"])

        # Get document IDs from JSON splits
        json_train_ids = set(document_splits['train_ids'])
        json_val_ids = set(document_splits['val_ids'])
        json_test_ids = set(document_splits['test_ids'])

        # Compare and assert
        assert train_doc_ids == json_train_ids, "Mismatch in train dataset document IDs"
        assert val_doc_ids == json_val_ids, "Mismatch in validation dataset document IDs"
        assert test_doc_ids == json_test_ids, "Mismatch in test dataset document IDs"

        print("Validation successful: Datasets contain the correct document IDs according to the JSON splits.")
         
    def custom_collate_fn(self, batch):
        '''
        Custom collate function for token-level model.
        Ensure proper/uniform padding for all elements in the batch.
        '''
        # Get embeddings and pad them
        embeddings = [torch.tensor(sample['embeddings'], dtype=torch.float) for sample in batch]
        embeddings_padded = pad_sequence([pad_sequence(e, batch_first=True, padding_value=0.0) for e in embeddings], batch_first=True)

        # Function to merge B and I labels (for binary classification)
        def merge_bi_labels(label_tensor):
            label_tensor[label_tensor == self.label_to_index["I"]] = self.label_to_index["B"]
            return label_tensor
        
        # If we want a binary model, merge B and I labels
        # Otherwise, keep BIO
        # Pad labels
        if self.SENTENCE_MODEL_IS_BINARY:
            labels = [merge_bi_labels(torch.tensor(sample['bio_labels'], dtype=torch.long)) for sample in batch]
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1) 
        else: 
            labels = [sample['bio_labels'] for sample in batch]
            labels_padded = pad_sequence([torch.tensor(l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-1) 

        # Setup batch dictionary
        batch_dict = {"src": embeddings_padded, "labels": labels_padded}

        # Checl for detailed labels as part of the auxiliary objective
        if self.HIER_LABELS_LEVELS:
            detailed_labels_padded = []
            
            NUMBER_OF_LEVELS = len(self.HIER_LABELS_LEVELS)
            
            for sample in batch:
                page_detailed_labels = sample.get('detailed_labels', [])

                filtered_page_detailed_labels = []
                for sentence_labels in page_detailed_labels:
                    # Filter sentence's labels based on HIER_LABELS_LEVELS
                    filtered_sentence_labels = [sentence_labels[idx] for idx in self.HIER_LABELS_LEVELS if idx < len(sentence_labels)]
                    filtered_page_detailed_labels.append(filtered_sentence_labels)
                
                # Pad each sentence's detailed labels to ensure uniform length within a page
                sentence_detailed_labels_padded = [sentence + [-1] * (NUMBER_OF_LEVELS - len(sentence)) for sentence in filtered_page_detailed_labels]

                if sentence_detailed_labels_padded:
                    detailed_labels_tensor = torch.tensor(sentence_detailed_labels_padded, dtype=torch.long)
                else:
                    # Placeholder tensor if there are no detailed labels
                    detailed_labels_tensor = torch.full((1, NUMBER_OF_LEVELS), -1, dtype=torch.long)
                
                detailed_labels_padded.append(detailed_labels_tensor)
            
            # Pad pages to ensure uniformity across the batch
            detailed_labels_padded_uniform = pad_sequence(detailed_labels_padded, batch_first=True, padding_value=-1)
            
            # Add to batch dict
            batch_dict["detailed_labels"] = detailed_labels_padded_uniform
        
        return batch_dict
        
    def setup_model(self):
        '''
        Choosing model architecture for sentence-level model.
        Options are BiLSTM, BiLSTM_CRF, and Transformer.
        '''
        if self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"]:
            self.LSTM_HIDDEN_DIM = 512
            self.LSTM_NLAYERS = 6
            self.LSTM_DROPOUT = 0.01
            model = SLM.SentenceTaggingBiLSTM(
                embedding_dim=self.EMBEDDING_DIMENSIONS,
                hidden_dim=self.LSTM_HIDDEN_DIM,
                nlayers=self.LSTM_NLAYERS,
                bidirectional=True,
                dropout=self.LSTM_DROPOUT,
                use_crf= self.SENTENCE_MODEL_ARCHITECTURE == "BiLSTM_CRF", # use conditional random fields or linear layers
                num_labels= 2 if self.SENTENCE_MODEL_IS_BINARY else len(self.label_to_index),  # For BIO tagging or binary task
                label_to_index=self.label_to_index,
                num_detailed_labels_per_level=[self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,  # Total unique labels across hierarchy
                detailed_label_weights = self.DETAILED_LABEL_WEIGHTS,
                alpha=self.ALPHA, # weight to give normal loss (--> detailed loss is 1- alpha), only applied for detailed labels
                loss_function="CrossEntropyLoss"
            ).to(self.device)
            param_dict = {
                "hidden_dim": self.LSTM_HIDDEN_DIM
                , "nalyers": self.LSTM_NLAYERS
                , "dropout": self.LSTM_DROPOUT
                , "embedding_dimension": self.EMBEDDING_DIMENSIONS
            }
        elif self.SENTENCE_MODEL_ARCHITECTURE == "Transformer":
            self.TRANSFORMER_NHEAD = 4 # default: 8
            self.TRANSFORMER_HIDDEN_DIM = 2048 # default: 1024
            self.TRANSFORMER_NLAYERS = 8 # default: 8
            self.TRANSFORMER_DROPOUT = 0.3069012560158778 # default: 0.1
            model = SLM.STATO(
                embedding_dim=self.EMBEDDING_DIMENSIONS,
                nhead=self.TRANSFORMER_NHEAD, #8
                hidden_dim=self.TRANSFORMER_HIDDEN_DIM, #2048
                nlayers=self.TRANSFORMER_NLAYERS,
                dropout=self.TRANSFORMER_DROPOUT,
                num_labels= 2 if self.SENTENCE_MODEL_IS_BINARY else len(self.label_to_index),  # For BIO tagging or binary task
                label_to_index=self.label_to_index,
                num_detailed_labels_per_level=[self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,  # Total unique labels across hierarchy
                detailed_label_weights = self.DETAILED_LABEL_WEIGHTS,
                alpha=self.ALPHA, # weight to give normal loss (--> detailed loss is 1- alpha), only applied for detailed labels
                loss_function="CrossEntropyLoss",
                positional_encoding=self.USE_POS_ENCODING
            ).to(self.device)
            param_dict = {
                "nhead": self.TRANSFORMER_NHEAD
                , "hidden_dim": self.TRANSFORMER_HIDDEN_DIM
                , "nalyers": self.TRANSFORMER_NLAYERS
                , "dropout": self.TRANSFORMER_DROPOUT
                , "embedding_dimension": self.EMBEDDING_DIMENSIONS
            }
        else:
            print("No model chosen!")
            
        print(model.device)
        print(next(model.parameters()).device)
        wandb.config.update(param_dict, allow_val_change=True)
        return model
    
    def get_training_arguments(self, trial=None):
        '''
        Define training arguments for Hugging Face Trainer class
        '''
        
        training_args = TrainingArguments(
            output_dir=f'{self.DATA_PATH}/Model/results/{self.unique_tag}',
            num_train_epochs= self.EPOCHS,
            per_device_train_batch_size=self.BATCH_SIZE, 
            per_device_eval_batch_size=self.BATCH_SIZE,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f'{self.DATA_PATH}/Model/logs',
            logging_steps=50,
            label_names=["labels", "detailed_labels"] if self.HIER_LABELS_LEVELS else ["labels"],
            evaluation_strategy="epoch", 
            save_strategy="epoch",
            save_steps=100,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            greater_is_better=False,
            report_to="wandb",
            # fp16=True,
            remove_unused_columns=False
        )
        
        return training_args
    
    def start_training(self, dataset_dict, model):
        '''
        Train the sentence-level model using the Hugging Face Trainer class
        '''
        training_args = self.get_training_arguments()
        
        # Check if a saved checkpoint exists
        latest_checkpoint = None
        if self.CHECKPOINT:
            checkpoint_directory = f'{self.DATA_PATH}/{self.CHECKPOINT}'
            if os.path.exists(checkpoint_directory):
                latest_checkpoint = checkpoint_directory

        # Bind collate function
        bound_collate_fn = functools.partial(self.custom_collate_fn)

        # Create Trainer instance
        trainer = CustomTrainer(
            model=model, 
            args=training_args,
            compute_metrics=evaluator.compute_metrics_wrapper(self.index_to_label, self.HIER_LABELS_LEVELS
                                                              , self.mapping_dicts
                                                              , pull_extra_metric="model_pk_value_segeval"),
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            data_collator=bound_collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.EARLY_STOPPING_PATIENCE)]
        )

        print(f"Start Training of Model: {self.unique_tag}")
        print(f"------ METADATA ------")
        print(f"{self.SENTENCE_MODEL_ARCHITECTURE=}")
        print(f"{self.SENTENCE_MODEL_IS_BINARY=}")
        print(f"{self.ALPHA=}")
        print(f"{self.HIER_LABELS_LEVELS=}")
        print(f"{self.DETAILED_LABEL_WEIGHTS=}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)

        # Save model
        model_save_path = f"{self.DATA_PATH}/Model/results/saved_model/{self.unique_tag}/torch_model.pth" 
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)

        # SAVE CONFIG
        model_config = {
            "model_type": self.SENTENCE_MODEL_ARCHITECTURE,
            "embedding_dim": self.EMBEDDING_DIMENSIONS,
            "hidden_dim": self.LSTM_HIDDEN_DIM if self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"] else self.TRANSFORMER_HIDDEN_DIM,
            "nlayers": self.LSTM_NLAYERS if self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"] else self.TRANSFORMER_NLAYERS,
            "bidirectional": self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"],
            "dropout": self.LSTM_DROPOUT if self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"] else self.TRANSFORMER_DROPOUT,
            "num_labels": 2 if self.SENTENCE_MODEL_IS_BINARY else len(self.label_to_index),
            "label_to_index": self.label_to_index,
            "num_detailed_labels_per_level": [self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,
            "detailed_label_weights": self.DETAILED_LABEL_WEIGHTS,
            "alpha": self.ALPHA,
            "loss_function": "CrossEntropyLoss",
            "nhead": self.TRANSFORMER_NHEAD if self.SENTENCE_MODEL_ARCHITECTURE == "Transformer" else None
        }
        config_save_path = os.path.join(os.path.dirname(model_save_path), "config.json")
        with open(config_save_path, 'w') as config_file:
            json.dump(model_config, config_file, indent=4)
            
        return trainer
        
    def evaluate_model(self, dataset):
        '''
        Evaluate the model on a dataset using the Hugging Face Trainer class
        '''
        test_results = trainer.evaluate(dataset)
        print("Test Set Evaluation Results:")
        for key, value in test_results.items():
            print(f"{key}: {value}")
            
            
            
    ######## HPO FUNCTIONS ########
    def generate_level_combinations(self):
        '''
        Helper function used to generate all possible combinations of hierarchical levels for HPO
        (e.g. if levels are [1,2] then the combinations are [], [1], [2], [1,2])
        '''
        levels = self.HIER_LABELS_LEVELS_SEARCH_SPACE
        combinations = []
        for i in range(len(levels) + 1):
            for combination in itertools.combinations(levels, i):
                combinations.append(list(combination))
        return combinations
    
    def start_HPO(self, dataset_dict, study_name=None, db_file=None):
        """
        Start a HPO study using Optuna.
        """
        if study_name is None:
            study_name = f'study-SENTENCE-{self.MODEL}-{"binary" if self.SENTENCE_MODEL_IS_BINARY else "non"}-{self.SENTENCE_MODEL_ARCHITECTURE}'  # Unique identifier of the study
            study_name = study_name.replace("/", "-")
            
        def objective(trial: optuna.trial.Trial):
            model=None
            torch.cuda.empty_cache()
            gc.collect()
                        
            # Choose levels and weights for integration of auxiliary objective
            # Define hyperparameters to be optimised
            hier_level_options = self.generate_level_combinations()
            hier_level_options[0] = None  # Handle empty set appropriately

            self.HIER_LABELS_LEVELS = trial.suggest_categorical("HIER_LABELS_LEVELS", hier_level_options)

            # Suggest weights based on the number of levels chosen
            if self.HIER_LABELS_LEVELS is None:
                self.DETAILED_LABEL_WEIGHTS = None
            else:
                self.DETAILED_LABEL_WEIGHTS  = []
                for _ in range(len(self.HIER_LABELS_LEVELS)):
                    self.DETAILED_LABEL_WEIGHTS.append(trial.suggest_float(f"weight_level_{_}", 0.1, 1.0))
            
            # update tag statistics with levels above:
            self.get_detailed_labels_info()
            
            # Parameters are defined based on the model architecture
            if self.SENTENCE_MODEL_ARCHITECTURE == "Transformer":
                nhead = trial.suggest_categorical("nhead", [2, 4, 8, 16])
            
            if self.SENTENCE_MODEL_ARCHITECTURE == "Transformer":
                hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024, 2048, 4096])
            elif self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"]:
                hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
            if self.SENTENCE_MODEL_ARCHITECTURE == "Transformer":
                nlayers = trial.suggest_int("nlayers", 1, 24)
            elif self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"]:
                nlayers = trial.suggest_int("nlayers", 1, 8)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            
            # If no detailed labels are included (i.e. no auxiliary objective), set alpha to 1
            if self.HIER_LABELS_LEVELS is None:
                self.ALPHA = 1
            else: 
                self.ALPHA = trial.suggest_float("alpha", 0, 1)
            
            trial_id = trial.number
            
            # Initialise wandb with HPO-specific tags or groups
            wandb.init(
                project=self.config_file["wandb"]["WANDB_PROJECT"],
                group=f"optuna_study_{study_name}",
                name=f"trial_{trial_id}",
                tags=["optuna", "hpo", f"{study_name}", f"{self.HPO_DESC}"],
                config=trial.params, 
            )
            
            # save config to wandb
            for key, value in self.config_file.items():
                wandb.config.update({key: value})
            
            # Get Model
            model = None
            if self.SENTENCE_MODEL_ARCHITECTURE in ["BiLSTM", "BiLSTM_CRF"]:
                model = SLM.SentenceTaggingBiLSTM(
                    embedding_dim=self.EMBEDDING_DIMENSIONS,
                    hidden_dim=hidden_dim,
                    nlayers=nlayers, 
                    bidirectional=True,
                    dropout=dropout, 
                    use_crf= self.SENTENCE_MODEL_ARCHITECTURE == "BiLSTM_CRF", 
                    num_labels= 2 if self.SENTENCE_MODEL_IS_BINARY else len(self.label_to_index), 
                    label_to_index=self.label_to_index,
                    num_detailed_labels_per_level=[self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,  # Total unique labels across hierarchy
                    detailed_label_weights = self.DETAILED_LABEL_WEIGHTS,
                    alpha=self.ALPHA, # weight to give normal loss (--> detailed loss is 1- alpha), only applied for detailed labels
                    loss_function="CrossEntropyLoss"
                ).to(self.device)
            elif self.SENTENCE_MODEL_ARCHITECTURE == "Transformer":
                model = SLM.STATO(
                    embedding_dim=self.EMBEDDING_DIMENSIONS,
                    nhead=nhead,
                    hidden_dim=hidden_dim,
                    nlayers=nlayers,
                    dropout=dropout,
                    num_labels= 2 if self.SENTENCE_MODEL_IS_BINARY else len(self.label_to_index),  # For BIO tagging or binary task
                    label_to_index=self.label_to_index,
                    num_detailed_labels_per_level=[self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,  # Total unique labels across hierarchy
                    detailed_label_weights = self.DETAILED_LABEL_WEIGHTS,
                    alpha=self.ALPHA, # weight to give normal loss (--> detailed loss is 1- alpha), only applied for detailed labels
                    loss_function="CrossEntropyLoss",
                    positional_encoding=self.USE_POS_ENCODING
                ).to(self.device)
            else:
                print("No model chosen!")

            # get training arguments and bind collate function
            training_args = self.get_training_arguments()
            bound_collate_fn = functools.partial(self.custom_collate_fn)
            
            # Define Trainer based on abive
            trainer = CustomTrainer(
                model=model, 
                args=training_args, 
                compute_metrics=evaluator.compute_metrics_wrapper(self.index_to_label, self.HIER_LABELS_LEVELS
                                                                  , self.mapping_dicts
                                                                  , pull_extra_metric="f1"),
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                data_collator=bound_collate_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.EARLY_STOPPING_PATIENCE), OptunaPruningCallback(trial, monitor="eval_f1")]
            )

            # Start training for this trial
            print("Start training.")
            print("Parameters to be tested:")
            print(f"{self.HIER_LABELS_LEVELS=}")
            print(f"{self.DETAILED_LABEL_WEIGHTS=}")
            print(f"{self.ALPHA=}")
            if self.SENTENCE_MODEL_ARCHITECTURE == "Transformer":
                print(f"{nhead=}")
            print(f"{hidden_dim=}")
            print(f"{nlayers=}")
            print(f"{dropout=}")
            trainer.train()

            # Evaluate the model
            print("Start evaluation.")
            eval_result = trainer.evaluate()

            # Choose loss according to config and report back
            if self.LOSS_TO_OPTIMISE == "total_loss":
                eval_loss = eval_result["eval_loss"]
            elif self.LOSS_TO_OPTIMISE == "bio_loss":
                eval_loss = eval_result["eval_normal_labels_metrics"]["cross_entropy_loss"]
            elif self.LOSS_TO_OPTIMISE == "model_pk_value_segeval":
                eval_loss = eval_result["eval_normal_labels_metrics"]["model_pk_value_segeval"]
            else:
                eval_loss = eval_result["eval_normal_labels_metrics"]["cross_entropy_loss"] # default = bio loss, as this is the main objective of the work
            
            print("Send to wandb.")
            wandb.finish()

            return eval_loss
        
        # Create db file to store HPO study to
        if db_file is None:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            db_file = f"{study_name}_{current_datetime}_{self.HPO_DESC}.db"
        
        db_file_path = f"sqlite:///{self.DATA_PATH}/optuna_studies/{db_file}"
        
        # Setup and run the study
        study = optuna.create_study(study_name=study_name
                                    , pruner=optuna.pruners.MedianPruner()
                                    , storage=db_file_path
                                    , direction="minimize"
                                    , load_if_exists=True)
        study.optimize(objective, n_trials=self.N_TRIALS)

    
        # Log best trial information to wandb
        wandb.init(
        project=config["wandb"]["WANDB_PROJECT"],
        group=f"optuna_study_{study_name}",
        name="best_trial_results"
        )
        wandb.log({
            "best_trial_loss": study.best_trial.value,
            "best_params": study.best_trial.params
        })
        wandb.finish()

        # Print best trial information
        print("Best trial:")
        print(f"Value: {study.best_trial.value}")
        print("Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

# Entry point for the script
if __name__ == "__main__":
    
    # Get command line arguments: configuration file
    parser = argparse.ArgumentParser(description="Run Sentence Level Model Training/HPO")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
        
    # Load configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Initialise sentence-level model using configuration
    sentence_level_model = SentenceLevelModel(config=config)
    
    print(f"################ GET DATA_DICT ################", "\n")
    data_dict = sentence_level_model.get_data_dict()
    
    print(f"################ GET DETAILED LABELS INFO ################", "\n")
    sentence_level_model.get_detailed_labels_info()
    
    print(f"################ LOAD DATASET ################", "\n")
    dataset_dict = sentence_level_model.load_dataset()
    
    # Optional: assert correctness of dataset
    # sentence_level_model.assert_dataset_correctness(dataset_dict)
    
    # Normal training
    if not sentence_level_model.IS_HPO_RUN:
        print(f"################ INITIATE TRAINING ################", "\n")
        sentence_level_model.configure_wandb(config)
        model = sentence_level_model.setup_model()
        trainer = sentence_level_model.start_training(dataset_dict, model)
        sentence_level_model.evaluate_model(dataset_dict["test"])
        wandb.finish()
    # HPO run
    else:
        print(f"################ INITIATE HPO ################", "\n")
        sentence_level_model.start_HPO(dataset_dict, sentence_level_model.STUDY_NAME, sentence_level_model.DB_FILE)
    
    
    
    
    
    

    