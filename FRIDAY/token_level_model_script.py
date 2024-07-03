import functools
import itertools
import os
import argparse
import sys
from datetime import datetime
from multiprocessing import cpu_count
import pickle
import json

# Preprocessing
sys.path.append('../DataPreprocessing')
from read_into_dicts import DocReader
from detailed_labels_handler import extract_and_analyze_tags

# Model
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer, EarlyStoppingCallback, AutoModelForTokenClassification, TrainerCallback
from datasets import DatasetDict
from token_level_model import TokenClassificationWithDetailedLabels

# Evaluation:
import evaluator
import optuna

# WANDB
import wandb
import random
import string
import yaml

# Custom compute_loss function for the trainer class for token-level models
# Implemented as an extension of the Hugging Face Trainer class (https://huggingface.co/docs/transformers/en/main_classes/trainer)
class CustomTokenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # extract labels for main and auxiliary objectives from inputs ("bio_labels" and "detailed_labels" respectively)
        bio_labels = inputs.pop("bio_labels", None)
        detailed_labels = inputs.pop("detailed_labels", None)
        
        # get outputs from model
        outputs = model(**inputs, bio_labels=bio_labels, detailed_labels=detailed_labels)
        loss = outputs[0]  # extract loss as first component of outputs
        
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
# Corresponds to Section 5.3 in report
# This class can be used to train and optimise a token-level model for FRI discovery
class TokenLevelModel():
    
    def __init__(self, config):
        '''
        Initialise model class with values from configuration file
        '''
        
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
        
        # HPO options
        self.IS_HPO_RUN = config["hpo"]["IS_HPO_RUN"]
        self.N_TRIALS = config["hpo"]["N_TRIALS"]
        self.STUDY_NAME = config["hpo"]["STUDY_NAME"]
        self.DB_FILE = config["hpo"]["DB_FILE"]
        self.HPO_DESC = config["hpo"]["HPO_DESC"]
        self.HIER_LABELS_LEVELS_SEARCH_SPACE = config["hpo"]["HIER_LABELS_LEVELS_SEARCH_SPACE"]
        self.LOSS_TO_OPTIMISE = config["hpo"]["LOSS_TO_OPTIMISE"]

        # TOKEN LEVEL MODEL
        self.DEFAULT_MODEL = config["token_level_model"]["DEFAULT_MODEL"]
        self.RETOKENIZATION_NEEDED = config["token_level_model"]["RETOKENIZATION_NEEDED"]
        self.DATASET = config["token_level_model"]["DATASET"]
        self.PRETRAINED_MODEL = config["token_level_model"]['PRETRAINED_MODEL']
        self.CHECKPOINT = config["token_level_model"]["CHECKPOINT"]
        self.EPOCHS = config["token_level_model"]["EPOCHS"]
        self.EARLY_STOPPING_PATIENCE = config["token_level_model"]["EARLY_STOPPING_PATIENCE"]
        self.BATCH_SIZE = config["token_level_model"]["BATCH_SIZE"]
        self.OVERLAP = config["token_level_model"]["SLIDING_WINDOW_OVERLAP"]
        self.ALPHA = config["token_level_model"]["ALPHA"]
        self.HIER_LABELS_LEVELS = config["token_level_model"]["HIER_LABELS_LEVELS"]  # also acts as flag if detailed models shall be used (i.e. model with auxiliary objective trained)
        self.DETAILED_LABEL_WEIGHTS = config["token_level_model"]["DETAILED_LABEL_WEIGHTS"]
        self.NUMBER_OF_LEVELS = len(self.HIER_LABELS_LEVELS) if self.HIER_LABELS_LEVELS else None

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.label_to_index = {'O': 0, 'B': 1, 'I': 2, 'E': 3, '-100': -100}
        self.index_to_label = {v: k for k, v in self.label_to_index.items()} # reverse of label_to_index

        self.NUMBER_BIO_LABELS = len(self.label_to_index) -1 # -1 for -100
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            "name": f'{config["wandb"]["WANDB_GROUP"]}-{str(self.MODEL).replace("/", "-")}-{self.HIER_LABELS_LEVELS}'
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
            "sliding_window_overlap": self.OVERLAP,
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
        # Print statistsics and unique counts
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
        dataset_dict = DatasetDict.load_from_disk(f"{self.DATA_PATH}/{self.DATASET}") 
        print("Datadict successfully loaded.")
        return dataset_dict

    def assert_dataset_correctness(self, dataset_dict):
        '''
        Check if dataset is correctly split into train, validation and test sets according to the centralised JSON splits file.
        '''
        # Load JSON splits
        with open(self.SPLITS_JSON, 'r') as file:
            document_splits = json.load(file)

        # Helper to extract document IDs from dataset samples
        def extract_doc_ids_from_dataset(dataset):
            return {sample['metadata']['doc_id'] for sample in dataset}

        # Get document IDs from each dataset in the dataset_dict
        print("Extract document IDs from train.")
        train_doc_ids = extract_doc_ids_from_dataset(dataset_dict['train'])
        print("Extract document IDs from validation.")
        val_doc_ids = extract_doc_ids_from_dataset(dataset_dict['validation'])
        print("Extract document IDs from test.")
        test_doc_ids = extract_doc_ids_from_dataset(dataset_dict['test'])

        # Get document IDs from JSON splits
        json_train_ids = set(document_splits['train_ids'])
        json_val_ids = set(document_splits['val_ids'])
        json_test_ids = set(document_splits['test_ids'])

        # Compare and assert
        assert train_doc_ids == json_train_ids, "Mismatch in train dataset document IDs"
        assert val_doc_ids == json_val_ids, "Mismatch in validation dataset document IDs"
        assert test_doc_ids == json_test_ids, "Mismatch in test dataset document IDs"

        print("Validation successful: Datasets contain the correct document IDs according to the JSON splits.")
        
    def token_classification_collate_fn(self, batch):
        '''
        Custom collate function for token-level model.
        Ensure proper/uniform padding for all elements in the batch.
        '''
        
        # MAIN OBJECTIVE
        # pad input_ids, attention_mask and bio_labels (i.e. main objective labels)
        input_ids = pad_sequence([torch.tensor(sample['input_ids'], dtype=torch.long) for sample in batch], batch_first=True, padding_value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token if hasattr(self.tokenizer, 'pad_token') else '[PAD]')) # if possible, we ensure that the padding value is the same as the one used by the tokenizer
        attention_mask = pad_sequence([torch.tensor(sample['attention_mask'], dtype=torch.long) for sample in batch], batch_first=True, padding_value=0) # attention mask consists of 0s and 1s and is thefore padded with 0s
        bio_labels_padded = pad_sequence([torch.tensor(sample['bio_labels'], dtype=torch.long) for sample in batch], batch_first=True, padding_value=-100)  # -100 is used as the padding value for the main objective (BIO(E) labels)

        # put batch elements together in a dictionary
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "bio_labels": bio_labels_padded}
        
        # AUXILIARY OBJECTIVE
        # pad detailed_labels if available
        if self.HIER_LABELS_LEVELS:
            batch_detailed_labels = []
            NUMBER_OF_LEVELS = len(self.HIER_LABELS_LEVELS)
            
            for sample in batch:
                detailed_labels = sample.get('detailed_labels', [])

                # filtered detailed labels for page
                filtered_token_labels_list = []

                for token_labels in detailed_labels:
                    # Filter the token's labels based on HIER_LABELS_LEVELS
                    # This step is necessary as the dataset contains all levels of labels, but the model only needs the specified levels
                    filtered_token_labels = [token_labels[idx] for idx in self.HIER_LABELS_LEVELS if idx < len(token_labels)]
                    filtered_token_labels_list.append(filtered_token_labels)
                
                # Pad detailed labels to ensure uniform length
                token_detailed_labels_padded = [labels + [-1] * (NUMBER_OF_LEVELS - len(labels)) for labels in filtered_token_labels_list]

                # Convert to tensor
                if token_detailed_labels_padded:
                    detailed_labels_tensor = torch.tensor(token_detailed_labels_padded, dtype=torch.long) 
                else:
                    # Create placeholder tensor if there are no detailed labels
                    detailed_labels_tensor = torch.full((1, NUMBER_OF_LEVELS), -1, dtype=torch.long)
                
                batch_detailed_labels.append(detailed_labels_tensor)
            
            detailed_labels_padded_uniform = pad_sequence(batch_detailed_labels, batch_first=True, padding_value=-1) # using -1 as padding value for detailed labels for consistency

            # Add to bacth dict
            batch_dict["detailed_labels"] = detailed_labels_padded_uniform

        return batch_dict

    def setup_model(self):
        '''
        Choosing model archietcture for token-level model (either default model from HuggingFace (i.e. without custom classification head) or custom model with detailed labels (i.e. with custom classification head)
        '''
        if self.DEFAULT_MODEL:
            token_model = AutoModelForTokenClassification.from_pretrained(
            self.MODEL,
            num_labels=self.NUMBER_BIO_LABELS
            )
        else:
            token_model = TokenClassificationWithDetailedLabels(
                model_name_or_path=self.MODEL,
                num_labels=self.NUMBER_BIO_LABELS,
                num_detailed_labels_per_level=[self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,
                alpha=self.ALPHA
            )
        
        token_model.to(self.device)
        print(token_model.device)
        print(next(token_model.parameters()).device)
        
        return token_model
    
    def get_training_arguments(self, trial=None):
        '''
        Define training arguments for Hugging Face Trainer class
        '''
        
        training_args = TrainingArguments(
            output_dir=f'{self.DATA_PATH}/Model/results/{self.unique_tag}',
            num_train_epochs=self.EPOCHS,
            per_device_train_batch_size=self.BATCH_SIZE,
            per_device_eval_batch_size=self.BATCH_SIZE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{self.DATA_PATH}/Model/logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_steps=500,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model='model_pk_value_segeval', #metric to determine the best model ("loss") (model can be optimised towards bio_loss, pK metric etc.)
            greater_is_better=False,
            report_to="wandb",
            fp16=True,
            label_names= None if self.DEFAULT_MODEL else (["bio_labels", "detailed_labels"] if self.HIER_LABELS_LEVELS else ["bio_labels"]), # necessary for custom model to know which labels to use
            remove_unused_columns=self.DEFAULT_MODEL # necessary to ensure that both objectives are correctly passed through the HF classes
        )
        
        return training_args
    
    def start_training(self, dataset_dict, model):
        '''
        Train the token-level model using the Hugging Face Trainer class
        '''
        
        # Rename 'bio_labels' to 'labels' for consistency with default implementation in Hugging Face
        if self.DEFAULT_MODEL:
            def rename_column(sample):
                sample['labels'] = sample.pop('bio_labels')
                return sample
            dataset_dict = dataset_dict.map(rename_column)

        # Get training arguments
        training_args = self.get_training_arguments()
        
        # Check if a saved checkpoint exists
        latest_checkpoint = None
        if self.CHECKPOINT:
            checkpoint_directory = f'{self.DATA_PATH}/{self.CHECKPOINT}'
            if os.path.exists(checkpoint_directory):
                latest_checkpoint = checkpoint_directory

        # Bind the custom collate function for use in trainer
        bound_collate_fn = functools.partial(self.token_classification_collate_fn)

        # Create Trainer instance (either default or custom)
        # early stopping is introduced according to value specified in config
        if self.DEFAULT_MODEL:
            trainer = Trainer(
                model=model, 
                args=training_args,
                train_dataset=dataset_dict['train'],
                eval_dataset=dataset_dict['validation'],
                compute_metrics=evaluator.compute_metrics_wrapper(self.index_to_label, self.HIER_LABELS_LEVELS
                                                                  , self.mapping_dicts, calc_windowdiff=False
                                                                  , pull_extra_metric="model_pk_value_segeval"),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.EARLY_STOPPING_PATIENCE)]
            )
        else:
            trainer = CustomTokenTrainer(
                model=model, 
                args=training_args,
                train_dataset=dataset_dict['train'],
                eval_dataset=dataset_dict['validation'],
                data_collator=bound_collate_fn,
                compute_metrics=evaluator.compute_metrics_wrapper(self.index_to_label, self.HIER_LABELS_LEVELS
                                                                  , self.mapping_dicts, calc_windowdiff=False
                                                                  , pull_extra_metric="model_pk_value_segeval"),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.EARLY_STOPPING_PATIENCE)]
            )

        print(f"Start Training of Model: {self.unique_tag}")
        if latest_checkpoint:
            print(f"Resuming from checkpoint {latest_checkpoint}")
        print(f"------ METADATA ------")
        print(f"{self.MODEL=}")
        print(f"{self.ALPHA=}")
        print(f"{self.HIER_LABELS_LEVELS=}")
        print(f"{self.DETAILED_LABEL_WEIGHTS=}")
        # Starting from the latest checkpoint if it exists, otherwise train from scratch
        trainer.train(resume_from_checkpoint=latest_checkpoint)

        # Save model to disk (weights and configuration)
        model_save_path = f"{self.DATA_PATH}/Model/results/saved_model/{self.unique_tag}/torch_model.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Saving model state dictionary
        torch.save(model.state_dict(), model_save_path)

        # Saving definition as json
        model_config = {
            "model_architecture": "TokenClassificationWithDetailedLabels",
            "base_model": self.MODEL,
            "num_labels": self.NUMBER_BIO_LABELS,
            "num_detailed_labels_per_level": [self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,
            "detailed_label_weights": self.DETAILED_LABEL_WEIGHTS,
            "alpha": self.ALPHA,
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
            study_name = f'study-TOKEN-{self.MODEL}'  # Unique identifier of the study
            study_name = study_name.replace("/", "-")
        
        
        def objective(trial: optuna.trial.Trial):
            
            # Choose levels and weights for integration of auxiliary objective
            # Define hyperparameters to be optimised
            hier_level_options = self.generate_level_combinations()
            hier_level_options[0] = None  # Handle empty set appropriately
            self.HIER_LABELS_LEVELS = trial.suggest_categorical("HIER_LABELS_LEVELS", hier_level_options) # choose from all options defined above
            
            # Suggest weights based on the number of levels chosen
            if self.HIER_LABELS_LEVELS is None:
                self.DETAILED_LABEL_WEIGHTS = None
            else:
                unnormalized_weights = []
                for _ in range(len(self.HIER_LABELS_LEVELS)):
                    unnormalized_weights.append(trial.suggest_float(f"weight_level_{_}", 0.1, 1.0))
                
                # Normalise weights to sum to 1
                total_weight = sum(unnormalized_weights)
                self.DETAILED_LABEL_WEIGHTS = [weight / total_weight for weight in unnormalized_weights]
    
            
            # Update tag statistics with levels above:
            self.get_detailed_labels_info()
            
            # If no detailed labels are included (i.e. no auxiliary objective), set alpha to 1
            if self.HIER_LABELS_LEVELS is None:
                self.ALPHA = 1
            else: 
                self.ALPHA = trial.suggest_float("alpha", 0, 1) # otherwise: suggest alpha value
            
            # Collect info and init wandb
            trial_id = trial.number
            # Initialise wandb with HPO-specific tags or groups
            wandb.init(
                project=config["wandb"]["WANDB_PROJECT"],
                group=f"optuna_study_{study_name}",
                name=f"trial_{trial_id}",
                tags=["optuna", "token", "hpo", f"{study_name}", f"{self.HPO_DESC}"],
                config=trial.params, 
            )
            
            # Save config to wandb
            for key, value in self.config_file.items():
                wandb.config.update({key: value})
            
            
            # Define the model
            model = TokenClassificationWithDetailedLabels(
                model_name_or_path=self.MODEL,
                num_labels=self.NUMBER_BIO_LABELS,
                num_detailed_labels_per_level=[self.unique_level_tags[level] for level in self.tag_statistics.keys()] if self.HIER_LABELS_LEVELS else None,
                alpha=self.ALPHA
            ).to(self.device)

            # Define the trainer
            training_args = self.get_training_arguments()
            bound_collate_fn = functools.partial(self.token_classification_collate_fn)
            trainer = CustomTokenTrainer(
                model=model, 
                args=training_args,
                compute_metrics=evaluator.compute_metrics_wrapper(self.index_to_label, self.HIER_LABELS_LEVELS
                                                                  , self.mapping_dicts, calc_windowdiff=False
                                                                  , pull_extra_metric="model_pk_value_segeval"),
                train_dataset=dataset_dict['train'],
                eval_dataset=dataset_dict['validation'],
                data_collator=bound_collate_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.EARLY_STOPPING_PATIENCE), OptunaPruningCallback(trial, monitor="eval_loss")] # add custom Optuna pruning callback
            )

            # Start training for this trial
            print("Start training.")
            print("Parameters to be tested:")
            print(f"{self.HIER_LABELS_LEVELS=}")
            print(f"{self.DETAILED_LABEL_WEIGHTS=}")
            print(f"{self.ALPHA=}")
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
            db_file = f"{study_name}_{current_datetime}.db"
        
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
    

# Entry point
if __name__ == "__main__":
    
    # Get command line arguments: configuration file
    parser = argparse.ArgumentParser(description="Run Sentence Level Model Training/HPO")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
        
    # Load configuration
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Initialise token-level model using configuration
    token_level_model = TokenLevelModel(config=config)
    
    print(f"################ GET DATA_DICT ################", "\n")
    data_dict = token_level_model.get_data_dict()
    
    print(f"################ GET DETAILED LABELS INFO ################", "\n")
    token_level_model.get_detailed_labels_info()
    
    print(f"################ LOAD DATASET ################", "\n")
    dataset_dict = token_level_model.load_dataset()
    
    # Optional: assert correctness of dataset
    # token_level_model.assert_dataset_correctness(dataset_dict)
    
    # Normal training
    if not token_level_model.IS_HPO_RUN:
        print(f"################ INITIATE TRAINING ################", "\n")
        token_level_model.configure_wandb(config)
        model = token_level_model.setup_model()
        trainer = token_level_model.start_training(dataset_dict, model)
        token_level_model.evaluate_model(dataset_dict['test'])
        wandb.finish()
    # HPO run
    else:
        print(f"################ INITIATE HPO ################", "\n")
        token_level_model.start_HPO(dataset_dict, token_level_model.STUDY_NAME, token_level_model.DB_FILE)