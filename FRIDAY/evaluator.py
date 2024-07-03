# Script to evaluate the performance of our snippet identifer models
# Two major elements:
# 1. Evaluation of the model during training, validation and test on prediction-level (using the HF Trainer API)
# 2. Evaluation of the snippet identifier system against the ground truth regions to evaluate the down-stream task performance (using custom metrics)
# for more details, refer to Chapter 6 of the report

from multiprocessing import Pool
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.metrics import jaccard_distance
from nltk.metrics import edit_distance
from rouge import Rouge
from itertools import islice
import torch
import torch.nn.functional as F
import segeval
from tqdm.notebook import tqdm
import sys
sys.path.append('../DataPreprocessing')

# Default config settings
metrics_config = {
    'iou': True,
    'bleu': True,
    'jaccard': True,
    'precision': True,
    'recall': True,
    'f1': True,
    'precision_region_lvl': True,
    'recall_region_lvl': True,
    'f1_region_lvl': True,
    'edit_distance': False,
    'rouge-1-f': True,
    'rouge-2-f': True,
    'rouge-l-f': True,
    'pk': False,
    'windowdiff': False,
    'cohen_kappa': False
}

def process_batch(batch):
    # Process a batch of pages
    return [calculate_metrics_for_page(page_content) for page_content in batch]

def batch_generator(data, batch_size=100):
    # Generate batches of data
    iterator = iter(data)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch

# Function to calculate Intersection over Union (IoU) / samee as Jaccard Index
def calculate_iou(region_tokens, snippet_tokens):
    region_set = set(region_tokens)
    snippet_set = set(snippet_tokens)
    intersection = region_set.intersection(snippet_set)
    union = region_set.union(snippet_set)
    iou = len(intersection) / len(union) if union else 0
    return iou

# Function to calculate Edit Distance
# using the difflib library
def calculate_edit_distance(region_tokens, snippet_tokens):
    return edit_distance(region_tokens, snippet_tokens)

# Function to calculate Boundary Matching
def boundary_array(tokenized_full_text, snippet_boundaries):
    #print(tokenized_full_text)
    #print(snippet_boundaries)
    start, end = snippet_boundaries
    #print(len(tokenized_full_text))
    boundary_markers = [0] * len(tokenized_full_text)
    boundary_markers[start] = 1
    if end < len(tokenized_full_text):
        boundary_markers[end] = 0
    return boundary_markers

# Function to calculate ROUGE Score
# using the ROUGE library
def calculate_rouge(region_text, snippet_text):
    if not region_text.strip() or not snippet_text.strip():
        return {}
    rouge = Rouge()
    scores = rouge.get_scores(snippet_text, region_text)[0]
    return {
        'rouge-1-f': scores['rouge-1']['f'],
        #'rouge-1-p': scores['rouge-1']['p'],
        #'rouge-1-r': scores['rouge-1']['r'],
        'rouge-2-f': scores['rouge-2']['f'],
        #'rouge-2-p': scores['rouge-2']['p'],
        #'rouge-2-r': scores['rouge-2']['r'],
        'rouge-l-f': scores['rouge-l']['f'],
        #'rouge-l-p': scores['rouge-l']['p'],
        #'rouge-l-r': scores['rouge-l']['r'],
    }

# Function to calculate BLEU score
# using the NLTK library
def calculate_bleu(region_tokens, snippet_tokens):
    reference = [region_tokens]
    candidate = snippet_tokens
    smoothing = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

# Function to calculate Jaccard similarity
def calculate_jaccard(region_tokens, snippet_tokens):
    region_set = set(region_tokens)
    snippet_set = set(snippet_tokens)

    # Check for the case where both sets are empty
    if not region_set and not snippet_set:
        return 1.0
    return 1 - jaccard_distance(region_set, snippet_set)

# Two different methods to obtaining precision, recall, and f1 score: one on the pair-level, one on a global level, using global TP, FP, FN (second option is preferred)
def calculate_f1_token_level(region_tokens, snippet_tokens):
    common_tokens = Counter(region_tokens) & Counter(snippet_tokens)
    true_positives = sum(common_tokens.values())
    false_positives = len(snippet_tokens) - true_positives
    false_negatives = len(region_tokens) - true_positives
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score

def calculate_f1_token_level_counts(region_tokens, snippet_tokens):
    common_tokens = Counter(region_tokens) & Counter(snippet_tokens)
    true_positives = sum(common_tokens.values())
    false_positives = len(snippet_tokens) - true_positives
    false_negatives = len(region_tokens) - true_positives
    
    return true_positives, false_positives, false_negatives

def process_region(region_gold, snippets_predicted, full_page_text, tokenized_full_text, config):
    '''
    Process region and snippets to calculate evaluation metrics
    '''
    region_text_gold = region_gold['text']
    region_tokens_gold = region_gold['tokenized_text']
    
    # Find the best matching snippet for the current region
    best_iou = 0
    best_snippet = None
    best_snippet_text = ""
    best_snippet_tokens = []
    best_snippet_start_idx = None
    best_snippet_end_idx = None
    for snippet in snippets_predicted:
        snippet_text = snippet['text']
        snippet_tokens = snippet['tokenized_text']
        iou = calculate_iou(region_tokens_gold, snippet_tokens) # using intersection over union as similarity metric (i.e. Jaccard Index)
        if iou > best_iou:
            best_iou = iou
            best_snippet_text = snippet_text
            best_snippet_tokens = snippet_tokens
            best_snippet = snippet
            best_snippet_start_idx, best_snippet_end_idx = None, None # find_snippet_indices_long(full_page_text, tokenized_full_text, snippet_tokens, best_snippet_text)
    
    
    # Calculate evaluation metrics
    calculated_metrics = {}
    if config['iou']:
        calculated_metrics['iou'] = best_iou
    if config['bleu']:
        calculated_metrics['bleu'] = calculate_bleu(region_tokens_gold, best_snippet_tokens)
    if config['jaccard']:
        calculated_metrics['jaccard'] = calculate_jaccard(region_tokens_gold, best_snippet_tokens)
    if config['precision'] or config['recall'] or config['f1']:
        precision, recall, f1_score = calculate_f1_token_level(region_tokens_gold, best_snippet_tokens)
        TP, FP, FN = calculate_f1_token_level_counts(region_tokens_gold, best_snippet_tokens)
        calculated_metrics['TP'] = TP
        calculated_metrics['FP'] = FP
        calculated_metrics['FN'] = FN
    if config['precision']:
        calculated_metrics['precision_region_lvl'] = precision
    if config['recall']:
        calculated_metrics['recall_region_lvl'] = recall
    if config['f1']:
        calculated_metrics['f1_region_lvl'] = f1_score
    if config['edit_distance']:
        calculated_metrics['edit_distance'] = calculate_edit_distance(region_tokens_gold, best_snippet_tokens)
    if any(key.startswith('rouge') for key in config):
        rouge_scores = calculate_rouge(region_text_gold, best_snippet_text)
        calculated_metrics.update(rouge_scores)
        
    if (config['pk'] or config['windowdiff'] or config['cohen_kappa']) and (best_snippet_start_idx and best_snippet_end_idx):
        region_boundaries = (region_gold['start_idx_in_page'], region_gold['end_idx_in_page'])
        predicted_boundaries = (best_snippet_start_idx, best_snippet_end_idx)      

        region_boundary_markers = boundary_array(tokenized_full_text, region_boundaries)
        predicted_boundary_markers = boundary_array(tokenized_full_text, predicted_boundaries)

        window_size = 128
        
        if config['pk']:
            pk_score = calculate_pk(region_boundary_markers, predicted_boundary_markers, window_size=window_size)
            calculated_metrics['pk'] = pk_score
        
        if config['windowdiff']:
            windowdiff_score = calculate_windowdiff(region_boundary_markers, predicted_boundary_markers, window_size=window_size)
            calculated_metrics['windowdiff'] = windowdiff_score
            
        true_labels = [1 if i in region_boundary_markers else 0 for i in range(len(tokenized_full_text))]
        predicted_labels = [1 if i in predicted_boundary_markers else 0 for i in range(len(tokenized_full_text))]

        
        if config['cohen_kappa']:
            calculated_metrics['cohen_kappa'] = cohen_kappa_score(true_labels, predicted_labels)

    
    return calculated_metrics

# Function to calculate metrics for each page
def calculate_metrics_for_page(args):
    page_content, ground_truth, comparison_object, config = args
    regions_gold = page_content[ground_truth] # regions
    snippets_predicted = page_content[comparison_object]
    full_page_text = page_content['full_text']
    tokenized_full_text = page_content['tokenized_full_text']
    
    # print(page_content['full_text'])
    if not regions_gold or not snippets_predicted:
        return None  # Return None to indicate no metrics should be calculated for this page

    page_metrics = [process_region(region_gold, snippets_predicted, full_page_text, tokenized_full_text, config) for region_gold in regions_gold]
    return page_metrics

def aggregate_metrics(page_metrics_list, config):
    # Initialize counters for TP, FP, FN
    total_TP, total_FP, total_FN = 0, 0, 0

    # Iterate through each batch (list of pages)
    for batch in page_metrics_list:
        for page_metrics in batch:
            # Check if page_metrics is None (skipped page)
            if page_metrics is None:
                continue  # Skip this page's metrics
            for region_metrics in page_metrics:
                # Aggregate TP, FP, FN from each region
                total_TP += region_metrics.get('TP', 0)
                total_FP += region_metrics.get('FP', 0)
                total_FN += region_metrics.get('FN', 0)

    aggregated_metrics = {}

    # Calculate and include global precision, recall, and F1 scores if requested
    if config.get('precision'):
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        aggregated_metrics['precision'] = precision
    if config.get('recall'):
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        aggregated_metrics['recall'] = recall
    if config.get('f1'):
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        aggregated_metrics['f1'] = f1_score
    
    for key in ['iou', 'bleu', 'jaccard', 'edit_distance', 'rouge'
                , 'precision_region_lvl', 'recall_region_lvl', 'f1_region_lvl'
                , 'pk', 'windowdiff', 'cohen_kappa'
                , 'rouge-1-f', 'rouge-2-f', 'rouge-l-f']:
        if key in config and config[key]:
            values = [metrics[key] for batch in page_metrics_list for page_metrics in batch if page_metrics for metrics in page_metrics if key in metrics]
            aggregated_metrics[key] = np.mean(values) if values else 0

    return aggregated_metrics



# Main evaluation function with parallel processing
def evaluate_snippets_parallel(data_dict, comparison_object:str = "predicted_snippets", ground_truth: str="refined_regions"
                               , config:dict = metrics_config, batch_size=10, DEBUG=False, tokenizer=None):
    '''
    :param data_dict: nested dictionary containing all the relevant data
    :param comparison_object: name of the page_level element that shall be used for comparison (needs to contain 'text' and 'tokenized_text' in dict)
    :param ground_truth: name of the page_level element that shall be used as the ground_truth (i.e. the regions)
    :param config: configuration dictionary stating which metrics shall be calculated
    :param batch_size: batch_size for multiprocessing (default: 10)
    '''
    global global_tokenizer
    global_tokenizer = tokenizer
    print("Evaluation Started.")
    page_contents = [(content, ground_truth, comparison_object, config) for doc_id, pages in data_dict.items() for page, content in pages.items() if not isinstance(content, str)]
    batches = batch_generator(page_contents, batch_size=batch_size)

    if DEBUG:
        results = [process_batch(batch) for batch in batches]
    else:
        with Pool(10) as pool:
            results = list(tqdm(pool.imap(process_batch, batches), total=len(page_contents) // batch_size))

    # Aggregate results
    aggregated_metrics = aggregate_metrics(results, config)

    # Print aggregated results
    for metric, value in aggregated_metrics.items():
        print(f"Average {metric.capitalize()} Score: {value}")
        
    return aggregated_metrics


###### Methods for HF Trainer API: ######
def generate_random_baseline(labels_flat, index_to_tag):
    # used as comparison
    indices = [index for index in index_to_tag.keys() if index != -100 and index != -1]
    random_indices = np.random.choice(indices, size=len(labels_flat))
    return random_indices

def simplify_boundaries(labels):
    '''
    Simplifies a sequence of boundary markers so that consecutive '1's
    are reduced to a single '1' marking the start of a segment.
    Example: [0, 0, 1, 1, 1, 0, 1, 1] --> [0, 0, 1, 0, 0, 0, 1, 0]
    '''
    simplified = np.array(labels)
    # Find indices where '1's occur
    ones_indices = np.where(simplified == 1)[0]
    # Keep only the first '1' in each series of consecutive '1's --> we set this as the start of a segment
    to_zero = ones_indices[1:][np.diff(ones_indices) == 1]
    simplified[to_zero] = 0
    return simplified

        
def calculate_pk(true_labels, predicted_labels, window_size):
    discrepancies = 0
    for i in range(len(true_labels) - window_size):
        true_segment_change = true_labels[i] != true_labels[i + window_size]
        predicted_segment_change = predicted_labels[i] != predicted_labels[i + window_size]
        if true_segment_change != predicted_segment_change:
            discrepancies += 1
    return discrepancies / (len(true_labels) - window_size)

def calculate_windowdiff(true_labels, predicted_labels, window_size):
    if len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted labels must have same length.")
    
    num_boundaries_diff = 0
    for i in range(len(true_labels) - window_size + 1):
        true_boundaries = sum(1 for j in range(i, i + window_size - 1) if true_labels[j] != true_labels[j + 1])
        predicted_boundaries = sum(1 for j in range(i, i + window_size - 1) if predicted_labels[j] != predicted_labels[j + 1])
        num_boundaries_diff += abs(true_boundaries - predicted_boundaries)
    
    return num_boundaries_diff / (len(true_labels) - window_size + 1)


def binary_boundaries_to_masses(binary_boundaries):
    '''
    Convert binary boundary marker sequence to segment lengths (masses)
    Example: [0, 0, 1, 0, 0, 0, 1, 0] --> [3, 4, 1] for segment lengths.
    '''
    segments = []
    current_segment_length = 0
    # go through binary boundaries and count segment lengths
    for marker in binary_boundaries:
        if marker == 1:
            if current_segment_length > 0:
                segments.append(current_segment_length)
            current_segment_length = 1 
        else:
            current_segment_length += 1
    if current_segment_length > 0: 
        segments.append(current_segment_length)
    return segments

def calculate_pk_segeval(true_labels, predicted_labels):
    '''
    Calculate the pk score using the segeval library
    '''
    try:
        true_masses = binary_boundaries_to_masses(true_labels)
        predicted_masses = binary_boundaries_to_masses(predicted_labels)
        return float(segeval.pk(true_masses, predicted_masses))
    except Exception as e:
        print(f"An error occurred while using segeval to calculat the pk score: {e}")
        return None

def calculate_windowdiff_segeval(true_labels, predicted_labels):
    '''
    Calculate the windowdiff score using the segeval library
    '''
    try:
        true_masses = binary_boundaries_to_masses(true_labels)
        predicted_masses = binary_boundaries_to_masses(predicted_labels)
        return float(segeval.window_diff(true_masses, predicted_masses))
    except Exception as e:
        print(f"An error occurred while using segeval to calculat the windowdiff score: {e}")
        return None

def custom_compute_metrics(eval_pred, index_to_tag, HIER_LABELS_LEVELS=None, mapping_dicts=None, calc_windowdiff=True, pull_extra_metric=None):
    logits_output, labels_output = eval_pred
    
    # Ensure logits are PyTorch tensors
    if HIER_LABELS_LEVELS:
        BIO_logits = torch.tensor(logits_output[0], dtype=torch.float32)
        detailed_logits = [torch.tensor(dl, dtype=torch.float32) for dl in logits_output[1]]
        BIO_labels = torch.tensor(labels_output[0], dtype=torch.long)
        detailed_labels = torch.tensor(labels_output[1], dtype=torch.long)
    else:
        BIO_logits = torch.tensor(logits_output, dtype=torch.float32)
        BIO_labels = torch.tensor(labels_output, dtype=torch.long)
    
    # Compute Cross Entropy Loss for BIO labels
    valid_indices_BIO = (BIO_labels != -1) & (BIO_labels != -100)
    bio_cross_entropy_loss = F.cross_entropy(BIO_logits[valid_indices_BIO], BIO_labels[valid_indices_BIO])

    # Compute metrics for BIO labels
    BIO_predictions = torch.argmax(BIO_logits, dim=-1).flatten()
    BIO_labels_flat = BIO_labels.flatten()
    valid_indices_BIO_flat = (BIO_labels_flat != -1) & (BIO_labels_flat != -100)
    valid_predictions_BIO = BIO_predictions[valid_indices_BIO_flat].cpu().numpy()
    valid_labels_BIO = BIO_labels_flat[valid_indices_BIO_flat].cpu().numpy()

    # Compute standard metrics for BIO labels
    normal_metrics = compute_standard_metrics(valid_labels_BIO, valid_predictions_BIO, index_to_tag
                                              , calc_windowdiff=calc_windowdiff
                                              , cross_entropy_loss=bio_cross_entropy_loss.item())
    results = {"normal_labels_metrics": normal_metrics}
    
    if pull_extra_metric:
        results[pull_extra_metric] = normal_metrics[pull_extra_metric]
    
    if HIER_LABELS_LEVELS:
        hierarchical_metrics = {}
        for i, level in enumerate(HIER_LABELS_LEVELS):  # Iterate through each hierarchical level

            # unflattened for loss:
            level_labels_unchanged = detailed_labels[:, :, i]
            level_logits_unchanged = detailed_logits[i]
            valid_indices_level_unchanged = valid_indices_level_unchanged = (level_labels_unchanged != -1) & (level_labels_unchanged != -100)
            level_cross_entropy_loss = F.cross_entropy(level_logits_unchanged[valid_indices_level_unchanged], level_labels_unchanged[valid_indices_level_unchanged])

            # flattened:
            level_labels = detailed_labels[:, :, i].flatten()  # Flatten to match predictions shape
            level_logits = detailed_logits[i].reshape(-1, detailed_logits[i].shape[-1])  # Reshape for softmax
            
            valid_indices_level = valid_indices_level = (level_labels != -1) & (level_labels != -100)  # Excluding padding
            valid_predictions_level = np.argmax(level_logits, axis=-1)[valid_indices_level]
            valid_labels_level = level_labels[valid_indices_level]
            
            # Compute metrics per hierarchical level
            level_cross_entropy_loss = F.cross_entropy(level_logits[valid_indices_level], level_labels[valid_indices_level], reduction='mean')
            hierarchical_metrics[f"level_{level}"] = compute_standard_metrics(valid_labels_level.cpu().numpy(), valid_predictions_level.cpu().numpy(), mapping_dicts[f"level_{level}"]
                                                                              , detailed=True
                                                                              , calc_windowdiff=calc_windowdiff
                                                                              , cross_entropy_loss=level_cross_entropy_loss.item())
        
        results["hierarchical_labels_metrics"] = hierarchical_metrics

    return results

def compute_standard_metrics(valid_labels, valid_predictions, index_to_tag, detailed=False, calc_windowdiff=True, cross_entropy_loss=None):
    if not detailed:
        binary_true_tags = [1 if label in index_to_tag and index_to_tag[label] in ["B", "I", "E"] else 0 for label in valid_labels]
        binary_predicted_tags = [1 if pred in index_to_tag and index_to_tag[pred] in ["B", "I", "E"] else 0 for pred in valid_predictions]
        average_method = 'binary'
    else:
        binary_true_tags = valid_labels
        binary_predicted_tags = valid_predictions
        average_method = 'macro'
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_true_tags, binary_predicted_tags, average=average_method, zero_division=0)
    accuracy = accuracy_score(binary_true_tags, binary_predicted_tags)
    kappa = cohen_kappa_score(valid_labels, valid_predictions)
    
    metrics_result = {
        "cross_entropy_loss": cross_entropy_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "cohen_kappa": kappa 
    }
    
    if not detailed:
        # Additional metrics for non-detailed labels only (i.e. pk and windowdiff only makes sense for main objective
        metrics_result.update(compute_additional_metrics(binary_true_tags, binary_predicted_tags, calc_windowdiff))
    
    return metrics_result

def compute_additional_metrics(true_labels, predicted_labels, calc_windowdiff=True):
    
    true_labels_simplified = simplify_boundaries(true_labels)
    predicted_labels_simplified = simplify_boundaries(predicted_labels)
    
    # Calculate P_k and WindowDiff, and metrics for random baseline
    window_size = 128
    model_pk_value = calculate_pk(true_labels_simplified, predicted_labels_simplified, window_size)
    if calc_windowdiff:
        model_windowdiff_value = calculate_windowdiff(true_labels_simplified, predicted_labels_simplified, window_size)
        model_windowdiff_value_segeval = calculate_windowdiff_segeval(true_labels_simplified, predicted_labels_simplified)
        
    model_pk_value_segeval = calculate_pk_segeval(true_labels_simplified, predicted_labels_simplified)
    
    # Generate random baseline predictions and calculate their metrics
    # most of the time around 0.5
    random_predictions = generate_random_baseline(true_labels, {i: 'tag' for i in set(true_labels)})
    random_predictions_simplified = simplify_boundaries(random_predictions)
    random_precision, random_recall, random_f1, _ = precision_recall_fscore_support(
        true_labels, random_predictions, average='binary', zero_division=0)
    random_accuracy = accuracy_score(true_labels, random_predictions)
    
    random_pk_value = calculate_pk(true_labels_simplified, random_predictions_simplified, window_size)
    if calc_windowdiff:
        random_windowdiff_value = calculate_windowdiff(true_labels_simplified, random_predictions_simplified, window_size)
        random_windowdiff_value_segeval = calculate_windowdiff_segeval(true_labels_simplified, random_predictions_simplified)
        
    random_pk_value_segeval = calculate_pk_segeval(true_labels_simplified, random_predictions_simplified)
    
    metrics_dict =  {
        "model_pk": model_pk_value,
        "model_pk_value_segeval": model_pk_value_segeval,
        "model_windowdiff": model_windowdiff_value,
        "model_windowdiff_value_segeval": model_windowdiff_value_segeval,
        "random_baseline_accuracy": random_accuracy,
        "random_baseline_precision": random_precision,
        "random_baseline_recall": random_recall,
        "random_baseline_f1": random_f1,
        "random_baseline_pk": random_pk_value,
        "random_pk_value_segeval": random_pk_value_segeval,
        "random_baseline_windowdiff": random_windowdiff_value,
        "random_windowdiff_value_segeval": random_windowdiff_value_segeval
    } if calc_windowdiff else {
        "model_pk": model_pk_value,
        "model_pk_value_segeval": model_pk_value_segeval,
        "random_baseline_accuracy": random_accuracy,
        "random_baseline_precision": random_precision,
        "random_baseline_recall": random_recall,
        "random_baseline_f1": random_f1,
        "random_baseline_pk": random_pk_value,
        "random_pk_value_segeval": random_pk_value_segeval
    }
    

    return metrics_dict

def compute_metrics_wrapper(index_to_tag, HIER_LABELS_LEVELS=None, mapping_dicts=None, calc_windowdiff=True, pull_extra_metric=None):
    '''
    Wrapper function to create a compute_metrics function with custom parameters
    Matching the format required for the HF Trainer API
    '''
    def compute_metrics(eval_pred):
        return custom_compute_metrics(eval_pred, index_to_tag, HIER_LABELS_LEVELS, mapping_dicts, calc_windowdiff, pull_extra_metric)
    return compute_metrics

