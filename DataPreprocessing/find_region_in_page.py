# This script implements the custom pre-processing algorithm for finding a region in a page of text.
# Please refer to Chapter 4 for more details on the implementation and the algorithm.

from collections import Counter
import custom_sentence_tokenizer
import re
import unicodedata
from transformers import AutoTokenizer
from functools import lru_cache

# compiled_pattern_exact_match = re.compile(r"[�|’'‘\"”*°\d\.\,]")
compiled_match_enum_marker = r'.*\n\s*((\d+(\.\d+)*)|(\([a-zA-Z]\))|(\([a-zA-Z])|([a-zA-Z]\))).*'
compiled_previous_sent_enum_marker = r'^.*\n\s*(\d+(\.\d+)*)\.$'
compiled_match_double_linebreak = r'\n\s*\n.*$'
compiled_match_decimal_with_optional_whitespaces = r'\s*(\d+(\.\d+)*)\.\s*'

@lru_cache(maxsize=128)
def is_similar(a, b, allowed_diff=2):
    '''
    Check if two strings are similar allowing up to `allowed_diff` character differences
    '''
    a_cleaned = a.translate(str.maketrans('', '', "[�|’'‘\"”*°\d\.\,]"))
    b_cleaned = b.translate(str.maketrans('', '', "[�|’'‘\"”*°\d\.\,]"))
    
    diff_count = sum(1 for ac, bc in zip(a_cleaned, b_cleaned) if ac != bc)
    length_diff = abs(len(a_cleaned) - len(b_cleaned))
    
    return diff_count + length_diff <= allowed_diff

@lru_cache(maxsize=128)
def compute_ngrams(text_tuple, n):
    text = list(text_tuple)
    return Counter(zip(*[text[i:] for i in range(n)]))

def ngram_overlap(source_ngrams, target_ngrams):
    '''
    Calculate the overlap ratio between two sets of n-grams
    '''
    common_ngrams = source_ngrams & target_ngrams
    total = sum(target_ngrams.values())
    
    return sum(common_ngrams.values()) / total if total > 0 else 0


def find_best_match_index_overlap(tokenizer: AutoTokenizer, tokenized_page_text: list[str], tokenized_region_text: list[str], n=5, threshold=0.05):
    
    # Quick check for direct match of the first 50 tokens (if region is smaller, use its length)
    region_len = min(len(tokenized_region_text), 50)
    region_start = "".join(tokenizer.convert_tokens_to_string(tokenized_region_text[:region_len]).split())
    len_tokenized_page_text = len(tokenized_page_text)
    len_tokenized_region_text = len(tokenized_region_text)

    # If region is extremely short, we should compare smaller n-grams
    if len_tokenized_region_text <= 16:
        n = 2

    for idx in range(len_tokenized_page_text - region_len + 1):
        page_window = "".join(tokenizer.convert_tokens_to_string(tokenized_page_text[idx:idx + region_len]).split())
        comparison_length  = min(len(region_start), len(page_window))
        if comparison_length < 8: # we should not compare less than e.g. 8 characters, otherwise we might match everything
            continue
        if is_similar(region_start[:comparison_length], page_window[:comparison_length], allowed_diff=2): # allow two characters to be different to accoutn for potential OCR mismatches
            return idx, False

    # Proceed with n-gram overlap test if no direct match found
    best_match_idx = None
    highest_overlap = threshold # minimum overlap needed (5%) --> TBD
    multiple_hits = False

    while best_match_idx is None:
        # get ngrams for region (stay the same)
        target_ngrams = compute_ngrams(tuple(tokenized_region_text), n)
        max_range = max(10, len_tokenized_page_text)
        for idx in range(max_range):
            window = tokenized_page_text[idx:idx + len_tokenized_region_text]

            source_ngrams = compute_ngrams(tuple(window), n)
            overlap = ngram_overlap(source_ngrams, target_ngrams)
            
            if overlap > highest_overlap:
                highest_overlap = overlap
                best_match_idx = idx
                if highest_overlap == 1:
                    break
            elif best_match_idx is not None and overlap == highest_overlap and idx >= best_match_idx+len_tokenized_page_text*0.25: # if not close, but same value --> potentially other correct region
                multiple_hits = True
        # another itereation with lower n --> if multiple hits later --> allow to search broader span
        if n < 3:
            break
        n -= 1
    return best_match_idx, multiple_hits


 # Dynamically determine if a character is a quotation mark
def is_quotation_mark(char):
    # Check if a character is a quotation mark based on its Unicode category
    category = unicodedata.category(char)
    # Categories for initial and final quotation marks, and general punctuation which might include some quotes
    return category in ('Pi', 'Pf', 'Po')

def remove_likely_OCR_mismatches(text, is_end: bool):
    # First, remove characters matched by compiled_pattern anywhere in string
    cleaned_text = text.replace('�', '').replace('*', '').replace('°', '').replace('|', '') 

    # check if character should be removed based on Unicode category
    def should_remove(char):
        category = unicodedata.category(char)
        return category == 'Co'

    if is_end:
        cleaned_text = ''.join(char if i < len(cleaned_text) - 1 or not is_quotation_mark(char) else '' 
                               for i, char in enumerate(cleaned_text))
    else:
        cleaned_text = ''.join(char if i > 0 or not is_quotation_mark(char) else '' 
                               for i, char in enumerate(cleaned_text))

    cleaned_text = ''.join(char for char in cleaned_text if not should_remove(char))

    return cleaned_text

def get_adjustment_value(tokenizer, matched_text, text, backwards=True):
    '''
    Helper function to get the adjustment value for the start or end index (toglged with backwards) based on the matched text
    '''
    # Reconstructed tokens list
    reconstructed_tokens = []
    tokens = tokenizer.tokenize(text)

    if backwards:
        # Get both token and its index in reverse order
        for i, token in reversed(list(enumerate(tokens))):
            reconstructed_tokens.insert(0, token)
            reconstructed_text = tokenizer.convert_tokens_to_string(reconstructed_tokens)
            if reconstructed_text.strip().startswith(matched_text):
                enumeration_start_idx = i  
                # Adjust index to include enumeration marker
                return len(tokens) - enumeration_start_idx
    else:
        for i, token in enumerate(tokens):
            reconstructed_tokens.append(token)
            reconstructed_text = tokenizer.convert_tokens_to_string(reconstructed_tokens)
            if reconstructed_text.strip().startswith(matched_text):
                enumeration_start_idx = i 
                # Adjust value in case single token leads to uknown character and requires previous one
                if tokenizer.convert_tokens_to_string([token]) == "�":
                    enumeration_start_idx -= 1
                return enumeration_start_idx

    return 0


def find_adjusted_start_index(tokenizer, tokenized_page_text, tokenized_region_text, region_token_length, initial_start_idx, max_iterations, move_tokens_per_iteration, matching_chars, scan_start_range):
    ''' 
    Find the adjusted start index for matching a region of text within a page of text
    '''
    
    adjusted_start_idx = initial_start_idx
    iteration_details = []
    
    # Dynamically adjust matching_chars in iterations
    for iteration in range(3):
        matching_char_count = int(matching_chars / (2 ** iteration))
        adjusted_idx, match_found_at_iteration = start_string_match(
            tokenizer, tokenized_page_text, tokenized_region_text, adjusted_start_idx, max_iterations,
            move_tokens_per_iteration, matching_char_count, scan_start_range
        )
        iteration_details.append((adjusted_idx, match_found_at_iteration, matching_char_count))
        
        # If a match is found in the current iteration, no need to continue
        if match_found_at_iteration != -1 and match_found_at_iteration <= max_iterations//2:
            break
    
    # Select the best match from the iterations
    best_ngram_overlap = -1
    for adjusted_idx, match_found_at_iteration, matching_char_count in iteration_details:
        if adjusted_idx != adjusted_start_idx:
            region_n_grams = compute_ngrams(tuple(tokenized_region_text), 4)
            assumed_region_n_grams = compute_ngrams(tuple(tokenized_page_text[adjusted_idx:adjusted_idx+region_token_length]), 4)
            current_overlap = ngram_overlap(region_n_grams, assumed_region_n_grams)
            
            # Compare with the current best match
            if current_overlap > best_ngram_overlap:
                adjusted_start_idx = adjusted_idx
                best_ngram_overlap = current_overlap
    
    return adjusted_start_idx

def start_string_match(tokenizer, tokenized_page_text, tokenized_region_text, adjusted_start_idx, MAX_ITERATIONS, MOVE_FORWARD_TOKENS_PER_IT, matching_chars, scan_start_range):
    """
    Finds the starting index of a matching string in the page text.
    Called by find_adjusted_start_index.
    """
    match_found = False
    match_found_at_iteration = -1
    iterations = 0

    while not match_found and iterations < MAX_ITERATIONS:
        # Adjust region text start based on iterations, progressively narrowing the search
        iteration_adjustment = MOVE_FORWARD_TOKENS_PER_IT * iterations
        # Since the adjustment might change tokenization, re-tokenize the adjusted part of the region_text
        adjusted_region_tokens = tokenized_region_text[iteration_adjustment:]
        adjusted_region_text = tokenizer.convert_tokens_to_string(adjusted_region_tokens)

        # Preprocess adjusted_region_text for comparison
        adjusted_region_text_start = remove_likely_OCR_mismatches("".join(adjusted_region_text[:matching_chars].split()), is_end=False)

        for idx in scan_start_range:
            page_window_text = tokenizer.convert_tokens_to_string(tokenized_page_text[idx:idx + len(adjusted_region_tokens)])
            page_window_text_cleaned = remove_likely_OCR_mismatches("".join(page_window_text.split()), is_end=False)
            
            # don't compare only a few characters!
            if not page_window_text_cleaned.isspace() and len(page_window_text_cleaned) >=8 and (page_window_text_cleaned.lower().startswith(adjusted_region_text_start.lower()) or adjusted_region_text_start.lower().startswith(page_window_text_cleaned.lower())):
                adjusted_start_idx = idx
                match_found = True
                match_found_at_iteration = iterations
                break

        if match_found:
            break 
        iterations += 1

    return adjusted_start_idx, (match_found_at_iteration != -1)



def end_string_match(tokenizer, tokenized_page_text, region_text, tokenized_region_text, adjusted_start_idx, adjusted_end_idx, MAX_ITERATIONS, MOVE_BACKWARD_TOKENS_PER_IT, matching_chars, scan_end_range):
    match_found = False
    match_found_at_iteration = -1
    iterations = 0
    while(True):
        region_text_end = remove_likely_OCR_mismatches("".join(region_text.split()).strip(" \n\t")[-matching_chars:], is_end=True) #remove elements that can likeley be attributed to OCR mismatches
        if len(region_text_end) <= 4: # don't compare exremely small words (i.e. characters)
            break
        for idx in scan_end_range:

            reconstructed_text = tokenizer.convert_tokens_to_string(tokenized_page_text[adjusted_start_idx:idx+1])
            reconstructed_text_cleaned = remove_likely_OCR_mismatches("".join(reconstructed_text.split()), is_end=True) # do not compare with apostroph types etc.
            
            if reconstructed_text and reconstructed_text_cleaned.lower().endswith(region_text_end.lower()): # Adjusted to check for the end of region_text --> remove "”\n?
                if reconstructed_text[-1].isspace() or reconstructed_text[-1] == "�": # if whitespace (i.e., \n, " ", \t, ...) found --> rather stop at period
                    continue
                adjusted_end_idx = idx
                match_found = True
                match_found_at_iteration = iterations
            elif match_found:
                break
        if match_found or iterations >= MAX_ITERATIONS:
            break
        else:
            iterations += 1
            region_text = tokenizer.convert_tokens_to_string(tokenized_region_text[:-MOVE_BACKWARD_TOKENS_PER_IT*iterations])
            # region_token_length = len(tokenized_region_text[:-MOVE_BACKWARD_TOKENS_PER_IT*iterations])
    return adjusted_end_idx, match_found_at_iteration

def adjust_region_indices(tokenizer: AutoTokenizer, sentenizer: custom_sentence_tokenizer.Sentenizer, best_match_idx: int, page_text: str, tokenized_page_text: list, region_text: str, span: int, matching_chars=64, is_case_sensitive=False):
    """
    Refines the initial identification of the beginning and end of a region inside a page by matching and comparing text around the initially identified points.
    The search areas (as specified by span) should be bigger for RoBERTa-like models as their tokenization generally yields more tokens.
    """

    ############## DEFINITIONS ##############
    no_initial_match_found = False # to save initial assumption
    if best_match_idx is None:
        no_initial_match_found = True
        best_match_idx = 0 # allow searching everything

    # Tokenize the region text to get the length in tokens
    tokenized_region_text = tokenizer.tokenize(region_text)
    region_token_length = len(tokenized_region_text)
    page_token_length = len(tokenized_page_text)

    # Define the range to scan around the best_match_idx for start and end adjustments
    # max_possible_start_to_fit_region = 512 if page_token_length == 512 else page_token_length - region_token_length + 1
    scan_start_range = range(max(0, best_match_idx - span), min(best_match_idx + span, page_token_length))

    adjusted_start_idx = best_match_idx
    adjusted_end_idx = min(best_match_idx + region_token_length, page_token_length-1)

    ############## ADJUST START ##############

    # 1. Step: Check for a better start index (keep match as close as possible to account for potential whitespace etc.)
    # if no match found --> adjust start of region to later token beginning to account for potetnial OCR mismatches at the beginning of the sentence, if still no match found -> continue
    
    MAX_ITERATIONS = 16 if no_initial_match_found else 32# 16
    MOVE_FORWARD_TOKENS_PER_IT = 2 if no_initial_match_found else 1
    adjusted_start_idx = find_adjusted_start_index(tokenizer, tokenized_page_text, tokenized_region_text, region_token_length, adjusted_start_idx
                                                   , MAX_ITERATIONS, MOVE_FORWARD_TOKENS_PER_IT, matching_chars, scan_start_range)
    if no_initial_match_found and adjusted_start_idx == 0:
        # declare matching unsuccessful --> stop and return original region
        raise Exception("No match possible!")

    # 2. Step: Check for potential enumeration markers or sentence beginnings (i.e. more sensible start of region)
    LOOK_BEFORE_TOKENS = 40 
    look_before_text = tokenizer.convert_tokens_to_string(tokenized_page_text[max(0, adjusted_start_idx-LOOK_BEFORE_TOKENS):adjusted_start_idx])
    look_before_sent_split = sentenizer.tokenize_into_sentences(look_before_text)
    current_start_token_string = tokenizer.convert_tokens_to_string([tokenized_page_text[adjusted_start_idx]])
    
    # 2.1. Check for enumeration markers
    match_enum_marker = re.findall(compiled_match_enum_marker, look_before_text)  # match numerical and (d) markers
    if match_enum_marker:
        matched_text = match_enum_marker[-1][0].strip()
        adjusted_start_idx -= get_adjustment_value(tokenizer, matched_text, look_before_text, backwards=True)

    # 2.2. Check for sentence beginnings
    # only if current start token is not a sensible start (i.e. whitespace, period, lowercase letter) and there exists a previous sentence
    elif ("." in current_start_token_string or ")" == current_start_token_string[-1] or current_start_token_string.isspace() or (is_case_sensitive and current_start_token_string.strip().islower())) and len(look_before_sent_split) > 1: 
        last_sent_tokenized = tokenizer.tokenize(look_before_sent_split[-1])
        if len(last_sent_tokenized) <= 32: # only adjust if previous sentence ending is close enough
            # First double-check: prediction is likely already accurate if best_match_idx and adjusted_start_idx closely align
            if not abs(adjusted_start_idx-best_match_idx) <=2:
                # Second double-check: is next sentence only a few tokens aways? --> rather take this as start, than going backwards (example: 19956318, page 38, region 3)
                look_for_close_next_sent_text = tokenizer.convert_tokens_to_string(tokenized_page_text[adjusted_start_idx+1:adjusted_start_idx+8])
                look_for_close_next_sent_split = sentenizer.tokenize_into_sentences(look_for_close_next_sent_text)
                if len(look_for_close_next_sent_split)>1: # there is  a close next sentence
                    next_sent_tokenized = tokenizer.tokenize(look_for_close_next_sent_split[0])
                    adjusted_start_idx += (len(next_sent_tokenized) + 1)
                else:
                    adjusted_start_idx -= get_adjustment_value(tokenizer, look_before_sent_split[-1], look_before_text, backwards=True) # adjust to beginning of last esntence

                    # check if previous sentence acts as an enumeration marker, then rather add this
                    second_last_sentence = look_before_sent_split[-2]
                    match = re.match(compiled_previous_sent_enum_marker, second_last_sentence)
                    if match:
                        matched_text = match.group(0)
                        tokens = tokenizer.tokenize(second_last_sentence)
                        # Find the index of the token that starts the enumeration marker
                        enumeration_start_idx = next((i for i, token in enumerate(tokens) if tokenizer.convert_tokens_to_string([token]).strip() == matched_text), None)
                        if enumeration_start_idx is not None:
                            # Adjust the start index to point to the enumeration marker
                            adjusted_start_idx -= (len(tokens) - enumeration_start_idx - 1)
            elif (current_start_token_string == "."): # if only ".", move forward to next non-whitespace
                look_ahead_text = tokenizer.convert_tokens_to_string(tokenized_page_text[adjusted_start_idx+1:adjusted_start_idx+8])
                match_character = re.search(r'\S', look_ahead_text)
                if match_character:
                    matched_text = match_character.group(0).strip()
                    adjusted_start_idx += 1 + get_adjustment_value(tokenizer, matched_text, look_ahead_text, backwards=False)
        # is there a double line break in close proximity? --> indicates new paragraph
        # specific match, but this is necessary to still allow as much flexibility as possible
        else:
            match_double_linebreak = re.search(compiled_match_double_linebreak, look_before_text)
            if match_double_linebreak:
                matched_text = match_double_linebreak.group(0).strip()
                adjusted_start_idx -= get_adjustment_value(tokenizer, matched_text, look_before_text, backwards=True)
    # last resort: if current_start_token_string is whitespace or period -> move to closest following next non-whitespace character
    elif (current_start_token_string.isspace() or current_start_token_string == "."):
        look_ahead_text = tokenizer.convert_tokens_to_string(tokenized_page_text[adjusted_start_idx+1:adjusted_start_idx+8])
        match_character = re.search(r'\S', look_ahead_text)
        if match_character:
            matched_text = match_character.group(0).strip()
            adjusted_start_idx += 1 + get_adjustment_value(tokenizer, matched_text, look_ahead_text, backwards=False)

    ############## ADJUST END ##############
    
    # 1. Decide if search space needs to be extended due to potential whitespace or error tokens
    upper_end = min(best_match_idx + region_token_length + span, page_token_length+1)
    percent_additional_tokens = 0
    
    if upper_end !=  page_token_length+1 or span == page_token_length:
        # regex for special whitespace characters (e.g., em space, en space, non-breaking space)
        special_whitespace_pattern = r'[\u2000-\u200B\u202F\u205F\u3000\u00A0]'
        special_whitespace_matches = re.findall(special_whitespace_pattern, page_text)

        # add potential error tokens (only RoBERTa)
        error_tokens = ['âĢ', 'ĥ']  # Specify known error tokens
        error_token_count = sum(token in error_tokens for token in tokenized_page_text)
        
        upper_end += len(special_whitespace_matches) + error_token_count
        percent_additional_tokens = (len(special_whitespace_matches) + error_token_count)/float(page_token_length)


    scan_end_range = range(max(0, adjusted_start_idx), min(upper_end, page_token_length-1)) # case 20501279, 17 --> not high enough because of all the whitespace tokens --> adjustment made above
    
    # 2. Check for a better end index

    # Check for a better end index (keep match as close to last end of region as possible)
    adjusted_end_idx_candidate, match_found_at_iteration = end_string_match(tokenizer, tokenized_page_text, region_text, tokenized_region_text, adjusted_start_idx, adjusted_end_idx
                                                     , MAX_ITERATIONS=MAX_ITERATIONS, MOVE_BACKWARD_TOKENS_PER_IT=MOVE_FORWARD_TOKENS_PER_IT, matching_chars=matching_chars, scan_end_range=scan_end_range)
    # do another iteration if no match --> compare less than matching chars before
    adjusted_end_idx_candidate2 = adjusted_end_idx_candidate
    if match_found_at_iteration == -1 or match_found_at_iteration > MAX_ITERATIONS//2:
        adjusted_end_idx_candidate2, match_found_at_iteration = end_string_match(tokenizer, tokenized_page_text, region_text, tokenized_region_text, adjusted_start_idx, adjusted_end_idx
                                                         , MAX_ITERATIONS=MAX_ITERATIONS, MOVE_BACKWARD_TOKENS_PER_IT=MOVE_FORWARD_TOKENS_PER_IT, matching_chars=int(matching_chars/2), scan_end_range=scan_end_range)
        # one last iteration if still no match found
        # computationally even more expensive!
        if match_found_at_iteration == -1 or match_found_at_iteration > MAX_ITERATIONS//2:
            adjusted_end_idx_candidate2, match_found_at_iteration = end_string_match(tokenizer, tokenized_page_text, region_text, tokenized_region_text, adjusted_start_idx, adjusted_end_idx
                                                         , MAX_ITERATIONS=MAX_ITERATIONS, MOVE_BACKWARD_TOKENS_PER_IT=MOVE_FORWARD_TOKENS_PER_IT, matching_chars=int(matching_chars/4), scan_end_range=scan_end_range)

    # choose best option:
    # if either already ends with a "." --> pick this
    if tokenizer.convert_tokens_to_string([tokenized_page_text[adjusted_end_idx_candidate]]) == ".":
        adjusted_end_idx = adjusted_end_idx_candidate
    elif tokenizer.convert_tokens_to_string([tokenized_page_text[adjusted_end_idx_candidate2]]) == ".":
        adjusted_end_idx = adjusted_end_idx_candidate2
    else:
        adjusted_end_idx = max(adjusted_end_idx_candidate, adjusted_end_idx_candidate2) 

    # 3. check for potentially better ending 
    LOOK_AHEAD_TOKENS = int(32 * (1+percent_additional_tokens))
    look_ahead_text = tokenizer.convert_tokens_to_string(tokenized_page_text[adjusted_end_idx+1:adjusted_end_idx + LOOK_AHEAD_TOKENS])
    look_ahead_sentence_split = sentenizer.tokenize_into_sentences(look_ahead_text)
    match_decimal_with_optional_whitespaces = re.search(compiled_match_decimal_with_optional_whitespaces, look_ahead_text) # stop before this new marker
    if adjusted_end_idx < page_token_length:  # Ensure we're not at the end of the page text and not already a "."
        current_end_token_string = tokenizer.convert_tokens_to_string([tokenized_page_text[adjusted_end_idx]])
        additionaly_defined_sentence_end_markers = ["—", '�']
        # do not proceed if already proper ending
        if all(not is_quotation_mark(char) for char in current_end_token_string) and all(char not in current_end_token_string for char in additionaly_defined_sentence_end_markers):
            if match_decimal_with_optional_whitespaces:
                matched_text = match_decimal_with_optional_whitespaces.group(0) # Get the whole matched string including the period
                tokens = tokenizer.tokenize(look_ahead_text)
                reconstructed_text = ""
                for i, token in enumerate(tokens, 1):
                    current_token_str = tokenizer.convert_tokens_to_string([token])
                    reconstructed_text += current_token_str

                    # Check if the reconstructed text ends with the matched enumeration pattern --> if we find a period before that, end here
                    if reconstructed_text.endswith(matched_text) or "." in reconstructed_text:
                        adjusted_end_idx += i
                        break

            # Check if any character in the list is not in the current_end_token_string
            elif len(look_ahead_sentence_split) > 1: # only if proper sentence ending was identified and ending isn'talready a "." or "-" 
                first_sent_end = tokenizer.tokenize(look_ahead_sentence_split[0])
                if len(first_sent_end) <= 32: # only adjust if sentence ending is close enough (for now: 32 tokens)
                    adjusted_end_idx += len(first_sent_end)

    # return final indexes
    return adjusted_start_idx, adjusted_end_idx

# Main function to transform a region in a page of text
def transform_region(tokenizer:AutoTokenizer, doc_id: str, page_index, region_text: str, tokenized_region_text: list[str], page_text: str, tokenized_page_text: list[str], search_span: int) -> (str, list[str], int, int):
    '''
    Pre-processing function enabling to refine the region in a page of text.
    '''
    is_case_sensitive = not getattr(tokenizer, "do_lower_case", False)
    try:
        sentenizer = custom_sentence_tokenizer.Sentenizer("spacy")
        best_match_idx, multiple_hits = find_best_match_index_overlap(tokenizer, tokenized_page_text, tokenized_region_text, threshold=0.2)
        if best_match_idx is None or multiple_hits: # search full_page
                search_span = len(tokenized_page_text)

        start_idx_in_page, end_idx_in_page = adjust_region_indices(tokenizer, sentenizer, best_match_idx, page_text, tokenized_page_text, region_text, span=search_span, is_case_sensitive=is_case_sensitive)

        transformed_region_tokens = tokenized_page_text[start_idx_in_page:end_idx_in_page+1]
        transformed_region_text = tokenizer.convert_tokens_to_string(transformed_region_tokens)
        return transformed_region_text, transformed_region_tokens, start_idx_in_page, end_idx_in_page
    except Exception as e:
        # Fallback
        print(f"WARNING: no refinement conducted for {page_index=} in {doc_id=}. Returning original region as fallback.\nReason: {e}")
        return region_text, tokenized_region_text, -1, -1