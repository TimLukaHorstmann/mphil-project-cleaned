# Script to handle detailed labels in the dataset
# Provides several helper functions to encode, decode and analyze detailed labels

from collections import defaultdict, Counter


def encode_detailed_labels(detailed_labels:list , mapping_dicts, NUMBER_OF_LEVELS):
    '''
    Encodes a list of detailed labels in a list of their numerical counterparts
    '''
    encoded_labels = []
    for label in detailed_labels:
        parts = label.split('-')  # Assuming '-' separates levels
        encoded_label = [mapping_dicts[f'level_{i}'].get(parts[i], -1) if len(parts) > i else -1 for i in range(NUMBER_OF_LEVELS)]
        encoded_labels.append(encoded_label)
    return encoded_labels



def decode_detailed_labels(encoded_labels, mapping_dicts, HIER_LABELS_LEVELS):
    '''
    Decodes a list of numerical detailed labels into their string counterparts
    '''
    if not HIER_LABELS_LEVELS:
        return []
    decoded_labels = []
    for label in encoded_labels:
        decoded_label = []
        #print(label)
        for i, level in enumerate(HIER_LABELS_LEVELS):  # Iterate only over specified levels
            level_mapping = mapping_dicts.get(f'level_{level}')
            if level_mapping and isinstance(label, list) and label[i] != -1:
                decoded_label.append(list(level_mapping.keys())[list(level_mapping.values()).index(label[i])])
            else:
                decoded_label.append(None)
        decoded_labels.append('-'.join(filter(None, decoded_label)))  # Joining non-None labels
    return decoded_labels

def extract_and_analyze_tags(data_dict, HIER_LABELS_LEVELS):
    '''
    Extracts tags from the dataset and analyzes them to provide mappings for hierarchical labels
    '''
    level_tags = defaultdict(Counter)
    full_tags = Counter()
    
    def process_tag(tag):
        '''
        Helper function to process a single tag 
        '''
        parts = tag.split('-')
        for i, part in enumerate(parts):
            if HIER_LABELS_LEVELS and i in HIER_LABELS_LEVELS:  # Process only specified levels
                level_tags[i][part] += 1  # Counting unique labels for each level
        full_tags[tag] += 1

    for doc_id, doc_data in data_dict.items():
        for page_index, page_data in doc_data.items():
            if 'refined_regions' in page_data:
                for region in page_data['refined_regions']:
                    if 'tags' in region:
                        for tag in region['tags']:
                            process_tag(tag)
    
    mapping_dicts = {}
    if HIER_LABELS_LEVELS:
        for level, tags_counter in level_tags.items():
            if level in HIER_LABELS_LEVELS:  # Create mappings only for specified levels
                hierarchical_mapping = {tag: idx for idx, (tag, _) in enumerate(tags_counter.items())}
                mapping_dicts[f'level_{level}'] = hierarchical_mapping
    
    return {f'level_{i}': tags for i, tags in level_tags.items() if i in HIER_LABELS_LEVELS}, mapping_dicts