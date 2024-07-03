from collections import defaultdict
import pickle
import multiprocessing
# from tqdm.contrib.concurrent import process_map
from tqdm.notebook import tqdm
import os
from lxml import etree as ET
from transformers import AutoTokenizer
import sys
import time
# setting path
sys.path.append('../SnippetAnalytics')
# from Snippet_Code import node_utils # here, we would need to import RegGenome's snippeting algorithm, for confidentiality reasons, we cannot provide this code
import find_region_in_page
import logging

logging.basicConfig(filename='processing.log', level=logging.DEBUG)

# data_dict structure we aim for:
""" data_dict = {
    doc_id: {
        title: "",
        doc_long_id: "",
        page_index: {
            'blocks': [
                {'text': block_text, 'tokenized_text': [tokenized_text], 'tags': [tag1, tag2, ...]},
                ...
            ],
            'regions': [
                {'x': x, 'y': y, 'width': width, 'height': height, 'text': region_text, 'tokenized_text': [tokenized_text], 'tags': [tag1, tag2, ...]},
                ...
            ],
            'refined_regions': [
                {'text': transformed_region_text, 'tokenized_text': tuple(transformed_region_tokens), 'tags': region['tags'], 'start_idx_in_page': start_idx_in_page, 'end_idx_in_page': end_idx_in_page, 'refine_regions_duration': duration}
            ],
            'snippets': [
                {'text': snippet_text, 'tokenized_text': snippet_tokenized_text, 'snippet_character_type': snippet_character_type, 'offset': snippet['offset'], 'limit': snippet['limit'], 'snippet_index': snippet['snippet_index']},
                ...
            ],
            'predicted_snippets': [
                { ... }
            ]
            full_text: "",
            tokenized_full_text: []
        },
        ...
    },
    ...
} """

############################################################################

class DocReader:
    def __init__(self, model:str = None, main_tokenizer:AutoTokenizer = None, refine_regions_tokenizer:AutoTokenizer = None) -> None:
        # Note: without MODEL/tokenizer --> no tokenization
        if not model and not main_tokenizer:
            print("NOTE: No tokenization will be conducted due to missing MODEL and/or tokenizer arguments.")

        self.model = model
        self.main_tokenizer = main_tokenizer
        self.refine_regions_tokenizer = refine_regions_tokenizer if refine_regions_tokenizer is not None else AutoTokenizer.from_pretrained("roberta-base") # use robertas wordpiece as default (keeps formatting, subtoken, ...)
        self.checkpoint_folder = None
        self.refine_regions = True
        self.create_original_snippets = False
        self.extract_title = False
        self.extract_doc_long_id = False

        # Check if same tokenizers are used (i.e. for main content and refine regions tokenization)
        main_config = self.main_tokenizer.config if hasattr(self.main_tokenizer, 'config') else {} 
        refine_config = self.refine_regions_tokenizer.config if hasattr(self.refine_regions_tokenizer, 'config') else {}
        if main_config.get('model_identifier') == refine_config.get('model_identifier'):
            self.same_tokenizers = True
        else:
            self.same_tokenizers = False

    def create_page_text_from_blocks(self, blocks):
        '''
        We obtain the full page text by concatenating all blocks
        '''
        full_page_text = ' '.join([block['text'] for block in blocks])
        return full_page_text

    def default_page(self):
        # define base page
        page = {
            'blocks': [],
            'regions': [],
            'predicted_snippets': [],
            'full_text': "",
            'tokenized_full_text': []
        }

        # optional additions
        if self.create_original_snippets:
            page['snippets'] = []
        if self.refine_regions:
            page['refined_regions'] = []

        return page

    def get_defaultdict_of_defaultpage(self):
        return defaultdict(self.default_page)

    def process_xml_file(self, file_path:str):
        local_data_dict = defaultdict(self.get_defaultdict_of_defaultpage)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            tokenize = self.model and self.main_tokenizer

            doc_attributes = {attr: root.attrib.get(attr) for attr in ['title', 'annotation', 'id']}
            title = doc_attributes['title']
            doc_id = doc_attributes['annotation']
            doc_long_id = doc_attributes['id']
        
            if not doc_id:
                return
            
            local_data_dict[doc_id] # necessary to even collect docs without pages!
            if self.extract_title:
                local_data_dict[doc_id]['title'] = title
            if self.extract_doc_long_id:
                local_data_dict[doc_id]['doc_long_id'] = doc_long_id

            # Prepare data collection structures
            texts_for_tokenization = []
            data_collection = {}

            # Iterate over pages to collect texts and additional data
            for page in root.findall('.//page'):
                page_index = page.attrib['index']
                page_data = self.default_page()

                # BLOCKS
                for block in page.findall('.//block'):
                    block_text = block.findtext('./text', default='')
                    block_tags = [tag.text for tag in block.findall('.//tag')]
                    texts_for_tokenization.append(block_text)
                    page_data['blocks'].append({'text': block_text, 'tags': block_tags})

                # Full page
                full_page_text = self.create_page_text_from_blocks(page_data['blocks'])
                texts_for_tokenization.append(full_page_text)
                page_data['full_text'] = full_page_text
                
                # REGIONS
                for region in page.findall('.//region'):
                    region_attrs = {attr: region.attrib.get(attr) for attr in ['id', 'x', 'y', 'width', 'height']}
                    region_text = region.findtext('./text', default='')
                    if region_text == '': # empty region: skip
                        continue
                    region_tags = [tag.text for tag in region.findall('.//tag')]
                    texts_for_tokenization.append(region_text)
                    region_attrs.update({'text': region_text, 'tags': region_tags})
                    page_data['regions'].append(region_attrs)

                # SNIPPETS
                if self.create_original_snippets:
                    snippets, snippet_character_type = # here, we would need to import RegGenome's snippeting algorithm, for confidentiality reasons, we cannot provide this code
                    for snippet in snippets:
                        snippet_text = snippet['text']
                        texts_for_tokenization.append(snippet_text)
                        page_data['snippets'].append({'text': snippet_text, 'snippet_character_type': snippet_character_type, 'offset': snippet['offset'], 'limit': snippet['limit'], 'snippet_index': snippet['snippet_index']})

                # add page to dict
                data_collection[page_index] = page_data

            # Tokenize texts in batch
            if tokenize and texts_for_tokenization:
                # Tokenize texts in batch
                encoded_inputs = self.main_tokenizer(texts_for_tokenization, add_special_tokens=False)

                # Initialize empty list to hold tokenized texts
                tokenized_texts = []
                for i in range(len(texts_for_tokenization)):
                    tokens = encoded_inputs.tokens(i) 
                    tokenized_texts.append(tuple(tokens))

            # Process tokenized texts and additional data --> ORDER IS CRUCIAL AND MUST MATCH THE ABOVE
            tokenized_text_iter = iter(tokenized_texts)
            for page_index, page_data in data_collection.items():
                for block in page_data['blocks']:
                    block['tokenized_text'] = next(tokenized_text_iter)

                page_data['tokenized_full_text'] = next(tokenized_text_iter)

                for region in page_data['regions']:
                    region['tokenized_text'] = next(tokenized_text_iter)
                    # add refined regions
                    if self.refine_regions and not tokenize:
                        print("ERROR: Cannot refine regions without tokenization. Please provide a model and tokenizer.")
                        raise
                    if self.refine_regions and self.model and self.refine_regions_tokenizer:
                        search_span = 60 if self.model == "roberta-base" else 30
                        start_time = time.time()  # Start timing
                        # Pre-process region text here!
                        transformed_region_text, transformed_region_tokens, start_idx_in_page, end_idx_in_page = find_region_in_page.transform_region(tokenizer=self.refine_regions_tokenizer
                                                             , doc_id=doc_id
                                                             , page_index=page_index
                                                             , region_text=region['text']
                                                             , tokenized_region_text=region['tokenized_text']
                                                             , page_text = page_data['full_text']
                                                             , tokenized_page_text=page_data['tokenized_full_text']
                                                             , search_span=search_span)
                        end_time = time.time()  # End timing
                        duration = end_time - start_time
                        if not self.same_tokenizers:
                           transformed_region_tokens = self.main_tokenizer.tokenize(transformed_region_text)
                        page_data['refined_regions'].append({'text': transformed_region_text, 'tokenized_text': tuple(transformed_region_tokens), 'tags': region['tags'], 'start_idx_in_page': start_idx_in_page, 'end_idx_in_page': end_idx_in_page, 'refine_regions_duration': duration})

                for snippet in page_data['snippets']:
                    snippet['tokenized_text'] = next(tokenized_text_iter)
                
                # conversion to tuples for better memory efficiency (--> not for predicted snippets! Kept as placeholder list)
                page_data['blocks'] = tuple(page_data['blocks'])
                page_data['regions'] = tuple(page_data['regions'])
                if self.create_original_snippets:
                    page_data['snippets'] = tuple(page_data['snippets'])
                if self.refine_regions:
                    page_data['refined_regions'] = tuple(page_data['refined_regions'])

                # clear in case it shall not be saved
                if not self.add_full_page:
                    page_data['full_text'] = ""
                    page_data['tokenized_full_text'] = []
                # add all to dict
                local_data_dict[doc_id][page_index] = page_data

            # Save processed data
            if doc_id:
                data_filename = os.path.join(self.checkpoint_folder, f"data_{doc_id}.pkl")
                with open(data_filename, 'wb') as file:
                    pickle.dump(local_data_dict[doc_id], file)

            logging.info(f"Successfully processed {file_path}")

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")


    def safe_load_pickle(self, file_path):
        if os.path.getsize(file_path) > 0:  # File is not empty
            with open(file_path, 'rb') as file:
                try:
                    return pickle.load(file)
                except EOFError:
                    logging.error(f"EOFError: Fil e incomplete or corrupted?! {file_path}")
                    return None
        else:
            logging.error(f"File is empty: {file_path}")
            return None

    def preprocess_folder(self, preprocess:bool, folder_path:str, num_workers = 6, data_size=1149, chunksize=1
                          , extract_title=False, extract_doc_long_id=False, refine_regions=True, create_original_snippets=False, add_full_page=False
                          , data_dict_folder='./data_dicts', file_name_additional_suffix=""):
        
        self.refine_regions = refine_regions
        self.create_original_snippets = create_original_snippets
        self.add_full_page=add_full_page
        self.extract_title = extract_title
        self.extract_doc_long_id = extract_doc_long_id

        
        self.checkpoint_folder = os.path.join(data_dict_folder, f'checkpoint{file_name_additional_suffix}')
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
            print(f"Creating checkpoint folder at {self.checkpoint_folder}")
        else:
            print(f"Checking checkpoint folder at {self.checkpoint_folder} for preprocessed files.")
        processed_files_set = {filename.split('_')[1].split('.')[0] for filename in os.listdir(self.checkpoint_folder) if filename.endswith('.pkl')}
        
        # Determine files to be processed, respecting the data_size limit
        all_file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xml')]

        # preprocess?
        if preprocess:
            file_paths_to_process = [path for path in all_file_paths if os.path.splitext(os.path.basename(path))[0] not in processed_files_set]
            remaining_files_count = data_size - len(processed_files_set)
            file_paths_to_process = file_paths_to_process[:max(0,remaining_files_count)]

            if not file_paths_to_process:
                print("No additional documents need processing or data size limit already met.")
            else:
                print(f"Processing {len(file_paths_to_process)} documents out of {min(len(all_file_paths), data_size)} ({len(processed_files_set)} already processed).")
                
                with multiprocessing.Pool(num_workers) as pool:
                    list(tqdm(pool.imap_unordered(self.process_xml_file, file_paths_to_process, chunksize=chunksize), initial=len(processed_files_set), total=data_size, unit="documents"))

            print("Finished preprocessing.")
            
        print(f"Create data_dict from checkpoint folder {self.checkpoint_folder}.")
        all_doc_ids = {os.path.splitext(os.path.basename(path))[0] for path in all_file_paths}

        # After processing, check processed files and create final data_dict if needed
        data_dict = defaultdict(self.get_defaultdict_of_defaultpage)
        processed_files_count = 0
        filenames = [filename for filename in os.listdir(self.checkpoint_folder) if filename.endswith('.pkl')]
        
        for filename in tqdm(filenames, total=len(filenames)):
            if processed_files_count >= data_size:
                break

            if filename.endswith('.pkl'):
                doc_data = self.safe_load_pickle(os.path.join(self.checkpoint_folder, filename))
                if doc_data:
                    doc_id = filename.split('_')[1].split('.')[0]
                    if doc_id in all_doc_ids:
                        limited_doc_data = {}
                        for key, value in doc_data.items():
                            if (key == 'title' and not self.extract_title) or \
                            (key == 'doc_long_id' and not self.extract_doc_long_id):
                                continue
                            elif not self.add_full_page and (key == 'full_text' or key == "tokenized_full_text"):
                                if key == 'full_text':
                                    limited_doc_data[key] = ""
                                else:
                                    limited_doc_data[key] = []
                            else:
                                limited_doc_data[key] = value
                        data_dict[doc_id] = limited_doc_data
                        processed_files_count += 1
                else:
                    for try_count in range(1, 6):  # Retry up to 5 times
                        print(f"Attempt {try_count}: Retrying file {filename}")
                        os.remove(os.path.join(self.checkpoint_folder, filename)) 
                        
                        original_xml_path = os.path.join(folder_path, filename.split(".")[0]+".xml")
                        print("Re-Process", original_xml_path)
                        self.process_xml_file(original_xml_path)  # Re-process from the original XML
                        
                        # Attempt to reload
                        doc_data = self.safe_load_pickle(os.path.join(self.checkpoint_folder, filename))
                        if doc_data:
                            doc_id = filename.split('_')[1].split('.')[0]
                            if doc_id in all_doc_ids:  # Ensure consistency
                                limited_doc_data = {}
                                for key, value in doc_data.items():
                                    if key == 'title' and not self.extract_title or key == 'doc_long_id' and not self.extract_doc_long_id:
                                        continue
                                    else:
                                        limited_doc_data[key] = value
                                data_dict[doc_id] = limited_doc_data
                                processed_files_count += 1
                                break
                        elif try_count == 5:
                            print("Maximum retry attempts reached. Skipping file.")

        print(f"Final Data Dict successfully created with {processed_files_count} entries.")
        return data_dict