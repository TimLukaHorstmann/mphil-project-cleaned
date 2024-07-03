import spacy
from spacy.language import Language
import re
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
from transformers import pipeline

class Sentenizer():
    '''
    Class providing the Sentence BOundary Detection (SBD) functionality for this work (Section 5.2.1)
    Options:
    - spacy: Uses the Spacy library
    - nltk: Uses the NLTK library
    - transformer: Uses the Huggingface Transformers library, specifically MultiLegalSBD model
    '''

    def __init__(self, type:str = "spacy", punkt_model_path=None, device="cuda") -> None:
        self.type = type
        if type == "spacy":
            self.nlp = spacy.load("en_core_web_sm")   # en_core_web_sm en_core_web_trf (for bigger model)    
            self.nlp.add_pipe("set_custom_sentence_beginnings", before='parser')
            
        elif type == "nltk":
            if punkt_model_path:
                with open(punkt_model_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
            else:
                print("Load training data.")
                with open('/home/tlh45/rds/hpc-work/preprocessing/punkt_training_data_aml.txt', 'r', encoding='utf-8') as f: # punkt training data is not included in this repository as it would include confidential data
                    text = f.read()
            
                # Train Punkt tokenizer
                trainer = PunktTrainer()
                trainer.INCLUDE_ALL_COLLOCS = True
                print("Train punkt model.")
                trainer.train(text, verbose=True)
                print("Training successful.")

                # Create tokenizer
                self.tokenizer = PunktSentenceTokenizer(trainer.get_params())
                
                with open('../Data/Model/punkt_tokenizer.pkl', 'wb') as f:
                    pickle.dump(self.tokenizer, f)
        elif type == "transformer":
            self.pipe = pipeline(
            'token-classification',
            model= 'rcds/distilbert-SBD-fr-es-it-en-de-judgements-laws',
            aggregation_strategy="simple",  # none, simple, first, average, max
            device = device
            )
        else:
            raise ValueError("No model chosen.")
            
    def tokenize_into_sentences_with_boundaries(self, text):
        if self.type == "spacy":
            doc = self.nlp(text)
            sentences = []
            sentence_boundaries = []
            for sent in doc.sents:
                trimmed_sentence = sent.text.strip()
                if trimmed_sentence:
                    sentences.append(trimmed_sentence)
                    sentence_boundaries.append((sent.start_char, sent.end_char))
            return sentences, sentence_boundaries

        elif self.type == "nltk":
            sentences = self.tokenizer.tokenize(text)
            sentence_boundaries = []
            start = 0
            for sentence in sentences:
                start = text.find(sentence, start)
                end = start + len(sentence)
                sentence_boundaries.append((start, end))
                start = end
            return sentences, sentence_boundaries

        elif self.type == "transformer":
            results = self.pipe(text)
            sentences = [result['word'] for result in results]
            sentence_boundaries = [(result['start'], result['end']) for result in results] # Extract start and end positions from results
            return sentences, sentence_boundaries

        else:
            raise ValueError("No model chosen.")

    def tokenize_into_sentences(self, text):
        if self.type == "spacy":
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        elif self.type == "nltk":
            return self.tokenizer.tokenize(text)
        elif self.type == "transformer":
            sentences = self.pipe(text)
            output = [sentence['word'] for sentence in sentences]
            return output
        
        else:
            raise ValueError("No model chosen.")
        
        
# also experimented with other regexes, but newline made the most sense
regex_patterns = {
                'newline': re.compile(r'.?\n\s*\n.?'),
            }
# add to spacy pipeline
@Language.component("set_custom_sentence_beginnings")
def set_custom_sentence_beginnings(doc):
    for token in doc[:-1]:
        if any(pattern.match(token.text) for pattern in regex_patterns.values()):
            token.is_sent_start = True
    return doc