from gensim import corpora
import spacy
import re

from spacy.lang.en.stop_words import STOP_WORDS

''' Return the list of cleaned sentences in the document'''
def regex_cleaner(doc):
    sentences = doc.split('.')
    # Removing more than one blank spaces
    cleaned_data = [re.sub(r'\s+', ' ', sent) for sent in sentences]
    # Removing links from the data
    cleaned_data = [re.sub(r'http\S+', '', sent) for sent in cleaned_data]
    # Remove out special characters
    cleaned_data = [re.sub(r'[^A-Za-z0-9]+', '', sent) for sent in cleaned_data]
    return cleaned_data

def lemmatize_doc(doc, allowed_postags=['NOUN','ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc_out = []
    for sent in doc:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def process_notes(notes):
    for note in notes:


