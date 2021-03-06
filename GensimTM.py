from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim import corpora
import spacy
import re
from gensim.utils import simple_preprocess

from spacy.lang.en.stop_words import STOP_WORDS

# Load English tokenizer, tagger and word vectors
nlp = nlp = spacy.load('en', disable=['parser', 'ner'])


# Remove Stopwords from notes
def remove_stopwords(notes):
    return [[word.lower() for word in simple_preprocess(str(note)) if word not in STOP_WORDS] for note in notes]


# Return the list of cleaned sentences in the document
def regex_cleaner(note):
    # Removing more than one blank spaces
    cleaned_data = re.sub(r'\s+', ' ', note)
    # Removing links from the data
    cleaned_data = re.sub(r'http\S+', '', cleaned_data)
    # Remove out special characters
    cleaned_data = re.sub(r'[^A-Za-z0-9]+', '', cleaned_data)
    return cleaned_data


# Lemmatizes the words in the text
def lemmatize_doc(doc, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    doc_out = []
    for sent in doc:
        sent_doc = nlp(" ".join(sent))
        doc_out.append([token.lemma_ for token in sent_doc if token.pos_ in allowed_postags])
    return doc_out


# Build base LDA model and LDA Mallet Model
def build_dictionary_corpus(notes):
    # Notes preprocessing
    no_stop_notes = remove_stopwords(notes)
    cleaned_notes = map(regex_cleaner, no_stop_notes)
    lemmatize_notes = lemmatize_doc(cleaned_notes, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Build dictionary
    id2word = corpora.Dictionary(lemmatize_notes)
    # Save dictionary for future reference
    id2word.save('notes_token.dict')
    # Build term frequency corpus
    corpus = [id2word.doc2bow(note) for note in lemmatize_notes]
    # Save corpus for future reference
    corpora.MmCorpus.serialize('notes_mmformat.mm', corpus)
    return lemmatize_notes, id2word, corpus


# Build model on topics
def build_model(dictionary, corpus, n_topics, lemmatized_notes):
    # Build LDA model
    coh_val_lda = []
    coh_val_lda_mallet = []
    model_lda = []
    model_mallet = []
    for topic in n_topics:
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic, random_state=100, update_every=1,
                             chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        coh_lda_model = CoherenceModel(model=model_lda, texts=lemmatized_notes, dictionary=dictionary, coherence='c_v')
        coh_val_lda.append(coh_lda_model.get_coherence())
        model_lda.append(lda_model)
        # Build LDA Mallet model
        mallet_path = 'mallet/bin/mallet'
        lda_mallet = LdaMallet(mallet_path, corpus=corpus, num_topics=n_topics, id2word=dictionary)
        coh_lda_model = CoherenceModel(model=lda_mallet, texts=lemmatized_notes, dictionary=dictionary, coherence='c_v')
        model_mallet.append(lda_mallet)
        coh_val_lda_mallet.append(coh_lda_model.get_coherence())
    return model_mallet, coh_val_lda_mallet, model_lda, coh_val_lda


# Get the dominant topic in each note
def format_topics_sentences(model, corpus, notes):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']


# Select optimal number of topics
def select_optimal_model_topics(notes):
    strt = 10
    end = 40
    lemmatized_notes, dict, corpus, = build_dictionary_corpus(notes)
    model_mallet, coh_val_lda_mallet, model_lda, coh_val_lda = build_model(dict, corpus, strt, end, lemmatized_notes)
    max_coh_idx_lda_mal = coh_val_lda_mallet.index(max(coh_val_lda_mallet))
    final_model = model_mallet[max_coh_idx_lda_mal]
    final_topic_data_df = format_topics_sentences(final_model, corpus, notes)
    return final_topic_data_df


# Driver method to get topics out of the notes and returns a dataframe with topic
def get_topics_from_notes(notes):
    final_df = select_optimal_model_topics(notes)
    return final_df
