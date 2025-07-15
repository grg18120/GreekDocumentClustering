# from GrDocClust.utils import labels_str_to_int
import GrDocClust.config as config
import os
from collections import Counter
import csv
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm 
from sentence_transformers import SentenceTransformer



# ------------------------ HELPFUL FUNCS ------------------------ #
def wrapper(func):
    return func()

def wrapper_args(func, args): 
    """
    Pass a list of arguments(args) into function(func) 
    """
    return func(*args)

def clean_corpus(corpus, labels_true):
    """
     Remove multiple spaces
     Remove empty documents
    """
    docs, doc_indx = zip(*[(' '.join(text.split()), index) for index, text in enumerate(corpus) if len(" ".join(text.split())) > 0])
    return list(docs), [labels_true[x] for x in doc_indx]


# ------------------------ Load Greek Datasets from "/Datasets/"" Folder ------------------------ #
def labels_str_to_int(labels_str):
    unique_labels_vals = set(labels_str)
    return [ list(unique_labels_vals).index(lab_str) for lab_str in labels_str], len(unique_labels_vals)

def load_dataset_test():
    corpus = [
        "",
        "η επιστήμη δεδομένων είναι ένας από τους σημαντικότερους κλάδους της επιστήμης",
        "το Game of Thrones είναι μια καταπληκτική τηλεοπτική σειρά!",
        "αυτό είναι ένα από τα καλύτερα μαθήματα επιστήμης δεδομένων",
        "οι επιστήμονες δεδομένων αναλύουν δεδομένα",
        "το Game of Thrones είναι η καλύτερη τηλεοπτική σειρά!",
        "Το αυτοκίνητο οδηγείται στον δρόμο",
        "το Game of Thrones είναι τόσο υπέροχο.",
        "Το φορτηγό οδηγείται στον αυτοκινητόδρομο",
        "      "
    ]
    labels_true = [2, 0, 2, 0, 0, 2, 1, 2, 1, 1]
    n_clusters = len(set(labels_true))
    return [corpus, labels_true, n_clusters]

def load_dataset_pubmed4000_greek():
    pubmed4000_path = "".join([config.local_datasets_path, "pubmed4000_greek\\"])
    pubmed4000_files = os.listdir(pubmed4000_path)
    corpus, labels_true_str = zip(*[(open(pubmed4000_path + file_name, 'r', encoding="utf8").read(), "".join([char for char in file_name if not char.isdigit()]).split(".")[0] ) for file_name in pubmed4000_files])
    labels_true, n_clusters = labels_str_to_int(labels_true_str)
    return [list(corpus), labels_true, n_clusters]

def load_dataset_greek_reddit():
    greek_reddit_path = "".join([config.local_datasets_path, "greek_reddit\\"])
    greek_reddit_csv_file = "greek_reddit_test.csv"
    corpus, labels_true_str = zip(*[(csv_row[2], csv_row[4]) for csv_row in csv.reader(open(greek_reddit_path + greek_reddit_csv_file, 'r', encoding='utf-8'))])
    labels_true, n_clusters = labels_str_to_int(list(labels_true_str)[1:])
    return [list(corpus)[1:], list(labels_true), len(set(labels_true))]

def load_dataset_greeksum():
    """
    GreeSum dataset (valid + test splits)
    """
    greeksum_path = "".join([config.local_datasets_path, "greeksum_test_valid\\"])
    greeksum_csv_file = "greeksum_test_valid.csv"
    corpus, labels_true = zip(*[(csv_row[0], int(csv_row[1])) for csv_row in csv.reader(open(greeksum_path + greeksum_csv_file, 'r', encoding='utf-8')) if csv_row[1].isdigit()])
    return [list(corpus), list(labels_true), len(set(labels_true))]


# ------------------------ NLP Models------------------------ #
def load_models():
    """
    Function which loads pre-trained NLP models.
    This needs to run once since all models need a few seconds to load.
    """

    spacy_model_gr = None
    bert_model_gr = None
    st_greek_media_model = None
    jina_v3_model = None
    sent_transformers_paraph_multi_model_gr = None
    xlm_roberta_model_gr = None

    spacy_model_gr = spacy.load('el_core_news_lg')
    bert_model_gr = SentenceTransformer(
        model_name_or_path = 'nlpaueb/bert-base-greek-uncased-v1',
        cache_folder = config.local_models_storage_path,
        device = 'cpu'
    )
    st_greek_media_model = SentenceTransformer(
        'dimitriz/st-greek-media-bert-base-uncased',
        cache_folder = config.local_models_storage_path,
        device = 'cpu'
    )
    jina_v3_model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        cache_folder = config.local_models_storage_path,
        device = 'cpu',
        trust_remote_code=True
    )
    sent_transformers_paraph_multi_model_gr =  SentenceTransformer(
        model_name_or_path = 'paraphrase-multilingual-mpnet-base-v2',
        cache_folder = config.local_models_storage_path,
        device = 'cpu'
    )
    xlm_roberta_model_gr =  SentenceTransformer(
        model_name_or_path = 'lighteternal/stsb-xlm-r-greek-transfer',
        cache_folder = config.local_models_storage_path,
        device = 'cpu'
    )

    return (
        spacy_model_gr, 
        bert_model_gr,
        st_greek_media_model,
        jina_v3_model,
        sent_transformers_paraph_multi_model_gr,
        xlm_roberta_model_gr,
    )


# ------------------------ EMBEDDINGS - WORD VECTORS ------------------------ #
def tfidf(corpus, labels_true):

    vectorizer = TfidfVectorizer(
        lowercase = True,
        use_idf = True,
        norm = None,
        stop_words = list(config.greek_stop_words),
        max_df = 0.99,
        min_df = 0.01
        #max_features = 4#5250
    )
    vectorizer_fitted = vectorizer.fit_transform(tqdm(corpus))
    #feature_names = vectorizer.get_feature_names_out()

    doc_vectors = []
    doc_indx = []
    for index, doc_vector in enumerate(vectorizer_fitted.todense()):

        # remove nan value & zero elements vectors
        if (np.any(doc_vector) and isinstance(doc_vector,np.matrix)):
            doc_vectors.append(np.squeeze(np.asarray(doc_vector)))
            doc_indx.append(index)

    return np.array(doc_vectors, dtype = object), [labels_true[x] for x in doc_indx]

def spacy_model_embeddings(corpus, spacy_model, labels_true):

    def spacy_useful_token(token):
        """
        Keep useful tokens which have 
        - Part Of Speech tag (POS): ['NOUN','PROPN','ADJ']
        - Alpha(token is word): True
        - Stop words(is, the, at, ...): False
        """
        return token.pos_ in ['NOUN','PROPN','ADJ'] and token.is_alpha and not token.is_stop and token.has_vector 

    """
    Spacy embeddings
    """
    doc_vectors = []
    doc_indx = []
    for index, text in enumerate(tqdm(corpus)):
        doc = spacy_model(text)
        if doc.has_vector:
    
            vector_list = [token.vector for token in doc if spacy_useful_token(token)]
            # vector_list to np array
            doc_vector = np.mean(vector_list, axis = 0)

            # remove nan value & zero elements vectors
            if np.any(doc_vector) and isinstance(doc_vector,np.ndarray):
                doc_vectors.append(doc_vector)
                doc_indx.append(index)

    return np.array(doc_vectors, dtype = object), [labels_true[x] for x in doc_indx]

def sent_transformers_model_embeddings(corpus, spacy_model, sent_transorfmers_model, labels_true):
    doc_vectors = []
    doc_indx = []
    for index, text in enumerate(tqdm(corpus)):

        # Take sentences(Span objects) from spacy
        doc = spacy_model(text)
        sents_spacy_span = [sent for sent in doc.sents]

        # Cut sentence in the middle if len(tokens of sentence) < transf_model.max_seq_length
        sents_spacy_str = []
        for sent in sents_spacy_span:
            if len([token for token in sent]) > sent_transorfmers_model.max_seq_length :
                middle = int(len(sent.text)/2)
                sents_spacy_str.append(sent.text[:middle])
                sents_spacy_str.append(sent.text[middle:])
            else:
                sents_spacy_str.append(sent.text)

        # Mean of sentenses vectors to create doc vector
        sent_vectors =  sent_transorfmers_model.encode(sents_spacy_str)
        doc_vector = np.mean(sent_vectors, axis = 0)

        # remove nan value & zero elements vectors
        if (np.any(doc_vector) and isinstance(doc_vector,np.ndarray)):
            doc_vectors.append(doc_vector)
            doc_indx.append(index)

    return np.array(doc_vectors, dtype = object), [labels_true[x] for x in doc_indx]