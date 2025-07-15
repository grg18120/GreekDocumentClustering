import GrDocClust.experiments as experiments
import GrDocClust.utils as utils
import GrDocClust.config as config



if config.ROOT_DIR == 'PATH_TO\\GreekDocumentClustering\\':
    print("Before running any code, you **MUST** set the `ROOT_DIR` variable in: `GreekDocumentClustering/GrDocClust/config.py`")
    exit()

# Create directories if they doesnt exist to store vectors-embedding 
experiments.create_serialized_datasets_vectors_dirs()

# Load NLP Models
(
    spacy_model_gr, 
    bert_model_gr,
    st_greek_media_model,
    jina_v3_model,
    sent_transformers_paraph_multi_model_gr,
    xlm_roberta_model_gr
) = utils.load_models()

# Main Loop
for dataset_string in config.datasets_strings:
    [corpus, labels_true, n_clusters]  = utils.wrapper(config.datasets_pointers().get(dataset_string))
    print("Corpus Size before clean: ", len(corpus))
    corpus, labels_true = utils.clean_corpus(corpus, labels_true)

    if config.plot_true_labels_dataset_distribution:
        experiments.save_plot_dataset_histogram(labels_true, dataset_string)
    labels_true_corpus = labels_true[:]
    print("Corpus Size After clean: ", len(corpus))

    for vectorizer_string in config.vectorizers_strings:
        if (vectorizer_string == "tfidf"): 
            if (experiments.folder_is_empty(dataset_string, vectorizer_string)):
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
            else:   
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)

        if (vectorizer_string == "greek_spacy_model_embeddings"): 
            if (experiments.folder_is_empty(dataset_string, vectorizer_string)):
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
            else:   
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)

        if (vectorizer_string == "greek_bert_model_embeddings"):
            if (experiments.folder_is_empty(dataset_string, vectorizer_string)):
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [bert_model_gr] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
            else:   
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)

        if (vectorizer_string == "st_greek_media_model_embeddings"): 
            if (experiments.folder_is_empty(dataset_string, vectorizer_string)):
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [st_greek_media_model] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
            else:
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)

        if (vectorizer_string == "jina_v3_model_embeddings"): 
            if (experiments.folder_is_empty(dataset_string, vectorizer_string)):
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [jina_v3_model] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
            else:
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)

        if (vectorizer_string == "sent_transformers_paraph_multi_model_embeddings"): 
            if (experiments.folder_is_empty(dataset_string, vectorizer_string)):
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [sent_transformers_paraph_multi_model_gr] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
            else:
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)

        if (vectorizer_string == "greek_xlm_roberta_model_embeddings"): 
            if (experiments.folder_is_empty(dataset_string, vectorizer_string)):
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [xlm_roberta_model_gr] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
            else:
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)

        all_eval_metric_values = []
        for clustering_algorithms_string in config.clustering_algorithms_strings:       
            arguments_list = config.clustering_algorithms_arguments(n_clusters).get(clustering_algorithms_string)
            for arguments in arguments_list:
                labels_pred = (utils.wrapper_args(config.clustering_algorithms_pointers().get(clustering_algorithms_string), [X] + [labels_true] + arguments )).tolist()
                if config.plot_vectors2D_predict_labels:
                    experiments.save_plot_vectors2D_labels(X, n_clusters, dataset_string, vectorizer_string, labels_true, labels_pred)
                for evaluation_metric_string in config.evaluation_metrics_strings:
                    score  = utils.wrapper_args(config.evaluation_metrics_pointers().get(evaluation_metric_string),[list(labels_true), list(labels_pred)])
                    all_eval_metric_values.append(score)
                    print(f"{evaluation_metric_string} = {score}")
                    experiments.save_results_csv(dataset_string, vectorizer_string, n_clusters, all_eval_metric_values) 