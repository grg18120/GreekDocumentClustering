import GrDocClust.config as config
import pickle
import os
import numpy as np
import pandas as pd
from functools import reduce
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px

def create_serialized_datasets_vectors_dirs():
    """
    Create folder for each dataset for each vectorize approach
    to store pickle files for each document
    """
    path = config.local_precomputed_vectors_path
    if not os.path.exists(path):
        os.makedirs(path)

    for datasets_folder in config.datasets_strings:
        if not os.path.exists("".join([path,datasets_folder,"\\"])):
            os.makedirs("".join([path,datasets_folder,"\\"])) 
    
    for datasets_folder in config.datasets_strings:
        for vectorizer_folder in config.vectorizers_strings:
            if not os.path.exists("".join([path,datasets_folder,"\\",vectorizer_folder])):
                os.makedirs("".join([path,datasets_folder,"\\",vectorizer_folder]))

def folder_is_empty(dataset_string, vectorizer_string):
    path = config.local_precomputed_vectors_path
    path = "".join([path,dataset_string,"\\",vectorizer_string,"\\"])
    return os.path.getsize(path) < 1000

def store_serialized_vector(dataset_string, vectorizer_string, vectors, labels_true):
    file_path = f"{config.local_precomputed_vectors_path}{dataset_string}\\{vectorizer_string}\\labels_true.pkl"
    dbfile = open(file_path, "ab")
    pickle.dump(labels_true, dbfile)                     
    dbfile.close()

    file_path = f"{config.local_precomputed_vectors_path}{dataset_string}\\{vectorizer_string}\\shape.pkl"
    dbfile = open(file_path, "ab")
    pickle.dump(vectors.shape, dbfile)                     
    dbfile.close()

    for indx, vector in enumerate(vectors):
        file_path = f"{config.local_precomputed_vectors_path}{dataset_string}\\{vectorizer_string}\\{indx}.pkl"
        dbfile = open(file_path, "ab")
        pickle.dump(vector, dbfile)                     
        dbfile.close()

def load_deselialized_vector(dataset_string, vectorizer_string):
    file_path = f"{config.local_precomputed_vectors_path}{dataset_string}\\{vectorizer_string}\\shape.pkl"
    dbfile = open(file_path, 'rb')     
    shape = pickle.load(dbfile)
    dbfile.close()

    file_path = f"{config.local_precomputed_vectors_path}{dataset_string}\\{vectorizer_string}\\labels_true.pkl"
    dbfile = open(file_path, 'rb')     
    labels_true = pickle.load(dbfile)
    dbfile.close()

    arr = np.array([])
    for indx in range(shape[0]):
        file_path = f"{config.local_precomputed_vectors_path}{dataset_string}\\{vectorizer_string}\\{indx}.pkl"
        dbfile = open(file_path, "rb")
        vector = pickle.load(dbfile)     
        arr = np.append(arr, vector)            
        dbfile.close()

    arr = arr.reshape(shape)
    return arr, labels_true

def save_plot_dataset_histogram(x_list, dataset_string):
    value_counts = Counter(x_list)
    values = list(value_counts.keys())
    counts = list(value_counts.values())

    values_sorted, counts_sorted = zip(*sorted(zip(values, counts), key=lambda x: x[0]))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(values_sorted, counts_sorted, width=0.5)
    ax.set_xticks(values_sorted)
    ax.set_xticklabels(values_sorted, rotation=45)
    ax.set_ylabel('Count')
    ax.set_xlabel('True Labels')
    ax.set_title(f"True Labels Distribution for <<{dataset_string.upper()}>>")
    ax.grid(True)

    for x, y in zip(values_sorted, counts_sorted):
        ax.text(x, y, str(y), ha='center', va='bottom')

    # Save figure
    output_path = os.path.join(config.figures_dir, f'True Labels Distribution {dataset_string}.png')
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def save_plot_vectors2D_labels(X, n_clusters, dataset_string, vectorizer_string, labels_true, labels_predict):
    # Render and save the TSNE 2D plot for the current k value.
    if len(X) < 20 :
        perplexity = len(X)/2
    else:
        perplexity = 20

    if config.plot_vectors2D_predict_labels:
        tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = config.random_state) 
        tsne_embeddings = tsne.fit_transform(X)
        fig = px.scatter(x = tsne_embeddings[:, 0], y = tsne_embeddings[:, 1], color = list(map(str, labels_predict)))
        fig.update_layout(
            title = 't-SNE 2D visualization',
            xaxis_title = 'X-dimension',
            yaxis_title = 'Y-dimension'
        )
        fig.write_image(os.path.join(config.figures_dir, f'tsne_Predict_Labels_{dataset_string}_{vectorizer_string}_{n_clusters}.png'))

    if config.plot_vectors2D_true_labels:
        tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = config.random_state) 
        tsne_embeddings = tsne.fit_transform(X)
        fig = px.scatter(x = tsne_embeddings[:, 0], y = tsne_embeddings[:, 1], color = list(map(str, labels_true)))
        fig.update_layout(
            title = 't-SNE 2D visualization',
            xaxis_title = 'X-dimension',
            yaxis_title = 'Y-dimension'
        )
        fig.write_image(os.path.join(config.figures_dir, f'tsne_True_Labels_{dataset_string}_{vectorizer_string}_{n_clusters}.png'))

def save_results_csv(dataset_name, vectorizer, n_clusters, all_eval_metric_values):

    def clust_algo_to_csv(clustering_algorithms_string, parameters, arguments):
        if (type(arguments[0]) is int): 
            return reduce(
                lambda x,y: f"{x}|{y}", [f"{clustering_algorithms_string}"] + [f"{a}:{b}" for a, b in zip(parameters[1:], map(str, arguments[1:]))] 
            )
        return reduce(
            lambda x,y: f"{x}|{y}", [f"{clustering_algorithms_string}"] + [f"{a}:{b}" for a, b in zip(parameters, map(str, arguments))]   
        )

    start = 0
    approaches_count = 0
    csv_scores = {}
    approachesList = []
    evaluation_metrics = []
    for clustering_algorithms_string in config.clustering_algorithms_strings:
        argumentsList = config.clustering_algorithms_arguments(n_clusters).get(clustering_algorithms_string)
        parameters = config.clustering_algorithms_parameteres().get(clustering_algorithms_string)
        for arguments in argumentsList:
            metrics_per_approach = all_eval_metric_values[start : start + len(config.evaluation_metrics_strings)]
            evaluation_metrics.append(metrics_per_approach)
            approachesList.append(clust_algo_to_csv(clustering_algorithms_string, parameters, arguments))
            start += len(config.evaluation_metrics_strings) 
            approaches_count += 1

    array = np.zeros((len(config.evaluation_metrics_strings), approaches_count))
    for i in range(len(all_eval_metric_values)):
        inx1 = int(i%len(config.evaluation_metrics_strings))
        inx2 = int(i/len(config.evaluation_metrics_strings))
        array[inx1][inx2] = round(all_eval_metric_values[i],2)

    csv_scores.update({"Approaches": approachesList}) 

    for i in range(len(config.evaluation_metrics_strings)):
        csv_scores.update({f"{config.evaluation_metrics_strings[i]}": list(array[i])}) 
        
    df = pd.DataFrame(csv_scores)
    csv_name = f'{dataset_name}_{vectorizer}.csv'
    
 
    # Write DataFrame to CSV File with Default params.
    df.to_csv(os.path.join(config.csv_dir, csv_name), index = False) #, a_rep = 'null'