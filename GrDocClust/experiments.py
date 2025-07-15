import GrDocClust.config as config
import pickle
import os
import numpy as np
import pandas as pd
from functools import reduce

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