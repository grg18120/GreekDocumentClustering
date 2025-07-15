from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids #pip install scikit-learn-extra

random_state = 42

def kmeans(X, labels_true, n_clusters, algorithm, init_centers):
    return KMeans(
        n_clusters = n_clusters, 
        init = init_centers, 
        algorithm = algorithm,
        n_init = 10, 
        tol = 1e-4,
        random_state = random_state
    ).fit(X).labels_


def kmedoids(X, labels_true, n_clusters, method, init_centers):
    return KMedoids(
        n_clusters = n_clusters,
        metric = 'cosine',
        method = method, 
        init= init_centers, 
        max_iter = 300, 
        random_state = random_state
    ).fit(X).labels_



