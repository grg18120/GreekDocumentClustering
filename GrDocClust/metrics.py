from sklearn import metrics
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

# """
# External evaluation, clustering is compared to an existing "ground truth" classification
# """

def fowlkes_mallows_index(labels_true, labels_pred):
	"""
	Fowlkes-Mallows metric
	Values close to zero indicate two label assignments that are largely independent, 
	while values close to one indicate significant agreement. 
	Further, values of exactly 0 indicate purely independent label assignments and a FMI of exactly 1 indicates that 
	the two label assignments are equal 
	"""
	return metrics.fowlkes_mallows_score(labels_true, labels_pred)

def v_measure_index(labels_true, labels_pred):
	"""
	Homogeneity(h), completeness(c) and V-measure metric
	V-measure = [(1+b)*h*c] / [(b*h + c)]   b = 1.0(default)
	0.0 is as bad as it can be, 1.0 is a perfect score.
	h = metrics.homogeneity_score(labels_true, labels_pred)
	c = metrics.completeness_score(labels_true, labels_pred)
	v = ((1+b)*h*c) / ((b*h + c))   b = 1.0
	"""
	return metrics.v_measure_score(labels_true, labels_pred, beta = 1.0) 

def adjusted_mutual_information(labels_true, labels_pred):
	return metrics.adjusted_mutual_info_score(labels_true, labels_pred)


def accuracy(true_row_labels, predicted_row_labels):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = metrics.confusion_matrix(true_row_labels, predicted_row_labels)
    indexes = linear_assignment(_make_cost_m(cm))
    total = 0

    for row, column in zip(indexes[0], indexes[1]):
        value = cm[row][column]
        total += value

    return (total * 1. / np.sum(cm))


def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)