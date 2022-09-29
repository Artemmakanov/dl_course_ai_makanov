import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    size = len(ground_truth)
    TP = sum([ ground_truth[i] == True  and prediction[i] == True  for i in range(size)])
    FP = sum([ ground_truth[i] == False and prediction[i] == True  for i in range(size)])
    FN = sum([ ground_truth[i] == True  and prediction[i] == False for i in range(size)])
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    accuracy = sum(np.isclose(prediction, ground_truth))/size
    f1 = 2*(precision*recall)/(precision + recall)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy
#     return accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    size = len(ground_truth)
    # TODO: Implement computing accuracy
    accuracy = sum(np.isclose(prediction, ground_truth))/size

    
    return accuracy
