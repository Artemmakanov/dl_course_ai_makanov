import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
#     raise Exception("Not implemented!")
#    print(prediction)
#    print(ground_truth)
    correct = np.sum(prediction == ground_truth)
    
    accuracy = correct / len(ground_truth)
    
    return accuracy
