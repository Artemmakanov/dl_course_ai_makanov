import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
#     raise Exception("Not implemented!")
    loss = reg_strength * np.sum(W**2)
    
    grad = np.zeros(W.shape)
    grad = reg_strength * 2 * W
    
    
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    pred = predictions.copy()
    
    if len(pred.shape) == 1:
        probs = None
        
        MEAN = pred.max()
        
        pred -= MEAN
        
        pred = n#         print(pred)p.exp(pred)
        
        SUM = sum(pred)
        
        probs = np.zeros(pred.shape)
        
        probs = pred/SUM 
    
    else:
        probs = None
        
        MEAN = np.max(pred, axis=1)

        pred = (pred.T - MEAN).T

        pred = np.exp(pred)

        SUM = np.sum(pred, axis=1)

        probs = np.zeros(pred.shape)
        
        probs = (pred.T/SUM).T
        
        
    # Your final implementation shouldn't have any loops
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    if len(probs.shape) == 1:
 
        H = - np.log(probs[target_index])
    else:
        H = np.zeros(probs.shape[0])
       
        aux_matr = -np.log(probs.T[target_index])

        aux_matr_reshaped = \
    aux_matr.reshape(target_index.shape[0],target_index.shape[0])

        H = np.sum(np.diagonal(aux_matr_reshaped))

    # Your final implementation shouldn't have any loops
    
    return H
def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
#     raise Exception("Not implemented!")
    if len(predictions.shape) == 1:
        loss = None
        dprediction = None
        
        probs = softmax(predictions)
        loss = cross_entropy_loss(probs, target_index)
        
        dprediction = np.zeros(predictions.shape)

        dprediction = probs.copy()
        
        dprediction[target_index] -= 1
        
        
    else:
        loss = None
        dprediction = None
        
        probs = softmax(predictions)
        
        loss = cross_entropy_loss(probs, target_index)
        
        dprediction = np.zeros(predictions.shape)
        
        dprediction = probs.copy()
        
        
        for i in range(len(predictions)):
            dprediction[i][target_index[i]] -= 1
            
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
#         raise Exception("Not implemented!")
        self.X = X
        result = np.where(X > 0, X, 0 )
#         analytic_grad = np.where(X > 0, 1, 0)
        return result
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")
        d_result = np.zeros(d_out.shape)
        d_result = np.where(self.X > 0, d_out, 0 )
#         print(self.X)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}

class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")
#         print(self.W.value)
#         return 0
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

#         raise Exception("Not implemented!")
#         d_input = np.zeros(d_out.shape)
        d_input = np.dot(d_out, self.W.value.T)
        
        self.W.grad = np.dot(self.X.T, d_out)
        
#         print('d_out: ')
#         print(d_out)
#         print('np.sum(d_out, axis = 0): ')
#         print(np.sum(d_out, axis = 0))
#         self.B.grad = d_out * self.B.value
#         self.B.grad = np.zeros(self.B.value.shape)
#         print()
        self.B.grad = np.sum(d_out, axis = 0).reshape(self.B.value.shape)
#         print(np.sum(d_out, axis = 0).reshape(self.B.value.shape))
#         print(self.B.value.shape)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
