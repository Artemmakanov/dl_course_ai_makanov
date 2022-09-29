import numpy as np


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
        
#         print(pred)
        
        MEAN = pred.max()
#         print(MEAN)
        
        pred -= MEAN
#         print(pred)
        
        pred = np.exp(pred)
        
        SUM = sum(pred)
        
        probs = np.zeros(pred.shape)
        
        probs = pred/SUM 
    
    else:
        probs = None
        
#         print(pred)
        
        MEAN = np.max(pred, axis=1)
        
#         print(MEAN)
        
        pred = (pred.T - MEAN).T
        
#         print(pred)
        
        pred = np.exp(pred)
        
#         print(pred)

        SUM = np.sum(pred, axis=1)
        
#         print(SUM)
        
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
        
#         H = None
        H = - np.log(probs[target_index])
    else:
#         H = None

#         mask = np.zeros((target_index.shape[0]))
        
        H = np.zeros(probs.shape[0])
        
#         for i in range(probs.shape[0]):
            
#             H[i] = -np.log(probs[i][target_index[i][0]])
            
        aux_matr = -np.log(probs.T[target_index])
#         print(aux_matr)
#         print()
#         print(aux_matr.shape)
#         print()
        aux_matr_reshaped = \
    aux_matr.reshape(target_index.shape[0],target_index.shape[0])
#         print(aux_matr_reshaped)
#         print()
#         print(np.diagonal(aux_matr_reshaped))
    
        H = np.sum(np.diagonal(aux_matr_reshaped))
#         np.append(mask, -np.log(probs.T[target_index]))
#         print()
#         print(mask)
    # Your final implementation shouldn't have any loops
    
    return H

def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
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


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    
#     loss = None
#     grad = None
    
    loss = reg_strength * np.sum(W**2)
    
    grad = np.zeros(W.shape)
    grad = reg_strength * 2 * W

    # Your final implementation shouldn't have any loops

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)


    
    loss, dprediction = \
        softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction) /  X.shape[0]
    loss = np.sum(loss) / X.shape[0]
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        
#         print(type(num_features), type(num_classes))
#         print(num_train)
        
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)
        
        loss_history = []
        for epoch in range(epochs):
            
            
            
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            
            
            sections = np.arange(batch_size, num_train, batch_size)
            

            batches_indices = np.array_split(shuffled_indices, sections)

#             TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            

            loss = 0
   
            for batch in range(int(num_train/batch_size)):
                loss_, dW = \
                linear_softmax(X[batches_indices[batch]], \
                               self.W, y[batches_indices[batch]])

                loss_reg, dW_reg = l2_regularization(self.W, reg)

                loss = loss_ + loss_reg

                self.W -= learning_rate*(dW + dW_reg)
                
               
#             print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        
        scores = np.dot(X, self.W)
        
#         print(scores)
        prob = softmax(scores)
        
#         print(prob)
        
#         print(prob)
        
        y_pred = np.argmax(prob, axis = 1)
        
#         print(y_pred)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
#         raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
