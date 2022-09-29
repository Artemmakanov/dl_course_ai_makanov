import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.n_input = n_input
        self.hidden_layer_size = hidden_layer_size
        self.n_output = n_output
        
#         TODO Create necessary layers
#         raise Exception("Not implemented!")
        self.Layer1 = FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size)
        self.Layer2 = FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output)
        
        self.W1 = self.Layer1.params()['W']
        self.B1 = self.Layer1.params()['B']
        self.W2 = self.Layer2.params()['W']
        self.B2 = self.Layer2.params()['B']
                
        self.ReLULayer = ReLULayer()

        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examplesReLULayer

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
#         raise Exception("Not implemented!")

        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        output1 = self.Layer1.forward(X)
        output2 = self.ReLULayer.forward(output1)
        output3 = self.Layer2.forward(output2)
        
        loss, dpeds = softmax_with_cross_entropy(output3, y)
        
        dinput3 = self.Layer2.backward(dpeds)
        dinput2 = self.ReLULayer.backward(dinput3)
        dinput1 = self.Layer1.backward(dinput2)
        
#         self.W2.grad = 
#         self.B2.grad = 
        
        
        
#         self.W1.grad = 
#         self.B1.grad = 

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
#         raise Exception("Not implemented!")
        
        loss_reg_1, grad_reg_1 = l2_regularization(self.W1.value, self.reg)
        loss_reg_2, grad_reg_2 = l2_regularization(self.W2.value, self.reg)

        loss = loss + loss_reg_1 + loss_reg_2
    
        self.W1.grad += grad_reg_1
        self.W2.grad += grad_reg_2
        
        

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

#         raise Exception("Not implemented!")

        output1 = self.Layer1.forward(X)
        output2 = self.ReLULayer.forward(output1)
        output3 = self.Layer2.forward(output2)
#         print(softmax(output3))
        pred = np.argmax(softmax(output3), axis=1)
        return pred

    def params(self):
        result = {}
        # TODO Implement aggregating all of the params

#         raise Exception("Not implemented!")
        result = {'W1': self.W1, 'B1': self.B1, 'W2': self.W2 ,'B2': self.B2}

        return result
