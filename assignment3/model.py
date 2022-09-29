import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization,
    softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
#         raise Exception("Not implemented!")

        image_width, image_height, in_channels = input_shape

        self.conv_layer1 = ConvolutionalLayer(in_channels=in_channels, out_channels=conv1_channels, filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(4, 4)
        
        self.conv_layer2 = ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv1_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(4, 4)
        
        self.flatten = Flattener()
        self.fully_connect = FullyConnectedLayer(8, n_output_classes)
        self.softmax = softmax_with_cross_entropy

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
#         raise Exception("Not implemented!")
        self.conv_layer1.W.grad = np.zeros(self.conv_layer1.W.value.shape)
        self.conv_layer1.B.grad = np.zeros(self.conv_layer1.B.value.shape)
    
        self.conv_layer2.W.grad = np.zeros(self.conv_layer2.W.value.shape)
        self.conv_layer2.B.grad = np.zeros(self.conv_layer2.B.value.shape)
        
        self.fully_connect.W.grad = np.zeros(self.fully_connect.W.value.shape)
        self.fully_connect.B.grad = np.zeros(self.fully_connect.B.value.shape)
        
        
        output1 = self.conv_layer1.forward(X)
        output2 = self.relu1.forward(output1)
        output3 = self.maxpool1.forward(output2)
        
        output4 = self.conv_layer2.forward(output3)
        output5 = self.relu2.forward(output4)
        output6 = self.maxpool2.forward(output5)
        
        output7 = self.flatten.forward(output6)
        output8 = self.fully_connect.forward(output7)
        
        loss, dprediction = self.softmax(output8, y)
        
        dinput1 = self.fully_connect.backward(dprediction)
        dinput2 = self.flatten.backward(dinput1)
        
        dinput3 = self.maxpool2.backward(dinput2)
        dinput4 = self.relu2.backward(dinput3)
        dinput5 = self.conv_layer2.backward(dinput4)
        
        dinput6 = self.maxpool1.backward(dinput5)
        dinput7 = self.relu1.backward(dinput6)
        dinput8 = self.conv_layer1.backward(dinput7)
        
        
        
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
#         pred = np.zeros(X.shape[0], np.int)


#         output1 = self.Layer1.forward(X)
#         output2 = self.ReLULayer.forward(output1)
#         output3 = self.Layer2.forward(output2)
# #         print(softmax(output3))
#         pred = np.argmax(softmax(output3), axis=1)
    
    
        output1 = self.conv_layer1.forward(X)
        output2 = self.relu1.forward(output1)
        output3 = self.maxpool1.forward(output2)
        
        output4 = self.conv_layer2.forward(output3)
        output5 = self.relu2.forward(output4)
        output6 = self.maxpool2.forward(output5)
        
        output7 = self.flatten.forward(output6)
        output8 = self.fully_connect.forward(output7)
        probs = softmax(output8)
#         print(probs)
        pred = np.argmax(probs, axis=1)
#         print(pred)
        
        return pred

    def params(self):
        result = {'conv_layer1_W': self.conv_layer1.W,
                 'conv_layer1_B': self.conv_layer1.B,
                 'conv_layer2_W': self.conv_layer2.W,
                 'conv_layer2_B': self.conv_layer2.B,
                 'fc_W': self.fully_connect.W,
                 'fc_B': self.fully_connect.B}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
#         raise Exception("Not implemented!")
        
        return result
