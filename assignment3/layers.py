import numpy as np


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
    # TODO: Copy from previous assignment
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
    # TODO copy from the previous assignment
    
    
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        result = np.where(X > 0, X, 0 )
        return result
    

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = np.zeros(d_out.shape)
        d_result = np.where(self.X > 0, d_out, 0 )
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_input = np.zeros(d_out.shape)
        d_input = np.dot(d_out, self.W.value.T)
        
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis = 0).reshape(self.B.value.shape)
        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        
        out_height = height + 2*self.padding - self.filter_size + 1
        out_width = width + 2*self.padding - self.filter_size + 1
        
        self.X = X
        pad = np.zeros((batch_size,height + 2*self.padding, width + 2*self.padding,channels))
        pad[ :, self.padding:height + self.padding, self.padding:width+ self.padding, :] += self.X
        
        self.X_pad = pad
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location

                low_border_xy_height = y
                high_border_xy_height = self.filter_size + y
           
                low_border_xy_width = x
                high_border_xy_width = self.filter_size + x
                
                I = self.X_pad[:,low_border_xy_height:high_border_xy_height, low_border_xy_width:high_border_xy_width,:]
                                
                prod = \
                np.dot(I.reshape((batch_size, self.filter_size*self.filter_size*self.in_channels)), \
                       self.W.value.reshape((self.filter_size*self.filter_size*self.in_channels, self.out_channels))) + \
                       self.B.value
                
                result[:,y,x,:] += prod
                
                
        return result
    

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        
        pad_height = height + 2*self.padding 
        pad_width = width + 2*self.padding 
        
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        d_input = np.zeros(self.X.shape)
        

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

#                 W_cut = self.W.value

#                 W_reshape = W_cut.reshape((self.filter_size*self.filter_size*channels, out_channels))
#                 d_out_part = d_out[:,y,x,:]

#                 d_input_part = np.dot(d_out_part, W_reshape.T)


#                 d_input_part_reshape = d_input_part.reshape((batch_size, height, width, channels))

 
#                 d_input[:, y:y + self.filter_size, \
#                         x:x + self.filter_size, :] += d_input_part_reshape 

                W_cut = self.W.value[0 + max(0, self.padding - y):
                                     self.filter_size - max(0, y - (height + self.padding - self.filter_size)),
                                     0 + max(0, self.padding - x):
                                     self.filter_size - max(0, x - (width + self.padding - self.filter_size)), :, :]
                
                low_border_xy_height = max(self.padding, y) 
                high_border_xy_height = min(height + self.padding, self.filter_size + y)
           
                low_border_xy_width = max(self.padding, x) 
                high_border_xy_width = min(width + self.padding, self.filter_size + x)
                
                
                filter_size_y = high_border_xy_height - low_border_xy_height
                filter_size_x = high_border_xy_width - low_border_xy_width
                
                W_reshape = W_cut.reshape((filter_size_y*filter_size_x*channels, out_channels))
                d_out_part = d_out[:,y,x,:]

                d_input_part = np.dot(d_out_part, W_reshape.T)


                d_input_part_reshape = d_input_part.reshape((batch_size, filter_size_y, filter_size_x, channels))

                
#                 print('W_cut shape is', W_cut.shape)
#                 print('filter_size_y is', filter_size_y)
#                 print('filter_size_x is', filter_size_x)
                
                
                d_input[:, low_border_xy_height - self.padding: high_border_xy_height - self.padding, \
                        low_border_xy_width - self.padding:high_border_xy_width - self.padding, :] += d_input_part_reshape 
                
        
#                 low_border_xy_height = y 
#                 high_border_xy_height = self.filter_size + y
           
#                 low_border_xy_width = x
#                 high_border_xy_width = self.filter_size + x

#                 I = self.X[:,low_border_xy_height:high_border_xy_height, low_border_xy_width:high_border_xy_width,:]
                 
#                 I_reshape = I.reshape((batch_size, self.filter_size*self.filter_size*channels))
            

#                 grad_part = np.dot(I_reshape.T, d_out_part)
#                 grad_part_reshape = grad_part.reshape((self.filter_size, self.filter_size, channels, out_channels))
                
#                 self.W.grad += grad_part_reshape

#                 self.B.grad += np.sum(d_out_part, axis = 0).reshape(self.B.value.shape)





#                 low_border_xy_height = y 
#                 high_border_xy_height = self.filter_size + y
           
#                 low_border_xy_width = x
#                 high_border_xy_width = self.filter_size + x

                I = self.X[:,low_border_xy_height - self.padding: high_border_xy_height - self.padding, \
                           low_border_xy_width - self.padding:high_border_xy_width - self.padding,:]
                 
                I_reshape = I.reshape((batch_size, filter_size_y*filter_size_x*channels))
            

                grad_part = np.dot(I_reshape.T, d_out_part)
                grad_part_reshape = grad_part.reshape((filter_size_y, filter_size_x, channels, out_channels))
                
                self.W.grad[0 + max(0, self.padding - y):
                            self.filter_size - max(0, y - (height + self.padding - self.filter_size)),
                            0 + max(0, self.padding - x):
                            self.filter_size - max(0, x - (width + self.padding - self.filter_size)), :, :] += \
                grad_part_reshape
                

                self.B.grad += np.sum(d_out_part, axis = 0).reshape(self.B.value.shape)
                
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.out_height = None
        self.out_width = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        
        self.X = X
        self.out_height = int((height - self.pool_size) / self.stride) + 1
        self.out_width = int((width - self.pool_size) / self.stride) + 1
        
        result = np.zeros((batch_size, self.out_height, self.out_width, channels))
        for batch in range(batch_size):
            for y in range(self.out_height):
                for x in range(self.out_width):
                    for channel in range(channels):
                        I = self.X[batch, \
                                   self.stride*y:self.stride*y + self.pool_size, \
                                   self.stride*x:self.stride*x + self.pool_size, \
                                   channel]
                        max_pool = np.max(I)
                        result[batch, y, x ,channel] += max_pool 

                
        
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        
        d_input = np.zeros((batch_size, height, width, channels))
        f = lambda c: [c // self.pool_size, c % self.pool_size]
        
    
        for batch in range(batch_size):
            for y in range(self.out_height):
                for x in range(self.out_width):
                    for channel in range(channels):
                        I = self.X[batch, \
                                   y*self.stride:y*self.stride + self.pool_size, \
                                   x*self.stride:x*self.stride + self.pool_size, \
                                   channel]
                        
                        I_flatten = I.flatten()
                        indx = np.argmax(I_flatten)
                        ix = f(indx)
                        
                        d_input[batch ,y*self.stride + ix[0],x*self.stride + ix[1], channel] += d_out[batch,y, x,channel]


        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
#         [batch_size, hight*width*channels]
#         raise Exception("Not implemented!")
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
#         raise Exception("Not implemented!")
         
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
