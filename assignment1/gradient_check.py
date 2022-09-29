import numpy as np


def check_gradient(f, x, delta=1e-4, tol = 1e-1):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float32
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
      
    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        ix = it.multi_index
        
        
        analytic_grad_at_ix = analytic_grad[ix]
        
#         print("x.shape equals ", x.shape)
#         print("x.shape len is %d" % (len(x.shape)))
        
        x_copy_pl_d = x.copy()
        x_copy_mn_d = x.copy()

        x_copy_pl_d[ix] += delta
        x_copy_mn_d[ix] -= delta

        value_pl, _ = f(x_copy_pl_d)
        value_mn, _ = f(x_copy_mn_d)

        value_delta = value_pl - value_mn

        numeric_grad_at_ix = value_delta/(2*delta)

        
            
        print('------------------------------------')
        print('индекс')
        print(ix)
        
        print('аналитический градиент')
        print(analytic_grad_at_ix)
        
        print('численный градиент')
        print(numeric_grad_at_ix)
        print('------------------------------------')
        
        
#         TODO compute value of numeric gradient of f to idx
        if not ((np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol)).all()):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            None
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
