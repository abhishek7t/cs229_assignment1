from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import pdb


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    (D, C) = W.shape
    N = X.shape[0]

    for i in range(X.shape[0]):

        f = X[i].dot(W).reshape([C])
        g = f - np.amax(f)

        exp_g = np.exp(g) #(C,)
        h = exp_g / np.sum(exp_g)

        l = -np.log(h[y[i]])
        loss = loss + l

        dl_dg = -np.eye(C)[y[i]] + h

        max_ind = np.argmax(f)
        dg_df = np.zeros([C,C])
        dg_df[:,max_ind] = -1
        dg_df = dg_df + np.eye(C)

        dl_df = np.matmul(dl_dg.reshape([1,C]), dg_df) # (1, C)
        
        for k in range(C):
            dW[:,k] = dW[:, k] + X[i] * dl_df[0, k]

    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    (D, C) = W.shape
    N = X.shape[0]

    f = np.matmul(X, W)
    g = f - np.amax(f, axis=1, keepdims=True)
    exp_g = np.exp(g) #(C,)
    h = exp_g / np.sum(exp_g, axis=1, keepdims=True)
    l = -np.log(h[np.arange(N), y])
    loss = np.sum(l) / N + reg * np.sum(W * W)

    dl_dg = -np.eye(C)[y] + h

    max_ind = np.argmax(f, axis=1)
    dg_df = np.zeros([N,C,C])
    dg_df[:,:,max_ind] = -1
    dg_df = dg_df + np.eye(C)

    dl_df = np.matmul(dl_dg.reshape([N,1,C]), dg_df) # (N, 1, C)

    grad = np.zeros([N,D,C])
    for k in range(C):
        grad[:,:,k] = grad[:, :, k] + X * dl_df[:, 0, k].reshape([-1,1])

    grad = np.sum(grad, axis=0)

    dW = grad / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
