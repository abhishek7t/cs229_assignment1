from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import pdb


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    grad_margin = np.zeros_like(W)
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                grad_margin[:,j] = grad_margin[:,j] + X[i]
                grad_margin[:,y[i]] = grad_margin[:,y[i]] - X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    grad_margin /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW = grad_margin + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = np.matmul(X, W)
    correct_class_scores = scores[np.arange(num_train), y]
    margins = scores - correct_class_scores.reshape([-1,1]) + 1
    losses = np.where(margins > 0, margins, 0)
    losses[np.arange(num_train), y] = 0

    X_reshaped = X.reshape([num_train, 1, -1]) + np.zeros(shape=[num_train, num_classes, X.shape[1]])
    add_grad_contrib = np.where(losses.reshape([num_train, num_classes, 1]) > 0, X_reshaped, 0)

    count_grad_contrib = np.where(losses > 0, 1, 0)
    count_grad_contrib = np.sum(count_grad_contrib, axis=1)
    sub_grad_contrib = np.zeros((num_train, num_classes, X.shape[1]))
    for i in range(num_train):
        sub_grad_contrib[i, y[i], :] = X[i] * count_grad_contrib[i]
    

    dW = -np.sum(sub_grad_contrib, axis=0).T / num_train + np.sum(add_grad_contrib, axis=0).T / num_train + 2 * reg * W

    loss = np.sum(losses) / num_train + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
