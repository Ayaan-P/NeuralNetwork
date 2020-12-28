"""Neural network model."""

from typing import Sequence

import numpy as np
from scipy.special import softmax

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.linear_outputs = {}
        self.relu_outputs = {}
        self.softmax_output = 0
 
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        # deriv w/ respect to W = X
        # deriv w/ respect to b = 1
        #return np.array((X @ W) + b, copy=True) 
        return (X @ W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        #return np.array(np.maximum(0, X), copy=True)
        return np.maximum(0, X)

    def my_softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        #expX = np.exp(X - np.max(X))
        #return expX / expX.sum(axis=1, keepdims=True)
        #return np.exp(X - self.logsumexp(X))
        return softmax(X, axis=1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
       
        for i in range(1, self.num_layers + 1):
            layer = str(i)
            
            X = self.linear(X, self.params["W" + str(layer)], self.params["b" + str(layer)])
            self.linear_outputs[layer] = X
            
            if (i != self.num_layers):
                X = self.relu(X)
                self.relu_outputs[layer] = X
                
        X = self.my_softmax(X)
        self.softmax_output = X

        return X

    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """    
        self.gradients = {}
        
        loss = self.cross_entropy(X, self.softmax_output, y)
        
        # Softmax/CE Layer
        upstream_gradient = 1.0 * self.cross_entropy_softmax_gradient(self.softmax_output, y)

        for i in range(self.num_layers, 0, -1):
            layer = str(i)
            
            # Relu Layer
            if (i != self.num_layers):
                upstream_gradient = np.multiply(upstream_gradient.T, self.relu_gradient(self.relu_outputs[layer]))
            
            # Paramter Update
            if (layer != "1"):
                self.gradients[str(i)] = self.relu_outputs[str(int(layer) - 1)].T @ upstream_gradient
            elif (layer == "1"):
                self.gradients[str(i)] = X.T @ upstream_gradient
            
            self.params["W" + layer] = self.params["W" + layer] - (lr * self.gradients[str(i)])
            self.params["b" + layer] = self.params["b" + layer] - (lr * (1 * upstream_gradient))
            
            # Linear Layer
            upstream_gradient = self.params["W" + layer] @ upstream_gradient.T # linear grad w/ respect to X is W.
            
        return loss
        
    def relu_gradient(self, X):
        X[X<=0] = 0
        X[X>0] = 1
        #return np.array(X, copy=True) 
        return X
        
    def cross_entropy_softmax_gradient(self, scores, Y):
        #print("Entered Cross Entropy Softmax Gradient Function")
        #print("Scores Pre Gradient")
        #print(scores)
        #print(Y)
        #for i in range(len(scores)):
            #print("i: " + str(i))
            #for j in range(len(scores[i])):
                #print("j: " + str(j))
                # correct class
                #if (j == Y[i]):
                    #scores[i][j] = (1 - scores[i][j]) / (-1 * len(scores))
                #else:
                    #scores[i][j] = (0 - scores[i][j]) / (-1 * len(scores))
        #print("Scores Post Gradient")
        #print(scores)
        #return np.array(scores, copy=True)
        m = Y.shape[0]
        grad = scores
        grad[range(m),Y] -= 1
        grad = grad/m
        return grad
    
    def cross_entropy(self, X, scores, y):
        loss = 0
        one_hot_labels = np.zeros((len(X), self.output_size))

        for i in range(len(X)):
            one_hot_labels[i, y[i]] = 1
        return np.mean(-one_hot_labels * np.log(scores))
            
    def logsumexp(self, x):
        c = x.max()
        return c + np.log(np.sum(np.exp(x - c), axis=1, keepdims=True))