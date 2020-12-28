"""Neural network model."""

from typing import Sequence

import numpy as np
from scipy.special import softmax

class NeuralNetwork:

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        
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
       
        # deriv w/ respect to W = X
        # deriv w/ respect to b = 1
        #return np.array((X @ W) + b, copy=True) 
        return (X @ W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        
        #return np.array(np.maximum(0, X), copy=True)
        return np.maximum(0, X)

    def my_softmax(self, X: np.ndarray) -> np.ndarray:
        
        # TODO: implement me
        #expX = np.exp(X - np.max(X))
        #return expX / expX.sum(axis=1, keepdims=True)
        #return np.exp(X - self.logsumexp(X))
        return softmax(X, axis=1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        
       
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
