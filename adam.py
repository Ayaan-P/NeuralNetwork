"""Neural network model."""

from typing import Sequence

import numpy as np
from scipy.special import softmax

class AdamNeuralNetwork:
   

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
        
        self.t = 0
        self.m_w = {"1":0, "2":0, "3":0}
        self.v_w = {"1":0, "2":0, "3":0}
        
        self.m_b = {"1":0, "2":0, "3":0}
        self.v_b = {"1":0, "2":0, "3":0}
 
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
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float, bias_1: float, bias_2: float, epsilon: float
    ) -> float:
            
        self.gradients = {}
        
        loss = self.cross_entropy(X, self.softmax_output, y)
        
        # Softmax/CE Layer
        upstream_gradient = 1.0 * self.cross_entropy_softmax_gradient(self.softmax_output, y)

        for i in range(self.num_layers, 0, -1):
            layer = str(i)

            # Relu LayeR
            if (i != self.num_layers):
                upstream_gradient = np.multiply(upstream_gradient.T, self.relu_gradient(self.relu_outputs[layer]))
            
            # Paramter Update
            if (layer != "1"):
                self.gradients[str(i)] = self.relu_outputs[str(int(layer) - 1)].T @ upstream_gradient
                #self.gradients[str(i)] = self.relu_outputs[str(int(layer) - 1)]
            elif (layer == "1"):
                self.gradients[str(i)] = X.T @ upstream_gradient
                #self.gradients[str(i)] = X
                
            self.t = self.t + 1
            
            # Adam weight update formula
            self.m_w[layer] = bias_1 * self.m_w[layer] + ((1 - bias_1) * self.gradients[str(i)])
            self.v_w[layer] = bias_2 * self.v_w[layer] + ((1 - bias_2) * np.square(self.gradients[str(i)]))
            m_hat_w = self.m_w[layer] / (1 - (bias_1 ** self.t))
            v_hat_w = self.v_w[layer] / (1 - (bias_2 ** self.t))
                
            
            
            # Adam bias update formula
            self.m_b[layer] = bias_1 * self.m_b[layer] + ((1 - bias_1) * upstream_gradient)
            self.v_b[layer] = bias_2 * self.v_b[layer] + ((1 - bias_2) * np.square(upstream_gradient))
            m_hat_b = self.m_b[layer] / (1 - (bias_1 ** self.t))
            v_hat_b = self.v_b[layer] / (1 - (bias_2 ** self.t))
            
            self.params["W" + layer] = self.params["W" + layer] - ((lr * m_hat_w) / (np.sqrt(v_hat_w) + epsilon))
            self.params["b" + layer] = self.params["b" + layer] - ((lr * m_hat_b) / (np.sqrt(v_hat_b) + epsilon))
            
            # Linear Layer
            upstream_gradient = self.params["W" + layer] @ upstream_gradient.T # linear grad w/ respect to X is W.
            
        return loss
        
    def relu_gradient(self, X):
        X[X<=0] = 0
        X[X>0] = 1
        return X
        
    def cross_entropy_softmax_gradient(self, scores, Y):
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
