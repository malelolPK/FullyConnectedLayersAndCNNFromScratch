import torch
from utils import optimizer_adam
import math
import os

class Three_Leyer_Model:
    def __init__(self, input_dim, hidden_dim_one, hidden_dim_two, output_dim, weight_scale, l2_reg, mode, dtype=torch.float32, device='cpu'):

        self.params = {}
        self.grads = {}
        self.l2_reg = l2_reg
        self.mode = mode
        self.use_drupout = False
        self.momentum_sgd = {}

        self.params["W1"] = torch.normal(0.0, weight_scale, size=(hidden_dim_one, input_dim), dtype=dtype, device=device)
        self.params["W2"] = torch.normal(0.0, weight_scale, size=(hidden_dim_two, hidden_dim_one), dtype=dtype, device=device)
        self.params["W3"] = torch.normal(0.0, weight_scale, size=(output_dim, hidden_dim_two), dtype=dtype, device=device)

        self.params["b1"] = torch.zeros((hidden_dim_one, 1), dtype=dtype, device=device)
        self.params["b2"] = torch.zeros((hidden_dim_two, 1), dtype=dtype, device=device)
        self.params["b3"] = torch.zeros((output_dim, 1), dtype=dtype, device=device)

    def reLU(self, X):
        x_relu = X.clone()
        x_relu[X < 0] = 0
        return x_relu

    def softmax(self, Z):
        exp_scores = torch.exp(Z - torch.max(Z, dim=0, keepdim=True)[0])
        probs = exp_scores / torch.sum(exp_scores, dim=0, keepdim=True)
        return probs

    def droput_forward(self, x, p):
        # inverted dropout
        mask = None
        out = None
        if self.mode == 'train':
            keep_prob = 1 - p
            mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
            mask /= keep_prob
            out = x * mask
        elif self.mode == 'test' or self.mode == 'val':
            out = x
            mask = None
        return out, mask

    def droput_backward(self, dout, mask):
        dx = None
        if self.mode == 'train':
            dx = dout * mask
        elif self.mode == 'test' or self.mode == 'val':
            dx = dout
        return dx

    def forward(self, X, dropout_p):

        mask_forw_drop = None
        if dropout_p != 0:
            self.use_drupout = True
        z1 = torch.matmul(self.params["W1"], X) + self.params["b1"]
        h1 = self.reLU(z1)
        if self.use_drupout:
            h1, mask_forw_drop = self.droput_forward(h1, dropout_p)
        z2 = torch.matmul(self.params["W2"], h1) + self.params["b2"]
        h2 = self.reLU(z2)
        z3 = torch.matmul(self.params["W3"], h2) + self.params["b3"]
        print("z3",z3.shape)
        y_pred = self.softmax(z3)
        return z1, h1, z2, h2, z3, y_pred, mask_forw_drop

    def loss(self, y_pred, y):
        N = y.shape[0]
        eps = 1e-15  # mała wartość aby uniknąć log(0)
        #  clipping wartości prawdopodobieństw
        y_pred = torch.clamp(y_pred, eps, 1.0)
        correct_class_probs = y_pred[y, range(N)]
        loss = -torch.sum(torch.log(correct_class_probs)) / N
        loss += self.l2_reg * (torch.sum(self.params["W1"] ** 2) + torch.sum(self.params["W2"] ** 2))
        return loss

    def backprob_reLU(self, dz, h):
        dx = dz.clone()
        dx[h < 0] = 0
        return dx

    def backward(self, X, y_pred, y, z1, h1, z2, h2, z3, mask_forw_drop):
        N = y.shape[0]
        m = X.shape[0]
        dloss = y_pred.clone()
        dloss[y, range(N)] -= 1
        dloss /= m

        self.grads["W3"] = torch.matmul(dloss, h2.T)
        self.grads["b3"] = torch.sum(dloss, dim=1, keepdim=True)
        dz2 = torch.matmul(self.params["W3"].T, dloss)
        if self.use_drupout:
            dz1 = self.droput_backward(dz2, mask_forw_drop)
        dz2 = self.backprob_reLU(dz2, h2)

        self.grads["W2"] = torch.matmul(dz2, h1.T)
        self.grads["b2"] = torch.sum(dz2, dim=1, keepdim=True)

        dz1 = torch.matmul(self.params["W2"].T, dz2)
        if self.use_drupout:
            dz1 = self.droput_backward(dz1, mask_forw_drop)
        dz1 = self.backprob_reLU(dz1, h1)

        self.grads["W1"] = torch.matmul(dz1, X.T)
        self.grads["b1"] = torch.sum(dz1, dim=1, keepdim=True)
        # dl2_reg
        # self.grads["W2"] += self.l2_reg * (2 * self.params["W2"])
        # self.grads["W1"] += self.l2_reg * (2 * self.params["W1"])

        return self.grads

    def SGD(self, lr):
        self.params["W2"] -= lr * self.grads["W2"]
        self.params["W1"] -= lr * self.grads["W1"]
        self.params["b2"] -= lr * self.grads["b2"]
        self.params["b1"] -= lr * self.grads["b1"]

    def initialize_velocity_momentum(self):
        self.momentum_sgd["W1"] = torch.zeros_like(self.params["W1"])
        self.momentum_sgd["b1"] = torch.zeros_like(self.params["b1"])
        self.momentum_sgd["W2"] = torch.zeros_like(self.params["W2"])
        self.momentum_sgd["b2"] = torch.zeros_like(self.params["b2"])
        self.momentum_sgd["W3"] = torch.zeros_like(self.params["W3"])
        self.momentum_sgd["b3"] = torch.zeros_like(self.params["b3"])

    def SGD_momentum(self, beta, lr):
        self.momentum_sgd["W1"] = beta * self.momentum_sgd["W1"] + (1 - beta) * self.grads["W1"]
        self.momentum_sgd["b1"] = beta * self.momentum_sgd["b1"] + (1 - beta) * self.grads["b1"]
        self.momentum_sgd["W2"] = beta * self.momentum_sgd["W2"] + (1 - beta) * self.grads["W2"]
        self.momentum_sgd["b2"] = beta * self.momentum_sgd["b2"] + (1 - beta) * self.grads["b2"]
        self.momentum_sgd["W3"] = beta * self.momentum_sgd["W3"] + (1 - beta) * self.grads["W3"]
        self.momentum_sgd["b3"] = beta * self.momentum_sgd["b3"] + (1 - beta) * self.grads["b3"]

        #update parametrs
        self.params["W1"] -= lr * self.momentum_sgd["W1"]
        self.params["W2"] -= lr * self.momentum_sgd["W2"]
        self.params["W3"] -= lr * self.momentum_sgd["W3"]
        self.params["b1"] -= lr * self.momentum_sgd["b1"]
        self.params["b2"] -= lr * self.momentum_sgd["b2"]
        self.params["b3"] -= lr * self.momentum_sgd["b3"]

    def adam(self, t, eta):
        adam_layer_one = optimizer_adam.Adam_Optimizer(eta=eta)
        adam_layer_two = optimizer_adam.Adam_Optimizer(eta=eta)
        self.params["W2"], self.params["b2"] = adam_layer_two.update(self.params["W2"], self.grads["W2"], self.params["b2"], self.grads["b2"], t)
        self.params["W1"], self.params["b1"] = adam_layer_one.update(self.params["W1"], self.grads["W1"], self.params["b1"], self.grads["b1"], t)

    def get_predictions(self, y_pred):
        return torch.argmax(y_pred, dim=0)

    def get_accuracy(self, predictions, y):
        return torch.sum(predictions == y) / y.shape[0]

    def save(self, path="C:/Users/malel/Desktop/python/mnist_project/data/models/linear_model.pth"):
        checkpoint = {
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path="C:/Users/malel/Desktop/python/mnist_project/data/models/linear_model.pth"):
        checkpoint = torch.load(path, weights_only=True)
        self.params = checkpoint['params']
        print("load checkpoint file: {}".format(path))

    # Funkcja Time-Based Decay
    def time_based_step_decay(self,epoch, eta, decay):
        lr = eta * (1. / (1. + decay * epoch))
        return lr

    def exponential_step_decay(self,epoch, eta, decay):
        lr = eta / (1 + decay * epoch)
        return lr