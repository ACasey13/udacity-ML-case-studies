import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_hidden, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param num_hidden: number of hidden layers
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # having not used pyTorch before, I think the 
        # attempt to make the number of hidden layers variable
        # has caused an issue....
        self.layers = nn.ModuleList()
        if num_hidden > 0:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.Dropout(0.25))
            for _ in range(num_hidden-1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU(True))
                self.layers.append(nn.Dropout(0.25))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        for layer in self.layers:
            x = layer(x)
        return x