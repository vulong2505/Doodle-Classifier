""" FCNN for Doodle Classification. """

##### Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Network Architecture
# =============================================================================

class DoodleANN(nn.Module):
    
    def __init__(self):
        # Init
        super(DoodleANN, self).__init__()

        # Neurons
        self.input_layer = 28*28 # features from flattened 28*28 image
        self.hidden_layer_1 = 200 # neurons
        self.hidden_layer_2 = 200 # neurons
        self.output_layer = 10 # categories

        # Architecture
        self.fc1 = nn.Linear(self.input_layer, self.hidden_layer_1)
        self.fc2 = nn.Linear(self.hidden_layer_1, self.hidden_layer_2)
        self.output = nn.Linear(self.hidden_layer_2, self.output_layer)

    def forward(self, x):
        # Flattening the input tensor
        x = x.view(-1, self.input_layer)

        x = F.relu(self.fc1(x))     # Relu activation function for hidden layer 1
        x = F.relu(self.fc2(x))     # Relu activation function for hidden layer 2
        x = self.output(x)

        return x
