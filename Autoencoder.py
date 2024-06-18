# -*- coding: utf-8 -*-
"""
Autoencoder

@author: Tomas Arzola Röber
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class NN:
    def __init__(self, input_nodes, output_nodes, hidden_nodes):
        """
        Initialize the neural network with given input, output, and hidden layers.

        Parameters:
        input_nodes (int): Number of input nodes.
        output_nodes (int): Number of output nodes.
        hidden_nodes (tuple): Tuple containing the number of nodes in each hidden layer.
        """
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes 
        self.hidden_nodes = hidden_nodes
        
        self.weights = []
        
        # Initialize weights for each layer
        last = input_nodes
        for nodes in hidden_nodes:
            self.weights.append(np.random.uniform(-0.5, 0.5, size=(nodes, last)))
            last = nodes
        
        self.weights.append(np.random.uniform(-0.5, 0.5, size=(self.output_nodes, last)))
    
    def add_layer(self, nodes):
        """
        Add an additional layer to the network.

        Parameters:
        nodes (int): Number of nodes in the new layer.
        """
        self.weights[-1] = np.random.uniform(-0.5, 0.5, size=(nodes, self.hidden_nodes[-1]))
        self.weights.append(np.random.uniform(-0.5, 0.5, size=(self.output_nodes, nodes)))  
    
    def feedforward(self, input_vector):
        """
        Perform feedforward operation through the network.

        Parameters:
        input_vector (numpy array): Input data.

        Returns:
        list: List of outputs from each layer.
        """
        last = input_vector
        outputs = [input_vector]
        for matrix in self.weights:
            input_ = np.dot(matrix, last)
            output = expit(input_)
            outputs.append(output)
            last = output
        return outputs
    
    def train(self, input_vector, target, learning_rate):
        """
        Train the network using backpropagation.

        Parameters:
        input_vector (numpy array): Input data.
        target (numpy array): Target data.
        learning_rate (float): Learning rate for weight updates.
        """
        target = np.array(target, ndmin=2).T
        outputs = self.feedforward(input_vector)
        last_output = np.array(outputs[-1], ndmin=2).T
        errors = []
        errors.append(target - last_output)
        last_error = errors[0]
        
        # Calculate errors for each layer
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(self.weights[i + 1].T, last_error)
            errors.append(error)
            last_error = error
            
        errors.reverse()  
        
        # Update weights for each layer
        for i in range(len(errors)):
            output_1 = np.array(outputs[i + 1], ndmin=2).T
            output_2 = np.array(outputs[i], ndmin=2).T
            self.weights[i] += learning_rate * np.dot(errors[i] * output_1 * (1 - output_1), output_2.T)
        

# Load and preprocess the dataset
df = pd.read_csv(r"mnist_data.csv", header=None)
df = df[df.iloc[:, 0] == 2]  # Filter for images of the digit '2'

y = df.iloc[:, 0]
X = df.iloc[:,1:]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0.01, 1))
X = pd.DataFrame(scaler.fit_transform(X))

# Set hyperparameters
epochs = 20
learning_rate = 0.01
input_nodes = 784
output_nodes = 784

# Initialize the neural network
nn = NN(input_nodes, output_nodes, (128,64,32,64,128))

# Train the neural network
for e in range(epochs):
    print(f"Training epoch {e+1}") 
    for i in range(len(y)):
        input_vector = X.iloc[i]
        target_vector = input_vector
        nn.train(input_vector, target_vector, learning_rate)

# Visualize the original and reconstructed images
num_images = 4  # Number of images to display
fig, axes = plt.subplots(num_images, 2, figsize=(12, 6*num_images))

for i in range(num_images):
    # Plot the original image
    sns.heatmap(np.array(X.iloc[i]).reshape(28, 28), ax=axes[i, 0], cbar=False, cmap='gray')
    axes[i, 0].set_title('Original')
    axes[i, 0].axis('off')
    
    # Plot the reconstructed image
    reconstructed_img = nn.feedforward(X.iloc[i])[-1].reshape(28, 28)
    sns.heatmap(reconstructed_img, ax=axes[i, 1], cbar=False, cmap='gray')
    axes[i, 1].set_title('Reconstructed')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()