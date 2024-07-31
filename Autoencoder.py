# -*- coding: utf-8 -*-
"""
Autoencoder

@author: Tomas Arzola Röber
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import expit, softmax
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
        for nodes, function in hidden_nodes:
            self.weights.append([np.random.uniform(-0.5, 0.5, size=(nodes, last)), function])
            last = nodes
        
        self.weights.append([np.random.uniform(-0.5, 0.5, size=(self.output_nodes[0], last)), self.output_nodes[1]])
    
    def add_layer(self, new):
        """
        Add an additional layer to the network.

        Parameters:
        nodes (int): Number of nodes in the new layer.
        """
        nodes, activation_function = new
        self.weights[-1][0] = np.random.uniform(-0.5, 0.5, size=(nodes, self.hidden_nodes[-1][0]))
        self.weights.append([np.random.uniform(-0.5, 0.5, size=(self.output_nodes[0], nodes)), activation_function])
        
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
            
            
        for matrix, function in self.weights:
            input_ = np.dot(matrix, last)
            output = function(input_)[0]
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
            error = np.dot(self.weights[i + 1][0].T, last_error)
            errors.append(error)
            last_error = error
            
        errors.reverse()  
        
        derivative = self.weights[-1][1]
        # Update weights for each layer
        for i in range(len(errors)):
            output_1 = np.array(outputs[i + 1], ndmin=2).T
            output_2 = np.array(outputs[i], ndmin=2).T
            self.weights[i][0] += learning_rate * np.dot(errors[i]*derivative(output_1)[1], output_2.T) 
            derivative = self.weights[i][1]

# Aktivierungsfunktionen:
def sigmoid(x):
    return expit(x), expit(x)*(1-expit(x))

def ReLU(x):
    return x*(x>=0), (x>=0)
        
def LeakyReLU(x, alpha=0.01):
    return np.where(x>0 , x, alpha*x), np.where(x>0,1,alpha)
   
        

# Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\mnist_data.csv", header=None)
df = df[df.iloc[:, 0] == 7]  # Filter for images of the digit '2'

y = df.iloc[:, 0]
X = df.iloc[:,1:]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0.01, 1))
X = pd.DataFrame(scaler.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Set hyperparameters
epochs = 10
learning_rate = 0.001
input_nodes = 784
# hidden_nodes = ((256,sigmoid),(128,sigmoid),(64,sigmoid),(128,sigmoid),(256,sigmoid))
hidden_nodes = [[256,LeakyReLU],[256,LeakyReLU]]
output_nodes = [784, sigmoid]
# Initialize the neural network
nn = NN(input_nodes, output_nodes, hidden_nodes)


# Train the neural network
for e in range(epochs):
    print(f"Training epoch {e+1}") 
    for i in range(len(y_train)): 
        input_vector = X_train.iloc[i]
        target_vector = input_vector
        nn.train(input_vector, target_vector, learning_rate)

# Visualize the original and reconstructed images
num_images = 4  # Number of images to display
fig, axes = plt.subplots(num_images, 2, figsize=(12, 6*num_images))

for i in range(num_images):
    # Plot the original image
    sns.heatmap(np.array(X_test.iloc[i]).reshape(28, 28), ax=axes[i, 0], cbar=False, cmap='gray')
    axes[i, 0].set_title('Original')
    axes[i, 0].axis('off')
    
    # Plot the reconstructed image
    reconstructed_img = nn.feedforward(X_test.iloc[i])[-1].reshape(28, 28)
    sns.heatmap(reconstructed_img, ax=axes[i, 1], cbar=False, cmap='gray')
    axes[i, 1].set_title('Reconstructed')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()