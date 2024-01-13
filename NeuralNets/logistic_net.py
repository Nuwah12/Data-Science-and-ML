import os
import torch
from torch import nn

# Define a neural network class
class LogisticNet(nn.Module):
	def __init__(self, input_size, prediction_threshold):
		super.__init__()
		self.flatten = nn.Flatten()
		self.net = nn.Sequential(
				nn.Linear(input_size, 1), # A Layer that calculates the weighted sum of input_size features
				nn.Sigmoid() # Sigmoid activation function
			)
		self.prediction_threshold = prediction_threshold
	def forward(self, x):
		x = self.flatten(x)
		sigmoid_out = self.model(x)
		predictions = (sigmoid_out > self.prediction_threshold).float()
		return predictions
