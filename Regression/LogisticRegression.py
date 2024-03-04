###############
# Class for Logistic Regression
# Noah Burget
###############
import numpy as np
class LogisticRegression():
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.residuals = []
		self.cost = []
		self.predictions=[]
		self.weights=np.zeros(X.shape[1]) 
		self.bias=0
	def __str__(self):
		msg = """
			Plain Logistic Regression object.
			X = {}
			y = {}
			""".format(self.X, self.y)
		return msg
	def log_loss_gradient_descent(self, num_iters=100, learning_rate_alpha=0.005):
		m = self.X.shape[1]
		for i in range(num_iters):
			# Make predictions with current data and weights
			yhat = self.predict(self.X, self.weights)
			self.predictions = yhat
			# Calculate  the gradient of the log loss (backpropogation/COST DERIVATIVE)
			gradient = np.dot(self.X.T, yhat - self.y)
			gradient /= m # avg gradient across all features
			gradient *= learning_rate_alpha # adjust graidient by learning rate
			# Adjust weights
			self.weights -= gradient # update 
			# Calculate the log loss function with the predicted probabilities and true labels just for auditing purposes
			log_loss_cost = self._log_loss_cost(yhat, np.array(self.y))
			self.cost.append(log_loss_cost)
	def _log_loss_cost(self, predictions, labels):
		"""
		Cost = (labels*log(predictions) + (1-labels))
		"""
		m = len(labels)
		pos_cost = -labels * np.log(predictions) # Error for the positive class (1)
		neg_cost = (1-labels) * np.log(1-predictions) # Error for the negative class (0)
		cost = pos_cost - neg_cost
		cost = cost.sum() / m # Average cost across all training instances
		return cost
	def predict(self, features, weights):
		"""
		Uses data * current weights to make predictions via sigmoid function
		arguments:
			features - data
			weights - current weights
		"""
		return self._sigmoid(np.dot(features, weights))
	def _sigmoid(self, x):
		"""
		Calculates sigma(x)
		arguments:
			x - data
		"""
		return 1/(1 + np.exp(-x))
