import numpy as np
from abc import ABCMeta, abstractmethod

class activation_function_class(metaclass=ABCMeta):
	@abstractmethod
	def activation_function(self, z):
		pass

	@abstractmethod
	def activation_prime(self, z):
		pass

class max(activation_function_class):
	def activation_function(self, z):
		f = .5 * (z + abs(z))
		return f

	def activation_prime(self, z):
		if z > 0:
			return 1
		else:
			return 0

class noisy_max(activation_function_class):
	def activation_function(self, z):
		f = .505*z + .495*abs(z)
		return f

	def activation_prime(self, z):
		if z > 0:
			return 1
		else:
			return 0.01