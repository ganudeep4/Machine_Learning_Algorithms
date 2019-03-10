import numpy as np 


def gradient_descent(dev, X, eta):
	
	stop_gradient = True
	X = X.astype(float)
	
	while stop_gradient:

		indices = np.where(dev(X) < 0.0001)
		temp_X = np.take(X, indices)

		X = X - (eta * dev(X))							# New X values.

		np.put(X, indices, temp_X)

		if len(X) <= 1:
			if dev(X) < 0.0001:
				stop_gradient = False

		elif all([x <= 0.0001 for x in dev(X) ]):
			stop_gradient = False

	return X


