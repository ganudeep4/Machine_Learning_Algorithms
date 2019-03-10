import numpy as np

def perceptron_train(X, Y):

	final_w = []
	final_b = 0.0
	temp_w = []
	temp_b = 0.0

	run_epoch = True
	x_change_in_nxt_epoch = 0
	epoch_count = 0

	for i in range(X.shape[1]):		# Initializes weights of all features to zero at first
		final_w.append(0)

	while run_epoch:
		count = 0
		epoch_count = epoch_count + 1

		for x,y in zip(X, Y):
			count = count + 1
			a = np.dot(final_w, x) + final_b

			if epoch_count > 1 and count == x_change_in_nxt_epoch:
				if set(temp_w) == set(final_w) and temp_b == final_b:
					run_epoch = False

			if not y*a > 0:
				for i in range(len(final_w)):
					final_w[i] = float(final_w[i] + (y * x[i]))
				
				final_b = final_b + y
				x_change_in_nxt_epoch = count
				temp_w = list(final_w)
				temp_b = final_b
		
	return final_w, final_b


def perceptron_test(X, Y, w, b):
	accuracy = 0
	for x, y in zip(X, Y):
		a = np.dot(w, x) + b
		if y * a > 0:
			accuracy = accuracy + 1

	accuracy = (accuracy/len(Y)) * 100
	return accuracy


