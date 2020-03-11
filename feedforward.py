"""
feedforward neural network
tensorflow
cross entropy loss function
softmax activation function
gradient descent optimizer
"""

import tensorflow as tf


class FNN:
	def __init__(self, num_classes, learning_rate):
		x = tf.get_default_graph().get_tensor_by_name('ensemble/x:0')
		y = tf.get_default_graph().get_tensor_by_name('ensemble/y:0')
		
		num_inputs = x.get_shape()[-1]
		weights = tf.Variable(tf.zeros([num_inputs, num_classes]))
		biases = tf.Variable([tf.zeros([num_classes])])
		self.logits = tf.matmul(x, weights) + biases
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
		
		self.prediction = tf.argmax(self.logits, axis=1)
		correct_prediction = tf.equal(self.prediction, tf.argmax(y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.predict = tf.nn.softmax(self.logits)
