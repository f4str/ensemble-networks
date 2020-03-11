"""
ensemble of feedforward neural networks
tensorflow
cross entropy loss function
softmax activation function
gradient descent optimizer
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class NeuralNetwork:
	def __init__(self, num_inputs, num_classes, learning_rate):
		self.x = tf.placeholder(tf.float32, [None, num_inputs])
		self.y = tf.placeholder(tf.float32, [None, num_classes])
		
		weights = tf.Variable(tf.zeros([num_inputs, num_classes]))
		biases = tf.Variable([tf.zeros([num_classes])])
		self.logits = tf.matmul(self.x, weights) + biases
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.nn.softmax(self.logits)


class Ensemble:
	def __init__(self):
		self.sess = tf.Session()
		
		self.models = 3
		self.learning_rate = 0.5
		self.batch_size = 128
		
		self.num_inputs = 784
		self.num_classes = 10
		
		self.load_data()
		self.build()
	
	def load_data(self):
		mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
		self.training_data = mnist.train
		self.X_valid = mnist.validation.images
		self.y_valid = mnist.validation.labels
		self.X_test = mnist.test.images
		self.y_test = mnist.test.labels
	
	def build(self):
		self.networks = [NeuralNetwork(self.num_inputs, self.num_classes, self.learning_rate)
							for _ in range(self.models)]
		self.loss = tf.group(*[net.loss for net in self.networks])
		
	def train(self, epochs=500):
		self.sess.run(tf.global_variables_initializer())
		
		for e in range(epochs):
			print(f'epoch {e + 1}')
			for idx, net in enumerate(self.networks):
				x_batch, y_batch = self.training_data.next_batch(self.batch_size)
				feed_dict = {net.x: x_batch, net.y: y_batch}
				self.sess.run(net.optimizer, feed_dict=feed_dict)
				train_loss, train_acc = self.sess.run([net.loss, net.accuracy], feed_dict=feed_dict)
				
				feed_dict = {net.x: self.X_valid, net.y: self.y_valid}
				valid_loss, valid_acc = self.sess.run([net.loss, net.accuracy], feed_dict=feed_dict)
				
				print(f'\tnetwork {idx + 1}:',
					f'train loss = {train_loss:.4f},',
					f'train acc = {train_acc:.4f},',
					f'valid loss = {valid_loss:.4f},',
					f'valid acc = {valid_acc:.4f}'
				)
		
		print('training complete')
		
		for idx, net in enumerate(self.networks):
			feed_dict = {net.x: self.X_test, net.y: self.y_test}
			loss, acc = self.sess.run([net.loss, net.accuracy], feed_dict=feed_dict)
			print(f'network {idx + 1}: test loss = {loss:.4f}, test acc = {acc:.4f}')
		
		
	
	def predict(self, x):
		results = {}
		for net in self.networks:
			feed_dict = {net.x: x}
			output = self.sess.run(tf.argmax(net.prediction, axis=1), feed_dict=feed_dict)
			if output in results:
				results[output] += 1
			else:
				results[output] = 1
		return max(results, key=results.get)
	

if __name__ == '__main__':
	ensemble = Ensemble()
	ensemble.train(500)
