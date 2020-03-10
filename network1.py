"""
feedforward neural network
tensorflow
cross entropy loss function
softmax activation function
gradient descent optimizer
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class NeuralNetwork:
	def __init__(self):
		self.sess = tf.Session()
		
		self.learning_rate = 0.5
		self.batch_size = 128
		
		self.patience = 32
		
		self.num_input = 784
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
		self.x = tf.placeholder(tf.float32, [None, self.num_input])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		
		self.weights = tf.Variable(tf.zeros([self.num_input, self.num_classes]))
		self.biases = tf.Variable([tf.zeros([self.num_classes])])
		self.logits = tf.matmul(self.x, self.weights) + self.biases
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.argmax(self.logits)
	
	def train(self, epochs=500):
		self.sess.run(tf.global_variables_initializer())
		total_train_loss = []
		total_train_acc = []
		total_valid_loss = []
		total_valid_acc = []
		best_acc = 0
		no_acc_change = 0
		
		for e in range(epochs):
			x_batch, y_batch = self.training_data.next_batch(self.batch_size)
			feed_dict = {self.x: x_batch, self.y: y_batch}
			self.sess.run(self.optimizer, feed_dict=feed_dict)
			train_loss, train_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			
			feed_dict = {self.x: self.X_valid, self.y: self.y_valid}
			valid_loss, valid_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			
			total_train_loss.append(train_loss)
			total_train_acc.append(train_acc)
			total_valid_loss.append(valid_loss)
			total_valid_acc.append(valid_acc)
			
			print(f'epoch {e + 1}:',
				f'train loss = {train_loss:.4f},',
				f'train acc = {train_acc:.4f},',
				f'valid loss = {valid_loss:.4f},',
				f'valid acc = {valid_acc:.4f}'
			)
			
			if valid_acc > best_acc:
				best_acc = valid_acc
				no_acc_change = 0
			else:
				no_acc_change += 1
			
			if no_acc_change >= self.patience:
				print('early stopping')
				break
		
		print('training complete')
		
		feed_dict = {self.x: self.X_test, self.y: self.y_test}
		acc = self.sess.run(self.accuracy, feed_dict=feed_dict)
		print(f'test accuracy = {acc:.4f}')
		
		return total_train_loss, total_train_acc, total_valid_loss, total_valid_acc


if __name__ == '__main__':
	net = NeuralNetwork()
	net.train(500)
