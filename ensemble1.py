"""
Ensemble of Convolutional Neural Networks
MNIST Classifiers
LeNet, AlexNet and VGGNet
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_networks import LeNet, AlexNet, VGGNet


def vote(tensor):
	unique, _, counts = tf.unique_with_counts(tensor)
	majority = tf.argmax(counts)
	prediction = tf.gather(unique, majority)
	return prediction


class Ensemble:
	def __init__(self):
		self.sess = tf.Session()
		
		self.learning_rate = 0.001
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
		with tf.variable_scope('ensemble', reuse=tf.AUTO_REUSE) as scope:
			self.x = tf.placeholder(tf.float32, [None, self.num_inputs], name='x')
			self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')
		
		self.networks = [LeNet(), AlexNet(), VGGNet()]
		self.loss = tf.reduce_mean([net.loss for net in self.networks])
		
		predictions = tf.stack([net.prediction for net in self.networks], axis=-1)
		prediction = tf.map_fn(vote, predictions)
		correct_prediction = tf.equal(prediction, tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
	def train(self, epochs=500):
		self.sess.run(tf.global_variables_initializer())
		
		for e in range(epochs):
			print(f'epoch {e + 1}')
			for idx, net in enumerate(self.networks):
				x_batch, y_batch = self.training_data.next_batch(self.batch_size)
				feed_dict = {self.x: x_batch, self.y: y_batch}
				self.sess.run(net.optimizer, feed_dict=feed_dict)
				train_loss, train_acc = self.sess.run([net.loss, net.accuracy], feed_dict=feed_dict)
				
				feed_dict = {self.x: self.X_valid, self.y: self.y_valid}
				valid_loss, valid_acc = self.sess.run([net.loss, net.accuracy], feed_dict=feed_dict)
				
				print(f'\tnetwork {idx + 1}:',
					f'train loss = {train_loss:.4f},',
					f'train acc = {train_acc:.4f},',
					f'valid loss = {valid_loss:.4f},',
					f'valid acc = {valid_acc:.4f}'
				)
		
		print('training complete')
		
		feed_dict = {self.x: self.X_test, self.y: self.y_test}
		for idx, net in enumerate(self.networks):
			loss, acc = self.sess.run([net.loss, net.accuracy], feed_dict=feed_dict)
			print(f'network {idx + 1}: test loss = {loss:.4f}, test acc = {acc:.4f}')
		
		loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
		print(f'test loss = {loss:.4f}, test acc = {acc:.4f}')


if __name__ == '__main__':
	model = Ensemble()
	model.train(10)
