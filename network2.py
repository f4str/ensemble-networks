"""
ensemble of convolutional neural networks
tensorflow 
cross entropy loss function
relu convolution activation function
max pooling
softmax fully connected activation function
adam optimizer
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def convolution_layer(input, num_filters, filter_size=5, strides=1, k=2):
	num_inputs = input.get_shape()[-1]
	shape = [filter_size, filter_size, num_inputs, num_filters]
	weights = tf.Variable(tf.random.truncated_normal(shape, stddev=0.05))
	biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
	layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides, strides, 1], padding='SAME') + biases
	layer = tf.nn.max_pool2d(layer, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
	return tf.nn.relu(layer)


def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	return tf.reshape(layer, [-1, num_features])


def fully_connected_layer(input, num_outputs, relu=True):
	num_inputs = input.get_shape()[-1]
	weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
	biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
	layer = tf.matmul(input, weights) + biases
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


def vote(tensor):
	unique, _, counts = tf.unique_with_counts(tensor)
	majority = tf.argmax(counts)
	prediction = tf.gather(unique, majority)
	return prediction


class NeuralNetwork:
	def __init__(self, num_classes, learning_rate):
		x = tf.get_default_graph().get_tensor_by_name('ensemble/x:0')
		y = tf.get_default_graph().get_tensor_by_name('ensemble/y:0')
		
		# Layer 0 = Reshape: 784 -> 28x28@1
		x_img = tf.reshape(x, shape=[-1, 28, 28, 1])
		# Layer 1 = Convolution + Pooling: 28x28@1 -> 14x14@16
		conv_layer1 = convolution_layer(x_img, num_filters=16)
		# Layer 2 = Convolution + Pooling: 14x14@16 -> 7x7@32
		conv_layer2 = convolution_layer(conv_layer1, num_filters=32)
		# Layer 3 = Flatten: 7x7@32 -> 1568
		flat_layer = flatten_layer(conv_layer2)
		# Layer 4 = Fully Connected: 1568 -> 128
		fc_layer = fully_connected_layer(flat_layer, num_outputs=128)
		# Layer 5 = Logits: 128 -> 10
		self.logits = fully_connected_layer(fc_layer, num_outputs=num_classes, relu=False)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
		
		self.prediction = tf.argmax(self.logits, axis=1)
		correct_prediction = tf.equal(self.prediction, tf.argmax(y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.predict = tf.nn.softmax(self.logits)


class Ensemble:
	def __init__(self):
		self.sess = tf.Session()
		
		self.models = 3
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
		
		self.networks = [NeuralNetwork(self.num_classes, self.learning_rate) for _ in range(self.models)]
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
	model.train(100)
