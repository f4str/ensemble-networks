"""
ensemble of convolutional neural networks
tensorflow
cross entropy loss function
relu activation function
max pooling
softmax logits
adam optimizer
based on LeNet
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def convolution_layer(input, filters, kernel_size=5, strides=1, padding='VALID'):
	channels = input.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[kernel_size, kernel_size, channels, filters],
		mean=0,
		stddev=0.1
	))
	biases = tf.Variable(tf.zeros([filters]))
	layer = tf.nn.conv2d(
		input,
		filter=weights,
		strides=[1, strides, strides, 1],
		padding=padding
	) + biases
	return tf.nn.relu(layer)


def pooling_layer(input, k=2, padding='VALID'):
	return tf.nn.max_pool2d(
		input,
		ksize=[1, k, k, 1],
		strides=[1, k, k, 1],
		padding=padding
	)


def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	return tf.reshape(layer, [-1, num_features])


def fully_connected_layer(input, num_outputs, relu=True):
	num_inputs = input.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[num_inputs, num_outputs],
		mean=0,
		stddev=0.1
	))
	biases = tf.Variable(tf.zeros([num_outputs]))
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
		# Layer 1 = Convolution: 28x28@1 -> 28x28@6 + ReLU
		conv1 = convolution_layer(x_img, filters=6, kernel_size=5, padding='SAME')
		# Layer 2 = Pooling: 28x28@6 -> 14x14@6
		pool1 = pooling_layer(conv1)
		# Layer 3 = Convolution: 14x14@6 -> 10x10@16 + ReLU
		conv2 = convolution_layer(pool1, filters=16, kernel_size=5, padding='VALID')
		# Layer 4 = Pooling: 10x10@16 -> 5x5@16
		pool2 = pooling_layer(conv2)
		# Layer 5 = Flatten: 5x5@16 -> 400
		flat = flatten_layer(pool2)
		# Layer 6 = Fully Connected: 400 -> 120
		fc1 = fully_connected_layer(flat, num_outputs=120)
		# Layer 7 = Fully Connected: 120 -> 84
		fc2 = fully_connected_layer(fc1, num_outputs=84)
		# Layer 8 = Logits: 84 -> 43
		logits = fully_connected_layer(fc2, num_outputs=num_classes, relu=False)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
		
		self.prediction = tf.argmax(logits, axis=1)
		correct_prediction = tf.equal(self.prediction, tf.argmax(y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.predict = tf.nn.softmax(logits)


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
	model.train(200)
