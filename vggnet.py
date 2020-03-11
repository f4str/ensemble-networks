"""
convolutional neural network
tensorflow
cross entropy loss function
relu activation function
max pooling
softmax logits
adam optimizer
based on simplified VGGNet
"""

import tensorflow as tf


def convolution_layer(inputs, filters, kernel_size=5, strides=1, padding='VALID'):
	channels = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[kernel_size, kernel_size, channels, filters],
		mean=0,
		stddev=0.1
	))
	biases = tf.Variable(tf.zeros([filters]))
	layer = tf.nn.conv2d(
		inputs,
		filter=weights,
		strides=[1, strides, strides, 1],
		padding=padding
	) + biases
	return tf.nn.relu(layer)


def pooling_layer(inputs, k=2, padding='VALID'):
	return tf.nn.max_pool2d(
		inputs,
		ksize=[1, k, k, 1],
		strides=[1, k, k, 1],
		padding=padding
	)


def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	return tf.reshape(layer, [-1, num_features])


def fully_connected_layer(inputs, num_outputs, relu=True):
	num_inputs = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[num_inputs, num_outputs],
		mean=0,
		stddev=0.1
	))
	biases = tf.Variable(tf.zeros([num_outputs]))
	layer = tf.matmul(inputs, weights) + biases
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


class VGGNet:
	def __init__(self, num_classes, learning_rate):
		x = tf.get_default_graph().get_tensor_by_name('ensemble/x:0')
		y = tf.get_default_graph().get_tensor_by_name('ensemble/y:0')
		
		# Layer 0 = Reshape: 784 -> 28x28@1
		x_img = tf.reshape(x, shape=[-1, 28, 28, 1])
		# Layer 1 = Convolution: 28x28@1 -> 28x28@32 + ReLU
		conv1 = convolution_layer(x_img, filters=32, kernel_size=3, padding='SAME')
		# Layer 2 = Pooling: 28x28@32 -> 14x14@32
		pool1 = pooling_layer(conv1, padding='SAME')
		# Layer 3 = Convolution: 14x14@32 -> 14x14@64 + ReLU
		conv2 = convolution_layer(pool1, filters=64, kernel_size=3, padding='SAME')
		# Layer 4 = Pooling: 14x14@64 -> 7x7@64
		pool2 = pooling_layer(conv2, padding='SAME')
		# Layer 5 = Convolution: 7x7@64 -> 7x7@128 + ReLU
		conv3 = convolution_layer(pool2, filters=128, kernel_size=3, padding='SAME')
		# Layer 6 = Pooling: 7x7@128 -> 4x4@128
		pool3 = pooling_layer(conv3, padding='SAME')
		# Layer 7 = Flatten: 4x4@128 -> 2048
		flat = flatten_layer(pool3)
		# Layer 8 = Fully Connected: 2048 -> 1024
		fc1 = fully_connected_layer(flat, num_outputs=1024)
		# Layer 9 = Fully Connected: 1024 -> 128
		fc2 = fully_connected_layer(fc1, num_outputs=128)
		# Layer 10 = Logits: 128 -> 10
		logits = fully_connected_layer(fc2, num_outputs=num_classes, relu=False)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
		
		self.prediction = tf.argmax(logits, axis=1)
		correct_prediction = tf.equal(self.prediction, tf.argmax(y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.predict = tf.nn.softmax(logits)


if __name__ == '__main__':
	num_inputs = 784
	num_classes = 10
	learning_rate = 0.001
	with tf.variable_scope('ensemble', reuse=tf.AUTO_REUSE) as scope:
		x = tf.placeholder(tf.float32, [None, num_inputs], name='x')
		y = tf.placeholder(tf.float32, [None, num_classes], name='y')
	model = VGGNet(num_classes, learning_rate)
