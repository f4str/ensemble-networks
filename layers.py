import tensorflow as tf


def linear(inputs, num_outputs, bias=True):
	num_inputs = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[num_inputs, num_outputs],
		mean=0,
		stddev=0.1
	))
	layer = tf.matmul(inputs, weights)
	
	if bias:
		biases = tf.Variable(tf.zeros([num_outputs]))
		layer = tf.add(layer, biases)
	
	return layer

def conv2d(inputs, filters, kernel_size=(5, 5), stride=1, padding='VALID', bias=True):
	channels = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[*kernel_size, channels, filters],
		mean=0,
		stddev=0.1
	))
	layer = tf.nn.conv2d(
		inputs,
		filter=weights,
		strides=[1, stride, stride, 1],
		padding=padding
	)
	
	if bias:
		biases = tf.Variable(tf.zeros([filters]))
		layer = tf.add(layer, biases)
	
	return layer


def maxpool2d(inputs, kernel_size=(2, 2), stride=None, padding='VALID'):
	stride = stride or kernel_size
	layer = tf.nn.max_pool2d(
		inputs,
		ksize=[1, *kernel_size, 1],
		strides=[1, *stride, 1],
		padding=padding
	)
	
	return layer


def flatten(inputs):
	layer_shape = inputs.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer = tf.reshape(inputs, [-1, num_features])
	
	return layer


def feedforward(X):
	# flatten: 28x28 -> 784
	flat = flatten(X)
	# linear: 784 -> 512 + relu
	linear1 = tf.nn.relu(linear(flat, 512))
	# linear: 512 -> 128 + relu
	linear2 = tf.nn.relu(linear(linear1, 128))
	# linear: 128 -> 10
	logits = linear(linear2, 10)
	
	return logits

def convolutional(X):
	# reshape: 28x28 -> 28x28@1
	reshaped = tf.reshape(X, [-1, 28, 28, 1])
	# convolution: 28x28@1 -> 24x24@16 + relu
	conv1 = tf.nn.relu(conv2d(reshaped, 16, (5, 5)))
	# max pooling: 24x24@16 -> 12x12@16
	pool1 = maxpool2d(conv1, (2, 2))
	# convolution: 12x12@16 -> 8x8@32 + relu
	conv2 = tf.nn.relu(conv2d(pool1, 32, (5, 5)))
	# max pooling: 8x8@32 -> 4x4@32
	pool2 = maxpool2d(conv2, (2, 2))
	# flatten: 4x4@32 -> 512
	flat = flatten(pool2)
	# linear: 512 -> 128 + relu
	fc1 = tf.nn.relu(linear(flat, 128))
	# linear: 120 -> 64 + relu
	fc2 = tf.nn.relu(linear(fc1, 64))
	# linear: 64 -> 10
	logits = linear(fc2, 10)
	
	return logits

def loss(logits, y):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	loss = tf.reduce_mean(cross_entropy)
	
	return loss


def accuracy(logits, y):
	prediction = tf.argmax(logits, axis=1)
	correct_prediction = tf.equal(prediction, tf.argmax(y, axis=1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	return accuracy
