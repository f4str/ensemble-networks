import time
import numpy as np
import tensorflow as tf
import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Ensemble:
	"""
	average logits
	include l2 difference in loss
	"""
	def __init__(self, num_classifiers=3, learning_rate=0.001, early_stopping=True, patience=4):
		self.sess = tf.Session()
		self.early_stopping = early_stopping
		self.patience = patience
		
		self._build(num_classifiers, learning_rate)
	
	def _build(self, num_classifiers, learning_rate):
		# inputs
		self.X = tf.placeholder(tf.float32, [None, 28, 28])
		self.y = tf.placeholder(tf.int32, [None])
		one_hot_y = tf.one_hot(self.y, 10)
		
		networks = [layers.feedforward(self.X) for _ in range(num_classifiers)]
		self.individual_loss = [layers.loss(net, one_hot_y) for net in networks]
		self.individual_accuracy = [layers.accuracy(net, one_hot_y) for net in networks]
		
		logits = tf.reduce_mean(tf.stack(networks, axis=-1), axis=-1)
		l2_distance = tf.add_n([
			tf.norm(networks[0] - networks[1]),
			tf.norm(networks[1] - networks[2]),
			tf.norm(networks[2] - networks[0])
		])
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
		self.loss = tf.reduce_mean(cross_entropy) + 1e-4 * l2_distance
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.train_op = optimizer.minimize(self.loss)
		
		correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(one_hot_y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.prediction = tf.argmax(logits, axis=1)
	
	def fit(self, X, y, epochs=100, batch_size=128, validation_split=0.2, verbose=True):
		# shuffle input data
		p = np.random.permutation(len(X))
		X = np.array(X)[p]
		y = np.array(y)[p]
		
		# split into training and validation sets
		valid_size = int(validation_split * len(X))
		train_size = len(X) - valid_size
		
		dataset = tf.data.Dataset.from_tensor_slices((X, y))
		train_dataset = dataset.skip(valid_size).shuffle(train_size, reshuffle_each_iteration=True).batch(batch_size)
		valid_dataset = dataset.take(valid_size).batch(batch_size)
		
		# create batch iterator
		train_iterator = train_dataset.make_initializable_iterator()
		valid_iterator = valid_dataset.make_initializable_iterator()
		
		X_train, y_train = train_iterator.get_next()
		X_valid, y_valid = valid_iterator.get_next()
		
		total_train_loss = []
		total_train_acc = []
		total_valid_loss = []
		total_valid_acc = []
		best_acc = 0
		no_acc_change = 0
		
		self.sess.run(tf.global_variables_initializer())
		
		for e in range(epochs):
			# initialize training batch iterator
			self.sess.run(train_iterator.initializer)
			
			if verbose:
				start = time.time()
				print(f'epoch {e + 1} / {epochs}:')
			
			# train on training data
			total = 0
			train_loss = 0
			train_acc = 0
			try:
				while True:
					X_batch, y_batch = self.sess.run([X_train, y_train])
					size = len(X_batch)
					
					_, loss, acc = self.sess.run(
						[self.train_op, self.loss, self.accuracy], 
						feed_dict={self.X: X_batch, self.y: y_batch}
					)
					train_loss += loss * size
					train_acc += acc * size
					
					if verbose:
						current = time.time()
						total += size
						print(f'[{total} / {train_size}] - {(current - start):.2f} s -', 
							f'train loss = {(train_loss / total):.4f},',
							f'train acc = {(train_acc / total):.4f}',
							end='\r'
						)
			except tf.errors.OutOfRangeError:
				pass
			
			train_loss /= train_size
			train_acc /= train_size
			total_train_loss.append(train_loss)
			total_train_acc.append(train_acc)
			
			# initialize validation batch iterator
			self.sess.run(valid_iterator.initializer)
			
			# test on validation data
			valid_loss = 0
			valid_acc = 0
			try:
				while True:
					X_batch, y_batch = self.sess.run([X_valid, y_valid])
					size = len(X_batch)
					
					loss, acc = self.sess.run(
						[self.loss, self.accuracy], 
						feed_dict={self.X: X_batch, self.y: y_batch}
					)
					valid_loss += loss * size
					valid_acc += acc * size
			except tf.errors.OutOfRangeError:
				pass
			
			valid_loss /= valid_size
			valid_acc /= valid_size
			total_valid_loss.append(valid_loss)
			total_valid_acc.append(valid_acc)
			
			if verbose:
				end = time.time()
				print(f'[{total} / {train_size}] - {(end - start):.2f} s -',
					f'train loss = {train_loss:.4f},',
					f'train acc = {train_acc:.4f},',
					f'valid loss = {valid_loss:.4f},',
					f'valid acc = {valid_acc:.4f}'
				)
			
			# early stopping
			if self.early_stopping:
				if valid_acc > best_acc:
					best_acc = valid_acc
					no_acc_change = 0
				else:
					no_acc_change += 1
				
				if no_acc_change >= self.patience:
					if verbose:
						print('early stopping')
					break
		
		return total_train_loss, total_train_acc, total_valid_loss, total_valid_acc
	
	def evaluate(self, X, y):
		loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={self.X: X, self.y: y})
		return loss, acc
	
	def evaluate_individuals(self, X, y):
		for idx, (network_loss, network_acc) in enumerate(zip(self.individual_loss, self.individual_accuracy)):
			loss, acc = self.sess.run([network_loss, network_acc], feed_dict={self.X: X, self.y: y})
			print(f'network {idx + 1}: test loss = {loss:.4f}, test acc = {acc:.4f}')
	
	def predict(self, X):
		y_pred = self.sess.run(self.prediction, feed_dict={self.X: X})
		return y_pred


if __name__ == '__main__':
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	X_train = X_train.astype(np.float32) / 255
	X_test = X_test.astype(np.float32) / 255
	
	model = Ensemble()
	model.fit(X_train, y_train, epochs=10)
	loss, acc = model.evaluate(X_test, y_test)
	print(f'ensemble: test loss = {loss:.4f}, test acc = {acc:.4f}')
	
	model.evaluate_individuals(X_test, y_test)
