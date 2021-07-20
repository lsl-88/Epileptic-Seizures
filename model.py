import os
import math
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Concatenate


class ResNet:

	def __init__(self, X, scalo_dir):
		self.input_shape = np.load(os.path.join(scalo_dir, X[0])).shape
		self.filters = 16

	def res_block(self, input_layer):
		"""Single Res block

		:param input_layer: Input layer
		:return: res_block_layer
		"""
		res_layer = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(input_layer)
		res_layer = Conv2D(filters=self.filters, kernel_size=3, strides=(1, 1), padding='same', activation='relu')(res_layer)
		res_block_layer = Concatenate()([input_layer, res_layer])
		return res_block_layer

	def hyperparameters_search_model(self):

		input_img = Input(shape=(self.input_shape[0], self.input_shape[1], 1))
		
		# Input layer
		input_layer = Conv2D(filters=self.filters, kernel_size=7, strides=(2, 2), padding='valid', activation='relu')(input_img)
		input_layer = MaxPool2D(pool_size=(3, 3))(input_layer)
		
		# 1st res block
		res_block_layer = self.res_block(input_layer=input_layer)

		# Output
		output_layer = Flatten()(res_block_layer)
		output_layer = Dense(units=3, activation='softmax')(output_layer)

		# Define model
		model = tf.keras.Model(input_img, output_layer)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='categorical_accuracy')
		return model

	def full_model(self):
		input_img = Input(shape=(self.input_shape[0], self.input_shape[1], 1))

		# Input layer
		input_layer = Conv2D(filters=self.filters, kernel_size=7, strides=(2, 2), padding='valid', activation='relu')(
			input_img)
		input_layer = MaxPool2D(pool_size=(3, 3))(input_layer)

		# 1st res block
		res_block_layer = self.res_block(input_layer=input_layer)

		# 2nd res block
		res_block_layer = self.res_block(input_layer=res_block_layer)

		# 3rd res block
		res_block_layer = self.res_block(input_layer=res_block_layer)

		# Output
		output_layer = Flatten()(res_block_layer)
		output_layer = Dense(units=3, activation='softmax')(output_layer)

		# Define model
		model = tf.keras.Model(input_img, output_layer)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='categorical_accuracy')
		return model


class DataGenerator(tf.keras.utils.Sequence):
	"""Data generator class for training."""
	def __init__(self, x, y, scalo_dir, batch_size, shuffle):

		self.batch_size = batch_size
		self.scalo_dir = scalo_dir
		if shuffle:
			self.x, self.y = sklearn.utils.shuffle(x, y)
		else:
			self.x, self.y = x, y

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		batch_x_list = []
		for x in batch_x:
			x = np.load(os.path.join(self.scalo_dir, x))
			batch_x_list.append(x)
		x_batch = np.stack(batch_x_list, axis=0)

		X = tf.cast(x_batch, tf.float32)
		Y = tf.cast(batch_y, tf.int32)
		Y = tf.one_hot(Y, depth=3, on_value=1, off_value=0)
		Y = tf.cast(Y, tf.float32)
		return X, Y
