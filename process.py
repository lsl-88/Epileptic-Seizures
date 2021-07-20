import numpy as np
import os
import pywt
import sklearn
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.preprocessing import scale


def create_scalogram(dataset, parameters, scalo_dir):
	"""Generate the scalograms and saves the scalogram data.

	:param dataset: Data from Set B, D or E
	:param parameters: The four parameters to optimize
	:param scalo_dir: The save directory for the scalograms
	"""
	# Initialize the parameters
	fs = 173.61
	time_step = 0.2

	# Unpack the parameters
	scales = parameters['scales']
	num_scales = parameters['num_scales']
	window_size = parameters['window_size']
	wavelet = parameters['wavelets']

	# Convert the window size in seconds to sampling points
	window_size = window_size * fs

	for condition, single_dataset in dataset.items():

		print('Creating Scalograms for {} data ...\n'.format(condition))
		# Perform scaling across the independent recordings [(x-mean)/s.d]
		data = scale(single_dataset, axis=1)

		num_samples = data.shape[1]  # Number of recordings
		samples_count = 0

		for j in range(num_samples):

			# Initialize time
			t = 0

			while data[:, j].shape[0] - (t * fs + window_size) > 0:

				samples_count += 1

				# Set the start sampling points
				start = int(np.floor(t * fs))

				# Set the end sampling points
				stop = int(np.floor(start + window_size))

				# Perform the actual wavelet transform
				frequencies = pywt.scale2frequency(wavelet, np.linspace(1, scales, num_scales)) / (1 / fs)
				signal, _ = pywt.cwt(data[start:stop, j], frequencies, wavelet)

				np.save(os.path.join(scalo_dir, '{}_{}'.format(condition, samples_count+1)), signal)

				# Increase the time by the time step
				t = round(t + time_step, 1)
	pass


# def obtain_data_shape():
# 	assert os.path.isfile('ictal_dataset.h5'), 'No such files exists. Need to process the raw dataset first.'
# 	with h5py.File('ictal_dataset.h5', 'r') as hf:
# 		data_shape = hf['ictal'][:].shape
# 	return data_shape


# def create_scalogram(dataset, parameters):
# 	"""Generate the scalograms and saves the scalogram data.
#
# 	:param dataset: Data from Set B, D or E
# 	:param parameters: The four parameters to optimize
# 	"""
# 	# Initialize the parameters
# 	fs = 173.61
# 	time_step = 0.2
#
# 	# Unpack the parameters
# 	scales = parameters['scales']
# 	num_scales = parameters['num_scales']
# 	window_size = parameters['window_size']
# 	wavelet = parameters['wavelets']
#
# 	# Convert the window size in seconds to sampling points
# 	window_size = window_size * fs
#
# 	for condition, single_dataset in dataset.items():
# 		print('Creating Scalograms for {} data ...\n'.format(condition))
# 		# Perform scaling across the independent recordings [(x-mean)/s.d]
# 		data = scale(single_dataset, axis=1)
#
# 		scalo_list = []
# 		num_samples = data.shape[1]  # Number of recordings
#
# 		for j in range(num_samples):
#
# 			# Initialize time
# 			t = 0
#
# 			while data[:, j].shape[0] - (t * fs + window_size) > 0:
# 				# Set the start sampling points
# 				start = int(np.floor(t * fs))
#
# 				# Set the end sampling points
# 				stop = int(np.floor(start + window_size))
#
# 				# Perform the actual wavelet transform
# 				frequencies = pywt.scale2frequency(wavelet, np.linspace(1, scales, num_scales)) / (1 / fs)
# 				signal, _ = pywt.cwt(data[start:stop, j], frequencies, wavelet)
#
# 				scalo_list.append(signal)
#
# 				# Increase the time by the time step
# 				t = round(t + time_step, 1)
#
# 		scalo_data = np.stack(scalo_list, axis=0)
# 		scalo_data = shuffle(scalo_data)
# 		with h5py.File('{}_dataset.h5'.format(condition), 'w') as hf:
# 			hf.create_dataset(condition, data=scalo_data)
# 			if condition == 'interictal':
# 				y = np.ones(scalo_data.shape[0]) * 0
# 			elif condition == 'preictal':
# 				y = np.ones(scalo_data.shape[0]) * 1
# 			elif condition == 'ictal':
# 				y = np.ones(scalo_data.shape[0]) * 2
# 			hf.create_dataset('labels', data=y)
# 	pass


def data_split(scalo_dir):
	"""Split the data into train, validation, and test data

	:param scalo_dir: The directory to where the scalograms are stored
	:return: X_train, X_val, X_test, Y_train, Y_val, Y_test
	"""
	# Shuffle the data
	X = os.listdir(scalo_dir)

	# Initialize empty list
	Y = []
	for x in X:
		if x.split('_')[0] == 'interictal':
			Y.append(0)
		elif x.split('_')[0] == 'preictal':
			Y.append(1)
		elif x.split('_')[0] == 'ictal':
			Y.append(2)

	# Shuffle the data
	X, Y = sklearn.utils.shuffle(X, Y)

	# Split into train, validation, and test
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.4)
	X_val, X_test, Y_val, Y_test = sklearn.model_selection.train_test_split(X_test, Y_test, test_size=0.5)
	return (X_train, X_val, X_test, Y_train, Y_val, Y_test)





