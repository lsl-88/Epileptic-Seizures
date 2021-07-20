import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


def plot_eeg(dataset):
	"""This function plots the time series EEG data of Interictal, Preictal and Ictal

	:param dataset: The
	:return: None
	"""
	# Perform scaling across the independent recordings [(x-mean)/s.d]
	b_data = dataset['interictal']
	d_data = dataset['preictal']
	e_data = dataset['ictal']

	b_data = scale(b_data, axis=1)
	d_data = scale(d_data, axis=1)
	e_data = scale(e_data, axis=1)

	# Select a random sample
	sample = np.random.randint(0, 100)

	# Initialize the time vector
	t = 23.6
	total_samp_pts = 4096
	time_vec = np.linspace(0, t, total_samp_pts)

	# Set the figure size
	fig = plt.figure(figsize=(20 ,18))
	fig.subplots(nrows=3, ncols=1)

	# Plot Interictal
	plt.subplot(3, 1, 1)
	plt.plot(time_vec, b_data[:, sample], color='green')
	plt.xlabel('Time (secs)', fontsize=18)
	plt.ylabel('Amplitude (uV)', fontsize=18)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlim([0, t])
	plt.title('Interictal', fontsize=24)

	# Plot Preictal
	plt.subplot(3, 1, 2)
	plt.plot(time_vec, d_data[:, sample], color='orange')
	plt.xlabel('Time (secs)', fontsize=18)
	plt.ylabel('Amplitude (uV)', fontsize=18)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlim([0, t])
	plt.title('Preictal', fontsize=24)

	# Plot Ictal
	plt.subplot(3, 1, 3)
	plt.plot(time_vec, e_data[:, sample], color='red')
	plt.xlabel('Time (secs)', fontsize=18)
	plt.ylabel('Amplitude (uV)', fontsize=18)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlim([0, t])
	plt.title('Ictal', fontsize=24)

	plt.tight_layout()
	plt.show()
	pass


def plot_scalograms(processed_data_dir, parameters):
	"""This function plots the scalograms from set B, D and E of the specific scale

	:param processed_data_dir: The save directory of scalograms
	:param parameters: The three parameters to optimize
	:return: None
	"""
	if processed_data_dir is None:
		processed_data_dir = 'processed_data'

	# Unpack the parameters
	window_size = parameters['window_size']
	wavelets = parameters['wavelets']
	scales = parameters['scales']
	num_scales = parameters['num_scales']

	full_data_list = glob(os.path.join(processed_data_dir, '*'))
	interictal_data_list = [single_file for single_file in full_data_list if single_file.split('\\')[-1].split('_')[0] == 'interictal']
	preictal_data_list = [single_file for single_file in full_data_list if single_file.split('\\')[-1].split('_')[0] == 'preictal']
	ictal_data_list = [single_file for single_file in full_data_list if single_file.split('\\')[-1].split('_')[0] == 'ictal']

	interictal_num = np.random.randint(0, len(interictal_data_list))
	preictal_num = np.random.randint(0, len(preictal_data_list))
	ictal_num = np.random.randint(0, len(ictal_data_list))

	interictal_data = np.load(interictal_data_list[interictal_num]).astype(float)
	preictal_data = np.load(preictal_data_list[preictal_num]).astype(float)
	ictal_data = np.load(ictal_data_list[ictal_num]).astype(float)

	# Define important parameters
	fs = 173.61
	time_pts = int(fs * window_size)

	# Define the time vector and scale vector
	time_vec = np.linspace(0, time_pts / fs, time_pts)
	scale_vec = np.linspace(0, scales, num_scales)

	# Set the figure
	fig = plt.figure(figsize=(6, 12))
	fig.subplots(nrows=3, ncols=1)

	# Plot Interictal
	plt.subplot(3, 1, 1)
	plt.pcolormesh(time_vec, scale_vec, interictal_data)
	plt.colorbar()
	plt.xlabel('Time (secs)', fontsize=18)
	plt.ylabel('Scale', fontsize=18)
	plt.xticks(np.arange(0, window_size + 0.01, step=0.5), fontsize=18)
	plt.yticks(np.arange(0, scales + 0.1, step=5), fontsize=18)
	plt.title('Interictal', fontsize=24)

	# Plot Preictal
	plt.subplot(3, 1, 2)
	plt.pcolormesh(time_vec, scale_vec, preictal_data)
	plt.colorbar()
	plt.xlabel('Time (secs)', fontsize=18)
	plt.ylabel('Scale', fontsize=18)
	plt.xticks(np.arange(0, window_size + 0.01, step=0.5), fontsize=18)
	plt.yticks(np.arange(0, scales + 0.1, step=5), fontsize=18)
	plt.title('Preictal', fontsize=24)

	# Plot Ictal
	plt.subplot(3, 1, 3)
	plt.pcolormesh(time_vec, scale_vec, ictal_data)
	plt.colorbar()
	plt.xlabel('Time (secs)', fontsize=18)
	plt.ylabel('Scale', fontsize=18)
	plt.xticks(np.arange(0, window_size + 0.01, step=0.5), fontsize=18)
	plt.yticks(np.arange(0, scales + 0.1, step=5), fontsize=18)
	plt.title('Ictal', fontsize=24)
	plt.suptitle('Wavelet - {}, Num of Scales - {}, Scales - {}'.format(wavelets, num_scales, scales), fontsize=28)
	plt.tight_layout()
	plt.show()
	pass
