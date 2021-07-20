import os
import shutil
import numpy as np
import pandas as pd
from data import BonnDataset
from process import create_scalogram, data_split
from model import ResNet, DataGenerator
import argparse
import pywt


PATH = '.\\'

SCALES_RANGE = np.arange(5, 71, 5)
NUM_SCALES_RANGE = np.arange(60, 180, 10)
WS_RANGE = np.arange(1.0, 3.01, 0.5)
WAVELETS = pywt.wavelist(kind='continuous')

SEARCH_RANGE = 10

BATCH_SIZE = 32
SCALO_DIR = '.\\processed_data'


def main(parameters, hp_search):

	# Load the raw EEG data
	dataset = BonnDataset(PATH).load()

	# Generate the scalogram
	create_scalogram(dataset, parameters, SCALO_DIR)

	# Split the data
	(X_train, X_val, X_test, Y_train, Y_val, Y_test) = data_split(scalo_dir=SCALO_DIR)
	print('X_train: ', X_train)
	print('X_val: ', X_val)
	print('X_test: ', X_test)
	print('Y_train: ', Y_train)
	print('Y_val: ', Y_val)
	print('Y_test: ', Y_test)

	# Initialize the hyperparameters search model
	res_net_instance = ResNet(X_train, scalo_dir=SCALO_DIR)
	train_generator = DataGenerator(x=X_train, y=Y_train, scalo_dir=SCALO_DIR, batch_size=BATCH_SIZE, shuffle=True)
	val_generator = DataGenerator(x=X_val, y=Y_val, scalo_dir=SCALO_DIR, batch_size=BATCH_SIZE, shuffle=True)
	test_generator = DataGenerator(x=X_test, y=Y_test, scalo_dir=SCALO_DIR, batch_size=BATCH_SIZE, shuffle=True)

	if hp_search:
		df = {}
		summary_df = pd.DataFrame(columns=['Wavelet', 'Scales', 'Num of Scales', 'Window Size', 'Train Acc', 'Val Acc', 'Test Acc'])

		model = res_net_instance.hyperparameters_search_model()
		hist = model.fit(x=train_generator, epochs=5, verbose=2, validation_data=val_generator)
		test_results = model.evaluate(x=test_generator, verbose=1)

		train_acc = hist.history['categorical_accuracy']
		val_acc = hist.history['val_categorical_accuracy']
		test_acc = test_results[1]

		df['Wavelet'] = parameters['wavelets']
		df['Scales'] = parameters['scales']
		df['Num of Scales'] = parameters['num_scales']
		df['Window Size'] = parameters['window_size']
		df['Train Acc'] = train_acc
		df['Val Acc'] = val_acc
		df['Test Acc'] = test_acc
		summary_df = summary_df.append(df, ignore_index=True)
		summary_df.to_csv('hyperparameters_search_summary.csv')
	else:
		model = res_net_instance.full_model()
		model.fit(x=DataGenerator(x=X_train, y=Y_train), epochs=20, verbose=2, validation_data=DataGenerator(x=X_val, y=Y_val))
		model.evaluate(x=DataGenerator(x=X_test, y=Y_test), verbose=1)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train the model')
	parser.add_argument('-rs', '--random_search', required=False, action='store_true', help='Specify whether to do a random search of the hyperparameters')
	parser.add_argument('-s', '--scales', required=False, help='Scales', type=int)
	parser.add_argument('-n', '--num_scales', required=False, help='Number of scales', type=int)
	parser.add_argument('-ws', '--window_size', required=False, help='Size of window', type=float)
	parser.add_argument('-w', '--wavelets', required=False, help='Wavelets')
	args = parser.parse_args()

	if args.random_search:
		import random
		for i in range(SEARCH_RANGE):

			if os.path.exists(SCALO_DIR):
				shutil.rmtree(SCALO_DIR, ignore_errors=True)
			os.mkdir(SCALO_DIR)

			scales = random.choice(SCALES_RANGE)
			num_scales = random.choice(NUM_SCALES_RANGE)
			window_size = random.choice(WS_RANGE)
			wavelet = random.choice(WAVELETS)

			print('Parameters Specified in search attempt {}: \n'.format(i+1))
			print('---> Scales: {}'.format(scales))
			print('---> Num of Scales: {}'.format(num_scales))
			print('---> Window Size: {}'.format(window_size))
			print('---> Wavelet: {}\n'.format(wavelet))

			parameters = {'scales': scales, 'num_scales': num_scales, 'window_size': window_size, 'wavelets': wavelet}

			main(parameters, hp_search=True)
	else:
		scales = args.scales
		num_scales = args.num_scales
		window_size = args.window_size
		wavelet = args.wavelets

		assert scales in SCALES_RANGE, 'Scales specified is out of range. Range is {} to {}' \
			.format(SCALES_RANGE[0], SCALES_RANGE[1])
		assert num_scales in NUM_SCALES_RANGE, 'Number of scales specified is out of range. Range is {} to {}' \
			.format(NUM_SCALES_RANGE[0], NUM_SCALES_RANGE[-1])
		assert window_size in WS_RANGE, 'Window size specified is out of range. Range is {} to {}' \
			.format(WS_RANGE[0], WS_RANGE[-1])
		assert wavelet in WAVELETS, 'Select the correct wavelet. Available wavelets: {}'.format(WAVELETS)

		print('Parameters Specified: \n')
		print('---> Scales: {}'.format(scales))
		print('---> Num of Scales: {}'.format(num_scales))
		print('---> Window Size: {}'.format(window_size))
		print('---> Wavelet: {}\n'.format(wavelet))

		if os.path.exists(SCALO_DIR):
			shutil.rmtree(SCALO_DIR, ignore_errors=True)
		os.mkdir(SCALO_DIR)

		# Initialize the signal processing parameters
		parameters = {'scales': scales, 'num_scales': num_scales, 'window_size': window_size, 'wavelets': wavelet}

		main(parameters, hp_search=False)
