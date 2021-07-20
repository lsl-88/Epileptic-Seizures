from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


class DatasetError(Exception):
	pass


class Dataset(ABC):
	"""
	Abstract base class representing a dataset. All datasets should subclass from this baseclass and need to implement
	the 'load' function. Initializing of the dataset and actually loading the data is separated as the latter may
	require significant time, depending on where the data is coming from. It also allows to implement different
	handlers for the remote end where the data originates, e.g. download from cloud server, etc.
	When subclassing from Dataset, it is helpful to set fields 'data_type', 'data_name', and 'data_id'.
	"""
	def __init__(self, **kwargs):
		"""Initialize a dataset."""
		self.data_name = None

	@abstractmethod
	def load(self, **kwargs):
		"""Load the data and prepare it for usage.

		:param kwargs: Arbitrary keyword arguments
		:return: self
		"""
		return self


class BonnDataset(Dataset):
	data_name = 'Bonn'

	def __init__(self, path, **kwargs):
		super(BonnDataset, self).__init__(**kwargs)
		if path is not None:
			self.path = os.path.join(path, 'Datasets')
		else:
			self.path = 'Datasets'

	def __repr__(self):
		return '{self.__class__.__name__}(path={self.path}, **kwargs)'.format(self=self)

	def load(self):

		# Set the name of the directories to the datasets
		dir_B = os.path.join(self.path, 'setB')
		dir_D = os.path.join(self.path, 'setD')
		dir_E = os.path.join(self.path, 'setE')

		# Loop through directory B and append single file to B_files
		b_file_list = sorted(os.listdir(dir_B))
		d_file_list = sorted(os.listdir(dir_D))
		e_file_list = sorted(os.listdir(dir_E))

		# Initialize empty arrays for all files
		b_full_files = np.array([]).reshape(4096, 0)
		d_full_files = np.array([]).reshape(4096, 0)
		e_full_files = np.array([]).reshape(4096, 0)

		for single_file in b_file_list:
			df = pd.read_csv(os.path.join(dir_B, single_file))
			b_full_files = np.concatenate((b_full_files, df), axis=1)

		for single_file in d_file_list:
			df = pd.read_csv(os.path.join(dir_D, single_file))
			d_full_files = np.concatenate((d_full_files, df), axis=1)

		for single_file in e_file_list:
			df = pd.read_csv(os.path.join(dir_E, single_file))
			e_full_files = np.concatenate((e_full_files, df), axis=1)

		dataset = {'interictal': b_full_files, 'preictal': d_full_files, 'ictal': e_full_files}
		return dataset

