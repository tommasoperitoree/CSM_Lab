import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# Custom dataset class for configurations
class ConfigurationsDataset (Dataset) :
	"""
	A custom dataset class for loading configurations from a file.
	Args:
		filepath (str): Path to the file containing configurations.
		testfraction (float): Fraction of data to be used for testing.
		train (bool): If True, load training data; if False, load testing data.
		transform (callable, optional): Optional transform to be applied on a sample.
	"""
	def __init__(self, filepaths, testfraction, train=True, transform=None):
		self.filepaths = filepaths
		self.transform = transform
		self.train = train
		
		all_data = []
		all_cond = []

		for filepath in filepaths:
			cond_value = None
			with open(filepath, "r") as f:
				lines = f.readlines()

				for line in lines:
					if line.startswith("# U_max:"):
						cond_value = float(line.split(":")[1].strip())
						break 

				# Extract the tensor x from the file
				# Find the start of the data section, skipping header lines
				data_start_idx = 0
				for i, line in enumerate(lines):
					if not line.startswith("#"):
						data_start_idx = i
						break
					
				data = np.loadtxt(lines[data_start_idx:])
				# print(f"Data shape: {data.shape}")

			# Split the data into training and testing sets
			split_idx = int((1-testfraction) * len(data))
			current_data_portion = data[:split_idx] if self.train else data[split_idx:]
			all_data.append(current_data_portion) 

			# Create the conditioning array with the same length as data, filled with cond_value
			current_cond = np.full((len(current_data_portion), 1), cond_value, dtype=np.float32)
			all_cond.append(current_cond)

		self.data = np.concatenate(all_data, axis=0)
		self.cond = np.concatenate(all_cond, axis=0)

		self.data = torch.tensor(self.data, dtype=torch.float32).clone().detach()
		self.cond = torch.tensor(self.cond, dtype=torch.float32).clone().detach()
		
		print(f"Loaded {'train' if self.train else 'test'} data. Final shape: Data={self.data.shape}, Cond={self.cond.shape}")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		condition = self.cond[idx]
		if self.transform:
			sample = self.transform(sample)
			
		return sample, condition
	
# OLD functions implemented in the class

def extract_configuration_from_file(filepath):

	with open(filepath, "r") as f:
		lines = f.readlines()
		
		# Extract the tensor x from the file
		x_start_idx = next(i for i, line in enumerate(lines) if line.startswith("# x")) + 1
		x_data = np.loadtxt(lines[x_start_idx:])  # Load the tensor data as a NumPy array

	# Convert to PyTorch tensor and return
	return torch.tensor(x_data, dtype=torch.float32).clone().detach()

def extract_U_max_from_file(filepath):
	with open(filepath, "r") as f:
		lines = f.readlines()
		
		# Find the line containing '# U_max:'
		for line in lines:
			if line.startswith("# U_max:"):
				# Extract the value after 'U_max:'
				return float(line.split(":")[1].strip())
	
	# Raise an error if '# U_max:' is not found
	raise ValueError(f"U_max not found in file: {filepath}")
