import torch
import numpy as np

from torch.utils.data import Dataset

class ConfigurationsDataset(Dataset):
	"""
	A custom dataset class for loading configurations from multiple files, 
	normalizing their conditioning energy values, and preparing training/test splits.

	Args:
		filepaths (list): List of file paths to load data from.
		testfraction (float): Fraction of data to be used for testing.
		train (bool): If True, load training data; if False, load testing data.
		cond (bool): If True, return conditioning values along with samples.
		transform (callable, optional): Optional transform to apply to samples.
	"""
	def __init__(self, filepaths, testfraction, train=True, cond=True, transform=None):
		self.filepaths = filepaths
		self.transform = transform
		self.train = train
		self.cond = cond

		all_data = []
		all_conds = []
		unnormalized_conds = []

		# First pass: collect all unnormalized conditioning values
		file_cond_map = {}
		for filepath in filepaths:
			with open(filepath, "r") as f:
				for line in f:
					if line.startswith("# U_max:"):
						cond_value = float(line.split(":")[1].strip())
						unnormalized_conds.append(cond_value)
						file_cond_map[filepath] = cond_value
						break

		# Normalize the conditioning values to [0, 1]
		cond_min = min(unnormalized_conds)
		cond_max = max(unnormalized_conds)
		self.cond_min = cond_min
		self.cond_max = cond_max
		# if train : print(f"[ConfigurationsDataset] cond_min (best energy): {cond_min:.4f}, cond_max (worst energy): {cond_max:.4f}")
		self.cond_range = cond_max - cond_min

		# Second pass: load data and assign normalized conditions
		for filepath in filepaths:
			norm_cond_value = self.normalize(file_cond_map[filepath])
			with open(filepath, "r") as f:
				lines = f.readlines()

			# Find where the data starts
			data_start_idx = next(i for i, line in enumerate(lines) if not line.startswith("#"))
			data = np.loadtxt(lines[data_start_idx:])

			# Train/test split
			split_idx = int((1 - testfraction) * len(data))
			current_data_portion = data[:split_idx] if self.train else data[split_idx:]
			all_data.append(current_data_portion)

			# Create the normalized condition array
			current_cond = np.full((len(current_data_portion), 1), norm_cond_value, dtype=np.float32)
			all_conds.append(current_cond)

		self.data = torch.tensor(np.concatenate(all_data, axis=0), dtype=torch.float32).clone().detach()
		self.conds = torch.tensor(np.concatenate(all_conds, axis=0), dtype=torch.float32).clone().detach()
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		condition = self.conds[idx]
		if self.transform:
			sample = self.transform(sample)
		return (sample, condition) if self.cond else sample

	def normalize(self, u):
		return (u - self.cond_min) / self.cond_range if self.cond_range != 0 else 0.0

	def denormalize(self, u_norm, new_cond_min):
	    return u_norm * (self.cond_max - new_cond_min) + new_cond_min
	
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

def denormalize(max_u, min_u, u) :
	return u * (max_u-min_u) + min_u
