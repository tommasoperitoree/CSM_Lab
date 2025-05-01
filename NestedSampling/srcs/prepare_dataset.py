import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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


# Custom dataset class for configurations
class ConfigurationsDataset (Dataset) :

	def __init__(self, filepath, testfraction, train=True, transform=None):
		self.filepath = filepath
		self.transform = transform
		self.train = train

		with open(filepath, "r") as f:
			lines = f.readlines()
			
			for line in lines:
				if line.startswith("# U_max:"):
					# Extract the value after 'U_max:'
					self.U_max = float(line.split(":")[1].strip())

			# Extract the tensor x from the file
			# data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("# x")) + 1
			data = np.loadtxt(lines, skiprows=3)  # Load the tensor data as a NumPy array
			# print(f"Data shape: {data.shape}")

		# Split the data into 90% training and 10% testing
		split_idx = int((1-testfraction) * len(data))
		if self.train:
			self.data = data[:split_idx]  # First 90% for training
		else:
			self.data = data[split_idx:]  # Last 10% for testing

		# Convert to PyTorch tensor
		self.data = torch.tensor(self.data, dtype=torch.float32).clone().detach()
		#print(f"Data shape after split: {self.data.shape}")	
		#print(f"single data element shape: {self.data[0].shape}")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]