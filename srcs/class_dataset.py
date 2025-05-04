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
	def __init__(self, filepath, testfraction, train=True, transform=None):
		self.filepath = filepath
		self.transform = transform
		self.train = train

		with open(filepath, "r") as f:
			lines = f.readlines()
			
			for line in lines:
				if line.startswith("# U_max:"):
					# Extract the value after 'U_max:' and store it as the conditioning variable
					self.cond = float(line.split(":")[1].strip())

			# Extract the tensor x from the file
			data = np.loadtxt(lines, skiprows=3)  
			# print(f"Data shape: {data.shape}")

		# Split the data into training and testing sets
		split_idx = int((1-testfraction) * len(data))
		if self.train:
			self.data = data[:split_idx]  
		else:
			self.data = data[split_idx:]  

		# Convert to PyTorch tensor
		self.data = torch.tensor(self.data, dtype=torch.float32).clone().detach()
		#print(f"Data shape after split: {self.data.shape}")	
		#print(f"single data element shape: {self.data[0].shape}")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.cond
	
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
