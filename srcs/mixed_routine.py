import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from class_dataset import ConfigurationsDataset
from simple_nn import SimpleNN
from train import train
from nested_smpl import nested_sampling_step
from class_double_well_potential import double_well

if __name__ == "__main__":

	# Define system parameters
	n_particles = 1  # Number of particles
	dimensions = 2  # 2D system
	n_live_points = int(1e4)  # Number of live points in nested sampling
	n_correl_steps = 5  # Number of correlation steps
	sampling_factor = int(1e3) # Sampling factor for the dataset (generating n_live_points*sampling_factor)

	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

	# Instantiate the double-well system
	dw = double_well(n_particles=n_particles, dimensions=dimensions, device=device, eps=3., c=1., d=0.5)

	# Initialize configurations for nested sampling
	x = dw.init_conf(n_live_points, lower_bounds=[-1, -3.5], upper_bounds=[1, 3.5])

	# Compute energy for all configurations
	U_x = dw.energy(x)

	