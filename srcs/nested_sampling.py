import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import os

from double_well import double_well

# Define system parameters
n_particles = 1  # Number of particles
dimensions = 2  # 2D system
n_live_points = int(1e4)  # Number of live points in nested sampling
n_correl_steps = 5  # Number of correlation steps

# Set device (MPS is for Apple Silicon Macs with Metal Performance Shaders support)
device = "cpu"  # Force CPU for compatibility
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")


# Instantiate the double-well system
dw = double_well(n_particles=n_particles, dimensions=dimensions, device=device, eps=3., c=1., d=0.5)

# Initialize configurations for nested sampling
x = dw.init_conf(n_live_points, lower_bounds=[-1, -3.5], upper_bounds=[1, 3.5])

# Compute energy for all configurations
U_x = dw.energy(x)  


# Randomly choose another index that is not max_idx
def rnd_idx (n_live_points, max_idx):
	while True:
		random_idx = torch.randint(0, n_live_points, (1,)).item()  # Generate a random index
		if random_idx != max_idx.item():  # Ensure it's not the same as max_idx
			return random_idx

def nested_sampling_step(x, U_x, U_max, dx, n_live_points, dimensions):
	acceptance = 0  # Initialize acceptance count
	for _ in range(n_correl_steps):
		# Generate random perturbations for all configurations in a batch
		x_step = (torch.rand((n_live_points, dimensions), device=x.device) - 0.5) * dx  # Random perturbations in [-0.1, 0.1]
		
		x_new = x + x_step  # Propose updated configurations for all points
		U_new_x = dw.energy(x_new)	# Compute energies for all proposed configurations 

		# Accept configurations where the new energy is less than U_max
		mask = U_new_x < U_max  # Boolean mask for accepted configurations
		acceptance += mask.sum().item()  # Count accepted configurations

		
		U_x[mask] = U_new_x[mask]  # Update energies for accepted configurations
		x[mask] = x_new[mask]  # Update configurations for accepted configurations
		
	acceptance /= (n_live_points * n_correl_steps)  # Calculate acceptance rate
	return acceptance  # Return acceptance rate


# conf_steps = [5e3]
conf_steps = [5e3, 1e4, 1.6e4, 2.3e4, 3.1e4, 4e4, 5e4, 6.1e4, 7.3e4]  # Number of configuration steps
max_steps = int(max(conf_steps))  # Maximum number of steps
x_conf, U_max_conf = [], []

U_max, max_idx = torch.max(U_x, dim=0)  # Get the maximum energy and its index
dx = 0.6

for i in range(max_steps):
	rnd_i = rnd_idx(n_live_points, max_idx)  # Get a random index that is not max_idx
	x[max_idx] = x[rnd_i]  # Replace the configuration with the one at random_idx
	U_x[max_idx] = U_x[rnd_i]

	acceptance_ratio = nested_sampling_step(x, U_x, U_max, dx, n_live_points, dimensions) 
	if acceptance_ratio < 0.5 : dx /= 2

	U_max, max_idx = torch.max(U_x, dim=0)  # Get the maximum energy and its index

	# Save the configuration if the current step is in conf_steps
	if (i + 1) in conf_steps:
		U_max_conf.append(U_max)
		x_conf.append(x.clone())  # Save a copy of the current configuration

	# Print progress bar
	print(f"\rStep {i + 1} of {max_steps} ({(i/max_steps)*100:.0f}%), acceptance = {acceptance_ratio:.4f}, dx = {dx} | Saved configurations: {len(x_conf)}", end="")

# Print a newline after the loop to avoid overwriting the last line
print()

# Directory to save the configuration files
output_dir = "./NestedSampling/nested_sampling_configs"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist


for i, conf_step in enumerate(conf_steps):
	x = x_conf[i]  # Get the configuration at the current step
	U_max = U_max_conf[i]  # Get the maximum energy for the current configuration

	# File name for the current configuration
	output_file = os.path.join(output_dir, f"conf_step_{int(conf_step)}.dat")

	# Save the configuration to the file
	with open(output_file, "w") as f:
		# Write the step and U_max
		f.write(f"# Step: {conf_step}\n")
		f.write(f"# U_max: {U_max.item()}\n")

		# Write the tensor x
		f.write("# x (configurations):\n")
		np.savetxt(f, x.cpu().numpy(), fmt="%.6f")  # Save the tensor as a NumPy array

	# print(f"Saved configuration for step {conf_step} to {output_file}")

	# Plot the configuration
	dw.plot_configuration(x, U_max, conf_step)  # Plot the configuration 