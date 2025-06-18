import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import os
import time
import csv
from datetime import datetime

from class_double_well_potential import double_well

# Randomly choose another index that is not max_idx
def rnd_idx (n_live_points, max_idx):
	while True:
		random_idx = torch.randint(0, n_live_points, (1,)).item()  # Generate a random index
		if random_idx != max_idx.item():  # Ensure it's not the same as max_idx
			return random_idx

def nested_sampling_step(dw, x, U_x, U_max, dx, n_live_points, dimensions, n_correl_steps):
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

def save_configurations (dw, x_confs, conf_steps, U_max_confs, fin_energy_calls, output_dir, plot=True, mixed=False, sampled=False):
	
	for i, conf_step in enumerate(conf_steps):
		if not mixed :
			x = x_confs[i]  # Get the configuration at the current step
			U_max = U_max_confs[i]  # Get the maximum energy for the current configuration
			en_calls = fin_energy_calls[i]
		else :
			x = x_confs
			U_max = U_max_confs
			en_calls = fin_energy_calls

		
		output_file = output_dir + f"pos_step{int(conf_step)}"
		img_output_file = output_dir + f"img_step{int(conf_step)}"
		if conf_step == -1 :
			output_file = output_dir + f"pos_final"
			img_output_file = output_dir + f"img_final"
		elif conf_step == -2 :
			output_file = output_dir + f"pos_extrapolated"
			img_output_file = output_dir + f"img_extrapolated"
		if sampled :
			output_file += ".5"
			img_output_file += ".5"

		# Save the configuration to the file
		with open(output_file + ".dat", "w") as f:
			# Write the step and U_max
			f.write(f"# Step: {conf_step}\n")
			f.write(f"# U_max: {U_max.item() if torch.is_tensor(U_max) else U_max}\n")
			f.write(f"# Energy-calls: {en_calls}\n")

			# Write the tensor x
			f.write("# x (configurations):\n")
			np.savetxt(f, x.cpu().numpy(), fmt="%.6f")  # Save the tensor as a NumPy array

		# print(f"Saved configuration for step {conf_step} to {output_file}")

		# Plot the configuration
		if plot : 
			dw.plot_configuration(img_output_file, x_conf=x, U_max=U_max)  # Plot the configuration 

def save_energy_vs_time(energy_time_data, output_dir):
	"""
	Save energy vs time data to DAT file
	
	Parameters:
	-----------
	energy_time_data : list of tuples
		List containing (step, time_elapsed, U_max, acceptance_ratio, dx) tuples
	output_dir : str
		Directory to save the file
	"""
	os.makedirs(output_dir, exist_ok=True)
	
	# Create filename with timestamp
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = os.path.join(output_dir, f"energy_vs_time_{timestamp}.dat")
	
	with open(filename, 'w') as f:
		# Write header with comments
		f.write("# Energy vs Time Data\n")
		f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
		f.write("# Columns: Step, Time_Elapsed_s, U_max, Acceptance_Ratio, dx\n")
		f.write("#\n")
		
		# Write data
		for data_point in energy_time_data:
			f.write(f"{data_point[0]:<8} {data_point[1]:<12.6f} {data_point[2]:<12.6f} {data_point[3]:<12.6f} {data_point[4]:<12.6f}\n")
	print(f"\nEnergy vs time data saved to: {filename}")

def save_energy_vs_calls(energy_calls_data, output_dir):
	"""
	Save U_max vs energy function calls data to DAT file
	
	Parameters:
	-----------
	energy_calls_data : list of tuples
		List containing (step, total_energy_calls, U_max) tuples
	output_dir : str
		Directory to save the file
	"""
	os.makedirs(output_dir, exist_ok=True)
	
	# Create filename with timestamp
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = os.path.join(output_dir, f"energy_vs_calls_{timestamp}.dat")
	
	has_sampling_type = len(energy_calls_data[0]) == 4

	with open(filename, 'w') as f:
		# Write header with comments
		f.write("# Energy vs Function Calls Data\n")
		if has_sampling_type : 
			f.write("# Columns: Step, NS=0/MS=1, Total_Energy_Calls, U_max\n")
		else : 
			f.write("# Columns: Step, Total_Energy_Calls, U_max\n")
		f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
		f.write("#\n")
		
		# Write data
		for data_point in energy_calls_data:
			if has_sampling_type :
				f.write(f"{data_point[0]:<8} {data_point[1]:<8} {data_point[2]:<12} {data_point[3]:<12.6f}\n")
			else : 
				f.write(f"{data_point[0]:<8} {data_point[1]:<12}{data_point[2]:<12.6f}\n")
	
	print(f"\nEnergy vs calls data saved to: {filename}")


if __name__ == "__main__":

	# Define system parameters
	n_particles = 1  # Number of particles
	dimensions = 2  # 2D system
	n_live_points = int(2e4)  # Number of live points in nested sampling
	n_correl_steps = 5  # Number of correlation steps

	# Set device
	device = "cpu"  # Force CPU for compatibility

	# Instantiate the double-well system
	dw = double_well(n_particles=n_particles, dimensions=dimensions, device=device, eps=3., c=1., d=0.5)

	# Initialize configurations for nested sampling
	x = dw.init_conf(n_live_points, lower_bounds=[-1, -3.5], upper_bounds=[1, 3.5])

	# Compute energy for all configurations - this is n_live_points energy calls
	U_x = dw.energy(x)  

	conf_steps = [1.5e4, 2e4, 2.6e4, 3.3e4, 5.1e4, 6e4, 7e4, 8.1e4, 9.2e4, 1.5e5]
	max_steps = int(max(conf_steps))
	x_confs, U_max_confs = [], []

	U_max, max_idx = torch.max(U_x, dim=0)
	dx = 0.6

	# Initialize time and energy call tracking
	start_time = time.time()
	energy_time_data = []
	energy_calls_data = []
	fin_energy_calls = []
	total_energy_calls = 0
	
	# Recording frequency
	record_every = max(1, max_steps // 1000)
	
	for i in range(max_steps):
		rnd_i = rnd_idx(n_live_points, max_idx)
		x[max_idx] = x[rnd_i]
		U_x[max_idx] = U_x[rnd_i]

		# Modified nested sampling step that returns energy call count
		acceptance_ratio = nested_sampling_step(
			dw, x, U_x, U_max, dx, n_live_points, dimensions, n_correl_steps
		)
		# Update total energy calls
		total_energy_calls += 1
		
		if acceptance_ratio < 0.5:
			dx /= 2

		U_max, max_idx = torch.max(U_x, dim=0)

		# Record both time and energy calls data
		if i % record_every == 0 or (i + 1) in conf_steps:
			current_time = time.time()
			elapsed_time = current_time - start_time
			energy_time_data.append((i + 1, elapsed_time, U_max.item(), acceptance_ratio, dx))
			energy_calls_data.append((i + 1, total_energy_calls, U_max.item()))

		# Save configurations
		if (i + 1) in conf_steps:
			U_max_confs.append(U_max)
			x_confs.append(x.clone())
			fin_energy_calls.append(total_energy_calls)


		# Progress bar
		print(f"\rStep {i + 1} of {max_steps} ({(i/max_steps)*100:.0f}%), "
			  f"acceptance = {acceptance_ratio:.4f}, dx = {dx}, "
			  f"energy calls = {total_energy_calls}", end="")

	# Final summary
	total_time = time.time() - start_time
	print(f"\n\nTotal runtime: {total_time:.2f} seconds")
	print(f"Total energy function calls: {total_energy_calls}")
	print(f"Average energy calls per second: {total_energy_calls/total_time:.2f}")

	output_dir = "./resources/nested_sampling_configs/"
	save_configurations(dw, x_confs, conf_steps, U_max_confs, fin_energy_calls, output_dir, plot=True)
	
	# Save both tracking files
	output_dir += "metrics/"
	save_energy_vs_time(energy_time_data, output_dir)
	save_energy_vs_calls(energy_calls_data, output_dir)
	print("")