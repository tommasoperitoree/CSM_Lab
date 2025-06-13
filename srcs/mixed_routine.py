import torch
import numpy
import time 
from datetime import datetime

from class_dataset import ConfigurationsDataset, extract_configuration_from_file
from train import train, plot_loss
from nested_smpl import nested_sampling_step, rnd_idx, save_configurations, save_energy_vs_calls, save_energy_vs_time
from class_double_well_potential import double_well
from sampling import sampling, histo_comparison, mean_displ_split


if __name__ == "__main__":

	### FLAGS FOR BEHAVIOR ###
	training_conditioning = True
	all_live_samples = True
	extrapolate = True

	if not training_conditioning : extrapolate = False

	n_mixed_routine_steps = 5
	max_live_samples_for_training = 3

	### Define system parameters
	n_particles = 1  							# Number of particles
	dimensions = 2  							# 2D system
	n_live_points = int(2e4)  					# Number of live points in nested sampling
	mean_disp_understeps = 25
	# to make higher to explore the energy surface with more fine grane 

	n_correl_steps = 5 							# Number of correlation steps
	n_nested_sampl_steps = [int(1.5e4/(i+1)) 		# Number of nested sampling steps
							for i in range(n_mixed_routine_steps)]

	### Define training parameters
	smpl_factor = 1								# Sampling factor for the dataset
	sample_num = int(n_live_points*smpl_factor) # Number of live points to generate on sample
	noise_std = 0.5 							# Standard deviation of noise
	test_fraction = 0.1 
	batch_size = 128
	max_epochs = 100
	diffusion_steps = 1000
	min_beta = 1e-4
	max_beta = 0.02
	learning_rate = 1e-3


	# set device (CUDA, MPS, or CPU)
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	#print(f"Using {device} device")

	# Instantiate the double-well system
	dw = double_well(n_particles=n_particles, dimensions=dimensions, device="cpu", eps=3., c=1., d=0.5)
	# Initialize configurations for nested sampling
	x = dw.init_conf(n_live_points, lower_bounds=[-1, -3.5], upper_bounds=[1, 3.5])
	# initialize from nested sampling configuration
	x = extract_configuration_from_file(f"./resources/nested_sampling_configs/pos_step{n_nested_sampl_steps[0]}.dat")	# Compute energy for all configurations
	
	U_x = dw.energy(x)
	# Initialize energy call tracking - initial evaluation
	total_energy_calls = n_live_points
	U_max, max_idx = torch.max(U_x, dim=0)  # Get the maximum energy and its index
	
	dx = 0.6

	routine_steps = []
	all_generated_filepaths = []

	# preparing output files prefix
	dir_prefix = "./resources/mixed_routine/"
	model_dir_prefix = "./trained/"
	if all_live_samples :
		dir_add = "all_live_samples/"
		if training_conditioning :
			dir_add += "conditioned/"
		else :
			dir_add += "unconditioned/"
	else :
		dir_add = f"last_live_samples/"
		if training_conditioning :
			dir_add += "conditioned/"
		else :
			dir_add += "unconditioned/"
	dir_prefix += dir_add
	model_dir_prefix += dir_add
	
	# Initialize time and energy call tracking
	start_time = time.time()
	energy_time_data = []
	energy_calls_data = []
	    
	# Initial data point
	energy_time_data.append((0, 0.0, U_max.item(), 0.0, dx))  # Step 0, time 0, initial U_max, no acceptance yet
	energy_calls_data.append((0, total_energy_calls, U_max.item()))
        
		
	print(f"\n\n Starting Mixed Routine schedule with : \n\tn_live_points = {n_live_points} \n\tconditioning = {training_conditioning} \n\tall_live_samples = {all_live_samples} \n\tn_mixed_routine_steps = {n_mixed_routine_steps} \n\tn_nested_sampl_steps = {n_nested_sampl_steps} \n\textrapolate = {extrapolate}")

	start_index = 0
	save_configurations(dw, x, [1], U_max, dir_prefix, plot=True, mixed=True)
	routine_steps.append(1)

	# Global step counter for tracking
	global_step = 0

	for routine_step in range(n_mixed_routine_steps) :

		print(f"\n ----- Starting step of mixed routine #{routine_step+1} -----")


		### Nested Sampling segment 
		print(f"\n__ Nested Sampling segment - routine step #{routine_step+1} __")
		
		if routine_step != 0 :
			for i in range(int(n_nested_sampl_steps[routine_step])) :
				global_step += 1

				rnd_i = rnd_idx(n_live_points, max_idx)  # Get a random index that is not max_idx
				x[max_idx] = x[rnd_i]  # Replace the configuration with the one at random_idx
				U_x[max_idx] = U_x[rnd_i]

				acceptance_ratio, step_energy_calls = nested_sampling_step(dw, x, U_x, U_max, dx, n_live_points, dimensions, n_correl_steps) 
				
				# Update total energy calls
				total_energy_calls += step_energy_calls
                
				if acceptance_ratio < 0.5 : dx /= 2

				U_max, max_idx = torch.max(U_x, dim=0)  # Get the maximum energy and its index

				# Record data periodically (every 100 steps or so to avoid too much data)
				if i % 100 == 0 or i == int(n_nested_sampl_steps[routine_step]) - 1:
					current_time = time.time()
					elapsed_time = current_time - start_time
					energy_time_data.append((global_step, elapsed_time, U_max.item(), acceptance_ratio, dx))
					energy_calls_data.append((global_step, total_energy_calls, U_max.item()))
				
				# Print progress bar
				print(f"\rStep {i + 1} of {int(n_nested_sampl_steps[routine_step])} ({(i/n_nested_sampl_steps[routine_step])*100:.0f}%), acceptance = {acceptance_ratio:.4f}, dx = {dx}", end="")

			print("")
			routine_steps.append(routine_step+1)

			save_configurations(dw, x, [routine_step+1], U_max, dir_prefix, plot=True, mixed=True)
			all_generated_filepaths.append(dir_prefix + f"pos_step{int(routine_step+1)}.dat")
		else : 
			print("Configuration loaded from default configuration from nested sampling")


		### Training segment
		print(f"\n__ Training segment - routine step #{routine_step+1} __")
		
		#print("using all samples" if all_live_samples else f"using last {max_live_samples_for_training} samples")
		all_generated_filepaths.append(dir_prefix + f"pos_step{int(routine_step+1)}.dat")

		if not all_live_samples :
			start_index = max(0, len(all_generated_filepaths) - max_live_samples_for_training)
		train_data_filepaths = all_generated_filepaths[start_index:]

		train_data  = ConfigurationsDataset(train_data_filepaths, test_fraction, train=True, cond=training_conditioning)
		test_data = ConfigurationsDataset(train_data_filepaths, test_fraction, train=False, cond=training_conditioning)

		output_model_path = model_dir_prefix + f"step{routine_steps[-1]}.pth"
        
		# Record time before training
		training_start_time = time.time()
        
		loss = train(
			train_data,
			test_data,
			batch_size,
			device,
			max_epochs,
			diffusion_steps,
			min_beta,
			max_beta,
			learning_rate,
			output_model_path,
		)
		# Record time after training
		training_end_time = time.time()
		elapsed_time = training_end_time - start_time
		energy_time_data.append((global_step, elapsed_time, U_max.item(), 0.0, dx))  # No acceptance ratio for training
		energy_calls_data.append((global_step, total_energy_calls, U_max.item()))
        
		# Plot and save the loss
		plot_loss(loss, dir_prefix, int(routine_step+1), all_live_samples)


		### Sampling segment
		print(f"\n__ Sampling segment - routine step #{routine_step+1} __")
		z = torch.randn(diffusion_steps, sample_num, dimensions)
		z[-1] = 0

		# Record time before sampling
		sampling_start_time = time.time()

		sampled_x_trajectory, displacements = sampling(output_model_path, z, diffusion_steps, min_beta, max_beta, U_max=0, cond=training_conditioning)
		#print(f"Shape of sampled_x: {sampled_x_trajectory.shape}")
		final_step_samples = sampled_x_trajectory[0, :, :]
		#print(f"Shape of final_step_samples: {final_step_samples.shape}")
		save_configurations(dw, final_step_samples, [routine_step+1], U_max, dir_prefix, plot=True, mixed=True, sampled=True)

		# Record time after sampling
		sampling_end_time = time.time()
		elapsed_time = sampling_end_time - start_time
		energy_time_data.append((global_step, elapsed_time, U_max.item(), 0.0, dx))  # No acceptance ratio for sampling
		energy_calls_data.append((global_step, total_energy_calls, U_max.item()))


		# Histogram to visualize accuracy 
		# histo_path = dir_prefix + f"histo_comparison_step{routine_step+1}"
		# histo_comparison(x, final_step_samples, bin_size=0.1, save_path=histo_path)
		# Mean displacement graphics
		# disp_path = dir_prefix + f"displacement_evolution/disp_ev_step{routine_step+1}"
		# mean_displ_split(sampled_x_trajectory, displacements, bin_size=0.15, save_dir=disp_path, n_understeps=mean_disp_understeps)
		# Accuracy calculation on output file


		### Using the model-generated data to progress the sampling algorithm
		
		samples_used = 0
		while final_step_samples.shape[0] > 0 : 
			idx_to_check = torch.randint(0, final_step_samples.shape[0], (1,)).item()
			new_sample = final_step_samples[idx_to_check]
			U_new_sample = dw.energy(new_sample)

			# Count energy call
			total_energy_calls += 1
			samples_used += 1

			if U_new_sample.item() < U_max.item():
				# print(f"Sample at index {idx_to_check} with energy {U_new_sample.item():.4f} < U_max ({U_max.item():.4f}). Replacing live point.")
				x[max_idx] = new_sample
				U_x[max_idx] = U_new_sample
			
			U_max, max_idx = torch.max(U_x, dim=0)  # Get the maximum energy and its index
			print(f"\rUsed sample {int(sample_num - final_step_samples.shape[0]+1)} of {int(sample_num)} ({((sample_num - final_step_samples.shape[0])/sample_num)*100:.0f}%) ", end=" ")

			# Record data every 100 samples to avoid too much data
			if samples_used % 100 == 0:
				current_time = time.time()
				elapsed_time = current_time - start_time
				energy_time_data.append((global_step, elapsed_time, U_max.item(), 0.0, dx))
				energy_calls_data.append((global_step, total_energy_calls, U_max.item()))


			final_step_samples = torch.cat((final_step_samples[:idx_to_check], final_step_samples[idx_to_check+1:]), dim=0)
		
		U_x = dw.energy(x)
		total_energy_calls += n_live_points  # Re-evaluation of all points
		U_max, max_idx = torch.max(U_x, dim=0)

		# Record final state of this routine step
		current_time = time.time()
		elapsed_time = current_time - start_time
		energy_time_data.append((global_step, elapsed_time, U_max.item(), 0.0, dx))
		energy_calls_data.append((global_step, total_energy_calls, U_max.item()))

		print("")

	print("")

	if extrapolate and training_conditioning :
		z = torch.randn(diffusion_steps, sample_num, dimensions)
		z[-1] = 0
		normalized_u = train_data.normalize(U_max)
		sampled_extrapolated_trajectory, displacement = sampling(output_model_path, z, diffusion_steps, min_beta, max_beta, U_max=normalized_u, cond=training_conditioning)
		extrapolated_sample = sampled_extrapolated_trajectory[0, :, :]
		save_configurations(dw, extrapolated_sample, [-2], U_max.item(), dir_prefix, plot=True, mixed=True)
	
	else : 
		if extrapolate :
			print("\nCareful, no extrapolation possible without conditioned training")
		else : save_configurations(dw, x, [-1], U_max, dir_prefix, plot=True, mixed=True)

	
	# Final summary
	total_time = time.time() - start_time
	print(f"\n\nMixed Routine Summary:")
	print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
	print(f"Total energy function calls: {total_energy_calls}")
	print(f"Average energy calls per second: {total_energy_calls/total_time:.2f}")

	# Save both tracking files
	save_energy_vs_time(energy_time_data, dir_prefix)
	save_energy_vs_calls(energy_calls_data, dir_prefix)