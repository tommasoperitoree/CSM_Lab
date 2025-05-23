import torch
import numpy

from class_dataset import ConfigurationsDataset
from train import train, plot_loss
from nested_smpl import nested_sampling_step, rnd_idx, save_configurations
from class_double_well_potential import double_well
from sampling import sampling


if __name__ == "__main__":

	### FLAGS FOR BEHAVIOR ###
	training_conditioning = True
	all_live_samples = False
	extrapolate = True

	if not training_conditioning : extrapolate = False

	n_mixed_routine_steps = 6
	max_live_samples_for_training = 3

	### Define system parameters
	n_particles = 1  							# Number of particles
	dimensions = 2  							# 2D system
	n_live_points = int(1e4)  					# Number of live points in nested sampling
	# to make higher to explore the energy surface with more fine grane 

	n_correl_steps = 5 							# Number of correlation steps
	n_nested_sampl_steps = [int(4e3/(i+1)) 		# Number of nested sampling steps
							for i in range(n_mixed_routine_steps)]

	### Define training parameters
	smpl_factor = 2 							# Sampling factor for the dataset
	sample_num = int(n_live_points*smpl_factor) # Number of live points to generate on sample
	noise_std = 0.5 							# Standard deviation of noise
	test_fraction = 0.1 
	batch_size = 128
	max_epochs = 50
	diffusion_steps = 500
	min_beta = 1e-4
	max_beta = 0.02
	init_learning_rate = 1e-3


	# set device (CUDA, MPS, or CPU)
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	#print(f"Using {device} device")

	# Instantiate the double-well system
	dw = double_well(n_particles=n_particles, dimensions=dimensions, device="cpu", eps=3., c=1., d=0.5)
	# Initialize configurations for nested sampling
	x = dw.init_conf(n_live_points, lower_bounds=[-1, -3.5], upper_bounds=[1, 3.5])
	# Compute energy for all configurations
	U_x = dw.energy(x)

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
	
		
	print(f"\n\n Starting Mixed Routine schedule with : \n\tn_live_points = {n_live_points} \n\tconditioning = {training_conditioning} \n\tall_live_samples = {all_live_samples} \n\tn_mixed_routine_steps = {n_mixed_routine_steps} \n\tn_nested_sampl_steps = {n_nested_sampl_steps} \n\textrapolate = {extrapolate}")

	start_index = 0

	for routine_step in range(n_mixed_routine_steps) :

		print(f"\n ----- Starting step of mixed routine #{routine_step+1} -----")


		### Nested Sampling segment 
		print(f"\n__ Nested Sampling segment - routine step #{routine_step+1} __")
		
		for i in range(int(n_nested_sampl_steps[routine_step])) :
			rnd_i = rnd_idx(n_live_points, max_idx)  # Get a random index that is not max_idx
			x[max_idx] = x[rnd_i]  # Replace the configuration with the one at random_idx
			U_x[max_idx] = U_x[rnd_i]

			acceptance_ratio = nested_sampling_step(dw, x, U_x, U_max, dx, n_live_points, dimensions, n_correl_steps) 
			if acceptance_ratio < 0.5 : dx /= 2

			U_max, max_idx = torch.max(U_x, dim=0)  # Get the maximum energy and its index
			
			# Print progress bar
			print(f"\rStep {i + 1} of {int(n_nested_sampl_steps[routine_step])} ({(i/n_nested_sampl_steps[routine_step])*100:.0f}%), acceptance = {acceptance_ratio:.4f}, dx = {dx}", end="")

		print("")
		routine_steps.append(routine_step+1)

		save_configurations(dw, x, [routine_step+1], U_max, dir_prefix, plot=True, mixed=True)


		### Training segment
		print(f"\n__ Training segment - routine step #{routine_step+1} __")
		
		print("using all samples" if all_live_samples else f"using last {max_live_samples_for_training} samples")
		all_generated_filepaths.append(dir_prefix + f"pos_step{int(routine_step+1)}.dat")

		if not all_live_samples :
			start_index = max(0, len(all_generated_filepaths) - max_live_samples_for_training)
		train_data_filepaths = all_generated_filepaths[start_index:]

		train_data  = ConfigurationsDataset(train_data_filepaths, test_fraction, train=True, cond=training_conditioning)
		test_data = ConfigurationsDataset(train_data_filepaths, test_fraction, train=False, cond=training_conditioning)

		output_model_path = model_dir_prefix + f"step{routine_steps[-1]}.pth"

		loss = train(
			train_data,
			test_data,
			batch_size,
			device,
			max_epochs,
			diffusion_steps,
			min_beta,
			max_beta,
			init_learning_rate,
			output_model_path,
		)
		# Plot and save the loss
		plot_loss(loss, dir_prefix, int(routine_step+1), mixed=True, all_ls=all_live_samples, cond=training_conditioning)


		### Sampling segment
		print(f"\n__ Sampling segment - routine step #{routine_step+1} __")
		z = torch.randn(diffusion_steps, sample_num, dimensions)
		z[-1] = 0

		sampled_x_trajectory = sampling(output_model_path, z, sample_num, diffusion_steps, min_beta, max_beta, cond=training_conditioning)
		#print(f"Shape of sampled_x: {sampled_x_trajectory.shape}")
		final_step_samples = sampled_x_trajectory[0, :, :]
		#print(f"Shape of final_step_samples: {final_step_samples.shape}")
		save_configurations(dw, final_step_samples, [routine_step+1], U_max, dir_prefix, plot=True, mixed=True, sampled=True)


		### Using the model-generated data to progress the sampling algorithm
		
		while final_step_samples.shape[0] > 0 : 
			idx_to_check = torch.randint(0, final_step_samples.shape[0], (1,)).item()
			new_sample = final_step_samples[idx_to_check]
			U_new_sample = dw.energy(new_sample)

			if U_new_sample.item() < U_max.item():
				# print(f"Sample at index {idx_to_check} with energy {U_new_sample.item():.4f} < U_max ({U_max.item():.4f}). Replacing live point.")
				x[max_idx] = new_sample
				U_x[max_idx] = U_new_sample
			
			U_max, max_idx = torch.max(U_x, dim=0)  # Get the maximum energy and its index
			print(f"\rUsed sample {int(sample_num - final_step_samples.shape[0])} of {int(sample_num)} ({((sample_num - final_step_samples.shape[0])/sample_num)*100:.0f}%) ", end=" ")

			final_step_samples = torch.cat((final_step_samples[:idx_to_check], final_step_samples[idx_to_check+1:]), dim=0)
		
		U_x = dw.energy(x)
		U_max, max_idx = torch.max(U_x, dim=0)
		print(f"\nAt the end of step #{routine_step+1}, reached energy of U={U_max}")

	print("")

	if extrapolate and training_conditioning :
		norm_U_to_sample = -0.1
		U_to_sample = train_data.denormalize(norm_U_to_sample, new_cond_min=U_max)
		print(f"[Extrapolation] norm_U_to_sample: {norm_U_to_sample}")
		print(f"[Extrapolation] U_to_sample (target energy): {U_to_sample}")
		print(f"[Extrapolation] Model trained up to energy: {U_max}")
		print(f"\nTrying to sample from model trained at energy = {U_max}, asking for energy = {U_to_sample}")
		z = torch.randn(diffusion_steps, sample_num, dimensions)
		z[-1] = 0
		sampled_extrapolated_trajectory = sampling(output_model_path, z, sample_num, diffusion_steps, min_beta, max_beta, U_max=norm_U_to_sample, cond=training_conditioning)
		extrapolated_sample = sampled_extrapolated_trajectory[0, :, :]
		save_configurations(dw, extrapolated_sample, [-2], U_to_sample.clone().detach(), dir_prefix, plot=True, mixed=True)
	
	else : 
		if extrapolate :
			print("\nCareful, no extrapolation possible without conditioned training")
		save_configurations(dw, x, [-1], U_max, dir_prefix, plot=True, mixed=True)