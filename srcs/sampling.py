import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from matplotlib.animation import FuncAnimation
import time

from forward_process import calculate_parameters
from class_dataset import extract_U_max_from_file, denormalize
from simple_nn import SimpleNN
from class_double_well_potential import double_well


def sampling(model_path, z, diffusion_steps, min_beta, max_beta, U_max=0, cond=False):
	model = SimpleNN(cond=cond)
	model.load_state_dict(torch.load(model_path, weights_only=True))
	model.eval()
	
	with torch.no_grad():
		#x_init = torch.randn(size=(sample_num, 2))
		beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
			diffusion_steps, min_beta, max_beta
		)
		denoised_x = torch.zeros(z.shape)
		# print(f"denoised_x size = {denoised_x.shape}")
		# z = torch.randn(diffusion_steps, sample_num, dimensions) for conditioning

		denoised_x[-1] = z[0]
		for t in range(diffusion_steps - 1, 0, -1):
			#if t > 1:
			#	z = torch.randn(x_init.shape)
			#else:
			#	z = 0
			ts = torch.full((z.shape[1], 1), t)
			if cond : 
				c = torch.full((z.shape[1], 1), U_max)
				mu = (
					1
					/ torch.sqrt(alpha_ts[t])
					* (
						(
							denoised_x[t]
							- (1 - alpha_ts[t])
							/ torch.sqrt(1 - bar_alpha_ts[t])
							* model.forward(denoised_x[t], ts, c)
						)
					)
				)
			else :
				mu = (
					1
					/ torch.sqrt(alpha_ts[t])
					* (
						(
							denoised_x[t]
							- (1 - alpha_ts[t])
							/ torch.sqrt(1 - bar_alpha_ts[t])
							* model.forward(denoised_x[t], ts)
						)
					)
				)
			denoised_x[t - 1] = mu + torch.sqrt(beta_ts[t]) * z[diffusion_steps-t]
			displacements = denoised_x[1:] - denoised_x[:-1]  # shape: (diffusion_steps-1, sample_num, dimensions)

	return denoised_x, displacements

def create_sampling_animation(denoised_x, save_path, duration_seconds=4., original_steps=None):
	fig, ax = plt.subplots(figsize=(6, 6))
	scatter = ax.scatter([], [], alpha=0.1, s=1)

	def init():
		ax.set_xlim(-5.5, 5.5)
		ax.set_ylim(-5.5, 5.5)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_title("Sampling")
		return (scatter,)

	# Create step map to map coarsened frame index to original diffusion step
	if original_steps is None:
		step_map = list(range(denoised_x.shape[0]))
	else:
		step_map = torch.linspace(0, original_steps - 1, steps=denoised_x.shape[0]).long().tolist()

	def update(rev_t):
		t = denoised_x.shape[0] - 1 - rev_t
		true_t = step_map[t]
		scatter.set_offsets(denoised_x[t])
		ax.set_title(f"Sampling - Step {true_t}/{original_steps}")
		return (scatter,)

	n_frames = len(denoised_x)
	fps = round(n_frames / duration_seconds)
	anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)
	anim.save(save_path, writer="pillow", fps=fps)
	plt.close(fig)

def mean_displ_split (denoised_x, displacements, bin_size, save_dir, n_understeps):

	T = denoised_x.shape[0] - 1
	segment_size = T // n_understeps

	for seg in range(n_understeps):
		start = seg * segment_size
		end = (seg + 1) * segment_size if seg < n_understeps - 1 else T

		positions = denoised_x[start:end]
		disps = displacements[start:end]

		x_min, x_max = positions[..., 0].min(), positions[..., 0].max()
		y_min, y_max = positions[..., 1].min(), positions[..., 1].max()

		x_bins = torch.arange(x_min, x_max + bin_size, bin_size)
		y_bins = torch.arange(y_min, y_max + bin_size, bin_size)

		pos_flat = positions.reshape(-1, 2)
		disp_flat = disps.reshape(-1, 2)

		x_idx = ((pos_flat[:, 0] - x_min) / bin_size).long()
		y_idx = ((pos_flat[:, 1] - y_min) / bin_size).long()

		vec_sum = torch.zeros((len(x_bins), len(y_bins), 2))
		vec_count = torch.zeros((len(x_bins), len(y_bins)))

		for i in range(pos_flat.shape[0]):
			xi, yi = x_idx[i], y_idx[i]
			if 0 <= xi < len(x_bins) and 0 <= yi < len(y_bins):
				vec_sum[xi, yi] += disp_flat[i]
				vec_count[xi, yi] += 1

		mean_disp = torch.zeros_like(vec_sum)
		mask = vec_count > 0
		mean_disp[mask] = vec_sum[mask] / vec_count[mask].unsqueeze(-1)

		fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
		X, Y = torch.meshgrid(x_bins, y_bins, indexing='ij')
		U, V = mean_disp[..., 0], mean_disp[..., 1]
		magnitude = torch.norm(mean_disp, dim=-1)
		max_magnitude = magnitude.max()

		U_norm = U / (max_magnitude + 1e-8)
		V_norm = V / (max_magnitude + 1e-8)

		Q = ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='viridis')
		cbar = fig.colorbar(Q, ax=ax)
		cbar.set_label("Displacement Magnitude")

		ax.set_xlabel("X")
		ax.set_ylabel("Y")
		ax.set_title(f"Mean Displacement (Steps {start}–{end})")

		save_path = save_dir + f"_{seg}.png"
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close(fig)


if __name__ == "__main__":

	conditioning = True
	zeros = False # test by generating not uniform but all zero distribution
	
	#conf_steps = [1.5e4]
	conf_steps = [1.5e4, 2e4, 2.6e4, 3.3e4, 5.1e4, 6e4, 7e4, 8.1e4, 9.2e4, 1.5e5]  # Number of configuration steps
	conf_steps = [int(step) for step in conf_steps]
	
	# Define system parameters
	n_particles = 1  # Number of particles
	dimensions = 2  # 2D system

	# Set device (CUDA, MPS, or CPU)
	# MPS is for Apple Silicon Macs with Metal Performance Shaders support
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	# print(f"Using {device} device")

	# Instantiate the double-well system
	dw = double_well(n_particles=n_particles, dimensions=dimensions, device=device, eps=3., c=1., d=0.5)

	sample_num = int(2e4)
	diffusion_steps = 50
	min_beta = 1e-4
	max_beta = 0.02

	norm_U_to_sample = [0.5, 0.2, 0.01]

	z = torch.randn(diffusion_steps, sample_num, dimensions)
	z[-1] = 0
	if conditioning and not zeros:
		U_max_list = []

		timing_file = f"./resources/sampling_results/conditioning/run_time_smpl_ds{diffusion_steps}.dat"
		with open(timing_file, "w") as f_time:
			f_time.write("# U_norm\tSamplingTime_seconds\n")

		print(f"\nSampling total model with diffusion steps : {diffusion_steps}, with conditions {norm_U_to_sample}")
		for i in range(len(conf_steps)):
			file_path = f"./resources/nested_sampling_configs/pos_step{conf_steps[i]}.dat"
			U_max_list.append(extract_U_max_from_file(file_path)) 
		for u in norm_U_to_sample:
			model_path = f"./trained/diffusion_model_tot.pth"
			# U_max = 0 # Conditioning variable (U_max), how should this be defined?
			start_time = time.time()
			denoised_x, displacements = sampling(model_path, z, diffusion_steps, min_beta, max_beta, u, True)
			elapsed_time = time.time() - start_time
			
			with open(timing_file, "a") as f_time:
				f_time.write(f"{u}\t\t\t{elapsed_time:.6f}\n")
			save_path = f"./resources/sampling_results/conditioning/"
			anim_path = save_path + f"smpl_anim_u={u}_ds{int(diffusion_steps)}"
			img_path = save_path + f"smpl_img_u={u}_ds{int(diffusion_steps)}"
			
			create_sampling_animation(denoised_x, anim_path + ".gif", original_steps=diffusion_steps)

			U_max = denormalize(max(U_max_list), min(U_max_list), u)
			dw.plot_configuration(
				img_path, denoised_x[0], U_max=U_max, U_max_cont=True, sampling=True
			)  # Plot the final configuration
			print(f"Sampling animation & final configuration saved for all configuration steps and input energy = {u}")
			displ_path = save_path + f"histograms/mean_disp_histo_u={u}_ds{diffusion_steps}"
			mean_displ_split(denoised_x, displacements, bin_size=0.15, save_dir=displ_path, n_understeps=10)

	elif not zeros :
		for i in range(len(conf_steps)):
			model_path = f"./trained/diffusion_model_step{conf_steps[i]}.pth"
			denoised_x = sampling(model_path, z, diffusion_steps, min_beta, max_beta, cond=False)
			save_path = f"./resources/sampling_results/steps/"
			anim_path = save_path + f"smpl_anim_step{conf_steps[i]}"
			img_path = save_path + f"smpl_img_step{conf_steps[i]}"
			create_sampling_animation(denoised_x, anim_path + ".gif")

			U_max = extract_U_max_from_file(f"./resources/nested_sampling_configs/pos_step{conf_steps[i]}.dat")

			dw.plot_configuration(
				img_path, denoised_x[0], U_max=U_max, U_max_cont=True, sampling=True
			)  # Plot the final configuration

			print(f"Sampling animation & final configuration saved for configuration step {conf_steps[i]}")
	elif zeros :
		z = torch.zeros(diffusion_steps, sample_num, dimensions)
		for u in norm_U_to_sample :
			model_path = f"./trained/diffusion_model_tot.pth"
			# U_max = 0 # Conditioning variable (U_max), how should this be defined?
			denoised_x = sampling(model_path, z, diffusion_steps, min_beta, max_beta, u, True)
			save_path = f"./resources/sampling_results/zeros/"
			anim_path = save_path + f"smpl_anim_zeros_u={u}"
			img_path = save_path + f"smpl_img_zeros_u={u}"
			create_sampling_animation(denoised_x, diffusion_steps, anim_path + ".gif")
			dw.plot_configuration(
				img_path, denoised_x[0], U_max=u, U_max_cont=True, sampling=True
			)  # Plot the final configuration
			print(f"Sampling animation & final configuration saved for all configuration steps and input energy = {u}")