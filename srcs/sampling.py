import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from matplotlib.animation import FuncAnimation
import time

from forward_process import calculate_parameters
from class_dataset import extract_U_max_from_file, extract_configuration_from_file
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
			displacements = denoised_x[:-1] - denoised_x[1:]  # shape: (diffusion_steps-1, sample_num, dimensions)

	return denoised_x, displacements

def create_sampling_animation(denoised_x, save_path, duration_seconds=4.0, original_steps=None):
	target_total_frames = 50  # fixed across all animations
	fps = int(target_total_frames / duration_seconds)

	# Subsample to fixed frame count
	t_indices = torch.linspace(0, denoised_x.shape[0] - 1, steps=target_total_frames).long()
	frames = denoised_x[t_indices]

	# Map frame indices back to diffusion steps
	if original_steps is None:
		step_map = t_indices.tolist()
	else:
		step_map = torch.linspace(0, original_steps - 1, steps=target_total_frames).long().tolist()

	fig, ax = plt.subplots(figsize=(6, 6))
	scatter = ax.scatter([], [], alpha=0.1, s=1)

	def init():
		ax.set_xlim(-5.5, 5.5)
		ax.set_ylim(-5.5, 5.5)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_title("Sampling")
		return (scatter,)

	def update(rev_t):
		t = frames.shape[0] - 1 - rev_t
		true_t = step_map[t]
		scatter.set_offsets(frames[t])
		ax.set_title(f"Sampling - Step {true_t}/{original_steps}")
		return (scatter,)

	anim = FuncAnimation(fig, update, frames=target_total_frames, init_func=init, blit=True)
	anim.save(save_path, writer="pillow", fps=fps)
	plt.close(fig)

def mean_displ_split (denoised_x, displacements, bin_size, save_dir, n_understeps):

	T = denoised_x.shape[0] - 1  # Total diffusion steps - 1
	segment_size = T // n_understeps  # Floor division: how many steps per segment
	remainder = T % n_understeps  # Remainder steps to add to first segment

	for seg in range(n_understeps):
		if seg == 0:
			# First segment gets the remainder steps + regular segment size
			start_reversed = 0
			end_reversed = segment_size + remainder
		else:
			# All other segments get regular segment size
			start_reversed = remainder + seg * segment_size
			end_reversed = remainder + (seg + 1) * segment_size
		
		# Convert to original indices (counting from the end)
		start = T - end_reversed
		end = T - start_reversed
		
		# Ensure we don't go below 0
		start = max(0, start)

		positions = denoised_x[1:][start:end]
		disps = displacements[start:end]

		x_min, x_max = positions[..., 0].min(), positions[..., 0].max()
		y_min, y_max = positions[..., 1].min(), positions[..., 1].max()

		x_bins = torch.arange(x_min, x_max + bin_size, bin_size)
		y_bins = torch.arange(y_min, y_max + bin_size, bin_size)

		pos_flat = positions.reshape(-1, 2)
		disp_flat = disps.reshape(-1, 2)
		x_idx = ((pos_flat[:, 0] - x_min) / bin_size).long()
		y_idx = ((pos_flat[:, 1] - y_min) / bin_size).long()

		x_idx = torch.clamp(x_idx, 0, len(x_bins) - 1)
		y_idx = torch.clamp(y_idx, 0, len(y_bins) - 1)
		
		vec_sum = torch.zeros((len(x_bins), len(y_bins), 2))
		vec_count = torch.zeros((len(x_bins), len(y_bins)))

		# checking valid bis
		valid = (x_idx >= 0) & (x_idx < len(x_bins)) & (y_idx >= 0) & (y_idx < len(y_bins))
		x_idx, y_idx, disp_flat = x_idx[valid], y_idx[valid], disp_flat[valid]
		# accumulate bins (vector form)
		vec_sum.index_put_((x_idx, y_idx), disp_flat, accumulate=True)
		vec_count.index_put_((x_idx, y_idx), torch.ones_like(x_idx, dtype=vec_count.dtype), accumulate=True)	

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
		ax.set_xlim(-5.5, 5.5)
		ax.set_ylim(-5.5, 5.5)
		ax.set_ylabel("Y")
		ax.set_title(f"Mean Displacement - Step {seg+1} (Original steps {T-end_reversed+1}–{T-start_reversed+1})")

		save_path = save_dir + f"_{seg+1}.png"
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close(fig)

def histo_comparison (x_original, denoised_x, bin_size, save_path):
	orig_positions = np.vstack(x_original)
	den_positions = np.vstack(denoised_x)

	# Set grid boundaries
	x_min, x_max = -5.5, 5.5
	y_min, y_max = -5.5, 5.5
	
	# Calculate number of bins based on bin_size
	num_bins_x = int((x_max - x_min) / bin_size)
	num_bins_y = int((y_max - y_min) / bin_size)

	# Compute 2D histogram for original data
	H_orig, xedges, yedges = np.histogram2d(
		orig_positions[:, 0],  # x positions
		orig_positions[:, 1],  # y positions
		bins=[num_bins_x, num_bins_y],
		range=[[x_min, x_max], [y_min, y_max]],
		density=True
	)
	
	# Compute 2D histogram for denoised data
	H_den, xedges, yedges = np.histogram2d(
		den_positions[:, 0],  # x positions
		den_positions[:, 1],  # y positions
		bins=[num_bins_x, num_bins_y],
		range=[[x_min, x_max], [y_min, y_max]],
		density=True
	)

	# Calculate bin-by-bin difference (original - denoised)
	H_diff = H_orig - H_den

	# Create figure with 3 subplots
	fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=180)
 	
	# Calculate adaptive color limits
	vmax_orig = H_orig.max()
	vmax_den = H_den.max()
	vmax_both = max(vmax_orig, vmax_den)  # Use same scale for both for comparison
	
	#print(f"Original max density: {vmax_orig:.6f}")
	#print(f"Denoised max density: {vmax_den:.6f}")
	# Plot 1: Original histogramMa
	im1 = axes[0].imshow(H_orig.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
						 vmin=0, vmax=vmax_both, cmap='inferno')
	axes[0].set_xlabel('x')
	axes[0].set_ylabel('y')
	axes[0].set_title('Original')
	plt.colorbar(im1, ax=axes[0], label='Density')

	# Plot 2: Denoised histogram
	im2 = axes[1].imshow(H_den.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
						 vmin=0, vmax=vmax_both, cmap='inferno')
	axes[1].set_xlabel('x')
	axes[1].set_ylabel('y')
	axes[1].set_title('Denoised')
	plt.colorbar(im2, ax=axes[1], label='Density')

	# Plot 3: Difference histogram (Original - Denoised)
	# Use a diverging colormap for the difference plot
	vmax_diff = max(abs(H_diff.min()), abs(H_diff.max()))
	im3 = axes[2].imshow(H_diff.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
						 vmin=-vmax_diff, vmax=vmax_diff, cmap='RdBu_r')
	axes[2].set_xlabel('x')
	axes[2].set_ylabel('y')
	axes[2].set_title('Difference (Orig - Denoised)')
	plt.colorbar(im3, ax=axes[2], label='Density Difference')

	plt.tight_layout()
	plt.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
	plt.close(fig)

def create_histogram_evolution(denoised_x, save_path, bin_size=0.1, duration_seconds=4.0, original_steps=None):
    """
    Create an animated GIF with a colorbar showing the evolution of the 2D histogram during sampling
    
    Parameters:
    -----------
    denoised_x : torch.Tensor
        Tensor of shape (diffusion_steps, sample_num, dimensions) containing all sampling steps
    save_path : str
        Path to save the GIF animation
    bin_size : float
        Size of histogram bins
    duration_seconds : float
        Total duration of the animation in seconds
    original_steps : int
        Original number of diffusion steps for labeling
    """
    target_total_frames = 50
    fps = int(target_total_frames / duration_seconds)

    # Subsample to fixed frame count
    t_indices = torch.linspace(0, denoised_x.shape[0] - 1, steps=target_total_frames).long()
    frames = denoised_x[t_indices]

    # Map frame indices back to diffusion steps
    if original_steps is None:
        step_map = t_indices.tolist()
    else:
        step_map = torch.linspace(0, original_steps - 1, steps=target_total_frames).long().tolist()

    # Set grid boundaries
    x_min, x_max = -5.5, 5.5
    y_min, y_max = -5.5, 5.5
    
    # Calculate number of bins
    num_bins_x = int((x_max - x_min) / bin_size)
    num_bins_y = int((y_max - y_min) / bin_size)

    # Pre-compute all histograms
    all_histograms = []
    max_density = 0
    
    for frame_idx in range(frames.shape[0]):
        positions = frames[frame_idx].numpy()
        
        H, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=[num_bins_x, num_bins_y],
            range=[[x_min, x_max], [y_min, y_max]],
            density=True
        )
        
        all_histograms.append(H)
        max_density = max(max_density, H.max())

    # Create figure with space for colorbar
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Initialize with first histogram to create colorbar
    H_init = all_histograms[-1]  # Start with final frame
    im = ax.imshow(H_init.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                  vmin=0, vmax=max_density, cmap='inferno', alpha=0.8)
    
    # Create colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Density', rotation=270, labelpad=20, fontsize=12)
    
    def update(rev_t):
        # Reverse time index
        t = frames.shape[0] - 1 - rev_t
        true_t = step_map[t]
        
        # Update image data
        H = all_histograms[t]
        im.set_array(H.T)
        
        # Update title
        ax.set_title(f'Density Evolution - Step {true_t}/{original_steps}', fontsize=14)
        
        return [im]

    # Set initial plot properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')

    # Create animation
    anim = FuncAnimation(fig, update, frames=target_total_frames, blit=False)
    anim.save(save_path + ".gif", writer="pillow", fps=fps)
    plt.close(fig)



if __name__ == "__main__":

	conditioning = True
	zeros = False # test by generating not uniform but all zero distribution
	save_time = False
	
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

	mean_disp_understeps = 10

	def normalize(U, U_min, U_max):
		range = U_max-U_min
		return (U - U_min) / range if range != 0 else 0.0

	conf_to_sample = [1, 3, 6]

	z = torch.randn(diffusion_steps, sample_num, dimensions)
	z[-1] = 0
	if conditioning and not zeros:
		U_max_list = []
		x_orig = []
		if save_time : 
			timing_file = f"./resources/sampling_results/conditioning/run_time_smpl_ds{diffusion_steps}.dat"
			with open(timing_file, "w") as f_time:
				f_time.write("# U_norm\tSamplingTime_seconds\n")

		metric_file = f"./resources/sampling_results/conditioning/metric_smpl_ds{diffusion_steps}.dat"
		with open(metric_file, "w") as f_metr:
			f_metr.write("# U_norm\tAccuracy (% points sampled within U_max)\n")

		
		for i in range(len(conf_steps)):
			file_path = f"./resources/nested_sampling_configs/pos_step{conf_steps[i]}.dat"
			U_max_list.append(extract_U_max_from_file(file_path))
			if i in conf_to_sample :
				x_orig.append(extract_configuration_from_file(file_path))

		U_to_sample = [U_max_list[i] for i in conf_to_sample]
		cond_max, cond_min = max(U_max_list), min(U_max_list)
		u_to_sample = [normalize(U, cond_min, cond_max) for U in U_to_sample]
		
		print(f"\nSampling total model with diffusion steps : {diffusion_steps}, with conditions {[f'{u:.2f}' for u in u_to_sample]}")

		for idx, u in enumerate(u_to_sample) :

			model_path = f"./trained/diffusion_model_tot.pth"
			# U_max = 0 # Conditioning variable (U_max), how should this be defined?
			start_time = time.time()
			denoised_x, displacements = sampling(model_path, z, diffusion_steps, min_beta, max_beta, u, True)
			elapsed_time = time.time() - start_time

			if save_time : 
				with open(timing_file, "a") as f_time:
					f_time.write(f"{u}\t\t\t{elapsed_time:.6f}\n")
			save_path = f"./resources/sampling_results/conditioning/"
			anim_path = save_path + f"smpl_anim_u={u:.2f}_ds{int(diffusion_steps)}"
			img_path = save_path + f"smpl_img_u={u:.2f}_ds{int(diffusion_steps)}"
			
			create_sampling_animation(denoised_x, anim_path + ".gif", original_steps=diffusion_steps)

			U_max = U_max_list[conf_to_sample[idx]]
			dw.plot_configuration(
				img_path, denoised_x[0], U_max=U_max, U_max_cont=True, sampling=True
			)  # Plot the final configuration
			print(f"Sampling animation & final configuration saved for all configuration steps and input energy = {u:.2f}")
	
			# Mean displacement evolution
			# displ_path = save_path + f"displeacement_evolution/mean_disp_histo_u={u}_ds{diffusion_steps}"
			# mean_displ_split(denoised_x, displacements, bin_size=0.15, save_dir=displ_path, n_understeps=mean_disp_understeps)
			
			# Histo comparison
			histo_path = save_path + f"histograms/histo_u={u:.2f}_ds{diffusion_steps}"
			histo_comparison(x_orig[idx], denoised_x[0, :, :], bin_size=0.1, save_path=histo_path)

			# histo animation
			histo_anim_path = save_path + f"histograms/histo_anim_u={u:.2f}_ds{diffusion_steps}"
			create_histogram_evolution(denoised_x, histo_anim_path, bin_size=0.1, original_steps=diffusion_steps)

			# metric to evaluate accuracy of sampling
			denoised_fin_u = dw.energy(denoised_x[0])
			u_mask = denoised_fin_u < U_max
			u_in_bound = u_mask.sum().item()
			accuracy = 100 * (u_in_bound / sample_num)
			with open(metric_file, "a") as f_metr:
				f_metr.write(f"{u}\t\t\t{accuracy:.2f}%\n")

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
		norm_U_to_sample = [0.5, 0.01]
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

	print("")