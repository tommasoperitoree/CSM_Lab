import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from forward_process import calculate_parameters
from class_dataset import extract_U_max_from_file
from simple_nn import SimpleNN
from class_double_well_potential import double_well


def sampling(model_path, z, sample_num, diffusion_steps, min_beta, max_beta, U_max=0, cond=True):
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

	return denoised_x


def create_sampling_animation(denoised_x, diffusion_steps, save_path):
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
		# Update scatter plot
		t = diffusion_steps - 1 - rev_t
		scatter.set_offsets(denoised_x[t])
		ax.set_title(f"Sampling - Step {t}/{diffusion_steps}")
		return (scatter,)

	# Create animation
	anim = FuncAnimation(fig, update, frames=diffusion_steps, init_func=init, blit=True)
	# Save animation as video
	anim.save(save_path, writer="pillow", fps=10)
	plt.close(fig)


if __name__ == "__main__":

	conditioning = True
	
	#conf_steps = [5e3]
	conf_steps = [5e3, 1e4, 1.6e4, 2.3e4, 3.1e4, 4e4, 5e4, 6.1e4, 7.3e4]  # Number of configuration steps
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

	sample_num = int(1e5)
	diffusion_steps = 50
	min_beta = 1e-4
	max_beta = 0.02

	U_max_list = [3, 0, -3]

	z = torch.randn(diffusion_steps, sample_num, dimensions)
	z[-1] = 0
	if conditioning :
		for u in U_max_list:
			model_path = f"./trained/diffusion_model_tot.pth"
			# U_max = 0 # Conditioning variable (U_max), how should this be defined?
			denoised_x = sampling(model_path, z, sample_num, diffusion_steps, min_beta, max_beta, u)
			save_path = f"./resources/sampling_results/smpl_U={u}"
			create_sampling_animation(denoised_x, diffusion_steps, save_path + ".gif")
			dw.plot_configuration(
				save_path, denoised_x[0], U_max=u, U_max_cont=True, sampling=True
			)  # Plot the final configuration
			print(f"Sampling animation & final configuration saved for all configuration steps and input energy = {u}")
	else :
		for i in range(len(conf_steps)):
			model_path = f"./trained/diffusion_model_step{conf_steps[i]}.pth"
			denoised_x = sampling(model_path, z, sample_num, diffusion_steps, min_beta, max_beta, cond=False)
			save_path = f"./resources/sampling_results/"
			anim_path = save_path + f"smpl_anim_step{conf_steps[i]}"
			img_path = save_path + f"smpl_img_step{conf_steps[i]}"
			create_sampling_animation(denoised_x, diffusion_steps, anim_path + ".gif")

			U_max = extract_U_max_from_file(f"./resources/nested_sampling_configs/pos_step{conf_steps[i]}.dat")

			dw.plot_configuration(
				img_path, denoised_x[0], U_max=U_max, U_max_cont=True, sampling=True
			)  # Plot the final configuration

			print(f"Sampling animation & final configuration saved for configuration step {conf_steps[i]}")