import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from forward_process import calculate_parameters
from prepare_dataset import extract_U_max_from_file
from simple_nn import SimpleNN
from double_well import double_well


def sampling(model_path, sample_num, diffusion_steps, min_beta, max_beta):
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        x_init = torch.randn(size=(sample_num, 2))
        beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
            diffusion_steps, min_beta, max_beta
        )
        denoised_x = torch.zeros((diffusion_steps, x_init.shape[0], x_init.shape[1]))
        denoised_x[-1] = x_init
        for t in range(diffusion_steps - 1, 0, -1):
            if t > 1:
                z = torch.randn(x_init.shape)
            else:
                z = 0
            ts = torch.full((x_init.shape[0], 1), t)
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
            denoised_x[t - 1] = mu + torch.sqrt(beta_ts[t]) * z

    return denoised_x


def create_sampling_animation(denoised_x, diffusion_steps, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [], alpha=0.1, s=1)

    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
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
    
	conf_steps = [5e3, 1e4, 1.6e4, 2.3e4, 3.1e4, 4e4, 5e4, 6.1e4, 7.3e4]  # Number of configuration steps
	conf_steps = [int(step) for step in conf_steps]
    
	# Define system parameters
	n_particles = 1  # Number of particles
	dimensions = 2  # 2D system
	n_live_points = int(1e4)  # Number of live points in nested sampling
	n_correl_steps = 15  # Number of correlation steps

	# Set device (CUDA, MPS, or CPU)
	# MPS is for Apple Silicon Macs with Metal Performance Shaders support
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	# print(f"Using {device} device")


	# Instantiate the double-well system
	dw = double_well(n_particles=n_particles, dimensions=dimensions, device=device, eps=3., c=1., d=0.5)

	for i in range(len(conf_steps)):
		model_path = f"./NestedSampling/trained/diffusion_model_{conf_steps[i]}.pth"
		sample_num = 100000
		diffusion_steps = 50
		min_beta = 1e-4
		max_beta = 0.02
		denoised_x = sampling(model_path, sample_num, diffusion_steps, min_beta, max_beta)
		save_path = f"./NestedSampling/resources/sampling_results/smpl_{conf_steps[i]}.gif"
		create_sampling_animation(denoised_x, diffusion_steps, save_path)
	
		U_max = extract_U_max_from_file(f"./NestedSampling/nested_sampling_configs/conf_step_{conf_steps[i]}.dat")
	
		dw.plot_configuration(
			denoised_x[0], U_max, conf_step=conf_steps[i], sampling=True
		)  # Plot the final configuration
          
		print(f"Sampling animation & final configuration saved for configuration step {conf_steps[i]}")
          
