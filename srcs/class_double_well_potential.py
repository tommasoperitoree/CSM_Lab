import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.colors as mcolors


# Base class for physical systems
class base_system(nn.Module):
	def __init__(self, n_particles, dimensions, device):
		super(base_system, self).__init__()
		
		self.device = device
		self.n_particles = n_particles  # Number of particles in the system
		self.dimensions = dimensions  # Dimensionality of the system
		self.dofs = self.n_particles * self.dimensions  # Degrees of freedom
		
	def energy(self, x):
		raise NotImplementedError  # Placeholder for energy function

	def init_conf(self):
		raise NotImplementedError  # Placeholder for initial configuration

# Double-well potential system
class double_well(base_system):
	def __init__(self, n_particles, dimensions, device, eps=1., c=1., d=1.):
		super().__init__(n_particles, dimensions, device)
		
		self.eps = eps  # Energy scaling factor
		self.c = c  	# Strength of double-well potential
		self.d = d  	# Linear term coefficient
		self.norm = 1.	# normalization

		# Normalization - deprecated: only needed for diffusion model
		# conf = self.init_conf(n_points=int(1e4))
		# conf_energy = self.energy(conf, normalize=False)
		# e_max = conf_energy.max()
		# self.eps = 1/e_max  # Normalize energy to 1

	# Define the energy function
	def energy(self, x, normalize=True):
		if len(x.shape) < 2:  # Single configuration
			energy = self.eps*(self.c * (x[0]**2 - 1)**2 + (x[0] - x[1])**2 + self.d * (x[0] + x[1]))
		elif len(x.shape) == 2:  # Batch of configurations
			energy = self.eps*(self.c * (x[:, 0]**2 - 1)**2 + (x[:, 0] - x[:, 1])**2 + self.d * (x[:, 0] + x[:, 1]))
		if normalize:
			energy /= self.norm
		return energy

	# Initialize random configuration within given bounds
	def init_conf(self, n_points=1, lower_bounds=[-2.5, -5], upper_bounds=[2.5, 5], asNumpy=False): 
		# careful, bounds should match number of dimensions
		if len(lower_bounds) != self.dimensions or len(upper_bounds) != self.dimensions:
			raise ValueError("Lower bounds must match the number of dimensions.")
		
		lower_bounds = np.array(lower_bounds)
		upper_bounds = np.array(upper_bounds)

		rndm = np.random.rand(n_points, self.dimensions)
		conf = lower_bounds + (upper_bounds - lower_bounds) * rndm

		if asNumpy:
			return conf
		else:
			return torch.from_numpy(conf.astype(np.float32)).to(self.device)
		
	def plot_configuration (self, output_dir, x_conf, U_max = 0, U_max_cont=True, sampling=False, normalized_en=False):
		# Define grid for visualization
		upper_lim, lower_lim = 5.5, -5.5
		x = np.linspace(lower_lim, upper_lim, 200)
		y = np.linspace(lower_lim, upper_lim, 200)
		# Create a meshgrid for contour plotting
		X, Y = np.meshgrid(x, y)

		# Compute energy landscape
		Z_target = np.zeros([len(X), len(Y)])
		for i in range(len(X)):
			for j in range(len(Y)):
				# conf = np.array([X[i][j], Y[i][j]])  # Current point in the grid
				conf = torch.tensor([X[i][j], Y[i][j]], dtype=torch.float32, device=self.device)  # Convert to PyTorch tensor
				Z_target[i, j] = self.energy(conf).squeeze().cpu().numpy()  # Ensure compatibility
				
		# Load particle samples and max energy level for contour

		cpu_samples = x_conf.cpu().numpy()  # Replace with actual sample tensor

		# Set figure size (in inches)
		#fig_size = (24 * 0.393701, 24 * 0.393701)  # Convert from cm to inches
		#fig, ax = plt.subplots(figsize=fig_size, dpi=100)
		# x_prop = x_upper_lim - x_lower_lim
		# y_prop = y_upper_lim - y_lower_lim
		#fig, ax = plt.subplots(figsize=(0.8*x_prop,0.8*y_prop), dpi=100)
		fig, ax = plt.subplots(figsize=(7,7), dpi=100)

		# Set plot limits
		ax.set_xlim(lower_lim, upper_lim)
		ax.set_ylim(lower_lim, upper_lim)

		# Scatter plot of sampled particle positions
		ax.scatter(cpu_samples[:, 0], cpu_samples[:, 1], s=0.05, zorder=10, alpha=0.8)

		num_contour_levels = 100
		upper_bound_contour = 45
		lower_bound_contour = -5
		if normalized_en :
			upper_bound_contour = 1.3
			lower_bound_contour = -0.3
		contour_levels = np.linspace(lower_bound_contour, upper_bound_contour, num_contour_levels)

		norm_greyscale = mcolors.Normalize(vmin=np.min(contour_levels), vmax=np.max(contour_levels))
	
		ax.contour(X, Y, Z_target, levels=contour_levels, cmap="Greys_r", norm=norm_greyscale, linewidths=0.9, alpha=0.2, zorder=0)

		# Highlight specific energy level with a contour
		if not sampling and isinstance(U_max, torch.Tensor):
			U_max_val =	U_max.cpu().numpy().item()
		else:
			U_max_val = U_max
		
		if U_max_cont:
			ax.contour(X, Y, Z_target, levels=[U_max_val], colors="C1", alpha=1, linewidths=1.5, linestyles='-', zorder=12)

		# Display the plot
		plt.savefig(output_dir + ".png", dpi=300, bbox_inches='tight')
		plt.close(fig)	

if __name__ == "__main__":

	# Define system parameters
	n_particles = 1  # Number of particles
	dimensions = 2  # 2D system

	device = "cpu"  # Force CPU for faster performance

	# Set device (CUDA, MPS, or CPU)
	# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	# print(f"Using {device} device")


	# Instantiate the double-well system
	double_well_2D = double_well(n_particles=n_particles, dimensions=dimensions, device=device, eps=3., c=1., d=0.5)

	double_well_2D.plot_with_contour()
	