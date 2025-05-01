import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
		self.c = c  # Strength of double-well potential
		self.d = d  # Linear term coefficient

	# Define the energy function
	def energy(self, x):
		if len(x.shape) < 2:  # Single configuration
			return self.eps * (self.c * (x[0]**2 - 1)**2 + (x[0] - x[1])**2 + self.d * (x[0] + x[1]))
		elif len(x.shape) == 2:  # Batch of configurations
			return self.eps * (self.c * (x[:, 0]**2 - 1)**2 + (x[:, 0] - x[:, 1])**2 + self.d * (x[:, 0] + x[:, 1]))

	# Initialize random configuration within given bounds
	def init_conf(self, n_points=1, lower_bounds=[-3.5, -6], upper_bounds=[3.5, 6], asNumpy=False): 
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
		
	def plot_configuration (self, x_conf, U_max, conf_step, sampling=False):
		# Define grid for visualization
		x = np.linspace(-2.5, 2.5, 100)
		y = np.linspace(-5, 5, 100)
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
		fig, ax = plt.subplots(dpi=100)

		# Set plot limits
		ax.set_xlim(-6.22, 6.22)
		ax.set_ylim(-6.22, 6.22)
		

		# Scatter plot of sampled particle positions
		ax.scatter(cpu_samples[:, 0], cpu_samples[:, 1], s=0.02, zorder=10, alpha=0.5)

		# Contour plot of energy landscape
		ax.contour(X, Y, Z_target, levels=np.arange(-30, 30, 0.5), cmap="Greys_r", alpha=1, linewidths=1, zorder=0)

		# Highlight specific energy level with a contour
		if not sampling:
			U_max =	U_max.cpu().numpy()
		ax.contour(X, Y, Z_target, levels=[U_max], colors="C1", alpha=1, linewidths=1.5, linestyles='-', zorder=12)

		# Display the plot
		# plt.title(f"Double Well Potential - NestedSamp step {int(conf_step)}")
		path = f"./NestedSampling/nested_sampling_configs/conf_{int(conf_step)}" 
		if sampling:
			path = f"./NestedSampling/resources/sampling_results/conf_{int(conf_step)}_smpl" 
		plt.savefig(path + ".png", dpi=300, bbox_inches='tight')
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
	