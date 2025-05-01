import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

from forward_process import calculate_data_at_certain_time, calculate_parameters
from prepare_dataset import extract_configuration_from_file, ConfigurationsDataset
from simple_nn import SimpleNN


def train(
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
):

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

	model = SimpleNN().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
	#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, threshold=1e-3, threshold_mode='rel', cooldown=0, min_lr=1e-9, eps=1e-10)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2)
	loss_fn = nn.MSELoss()
	beta_ts, alpha_ts, bar_alpha_ts = calculate_parameters(
		diffusion_steps, min_beta, max_beta
	)

	e_loss = [] # list to store loss values
	for epoch in range(max_epochs):

		# training loop
			
		model.train()  # Set the model to training mode
		train_loss = 0

		for x in train_loader:
			# print("x shape: ", x.shape)
			random_time_step = torch.randint(0, diffusion_steps, size=[len(x), 1])
			noised_x_t, eps = calculate_data_at_certain_time(
				x, bar_alpha_ts, random_time_step
			)
			predicted_eps = model.forward(
				noised_x_t.to(device), random_time_step.to(device)
			)
			loss = loss_fn(predicted_eps, eps.to(device))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.item()

		train_loss /=  len(train_loader) # Calculate average train loss

		# testing loop

		model.eval()  # Set the model to evaluation mode 
		test_loss = 0
		
		with torch.no_grad():  # Disable gradient calculations for efficiency
			for x in test_loader :  # Iterate over test data
				random_time_step = torch.randint(0, diffusion_steps, size=[len(x), 1])
				noised_x_t, eps = calculate_data_at_certain_time(
					x, bar_alpha_ts, random_time_step
				)
				predicted_eps = model.forward(
					noised_x_t.to(device), random_time_step.to(device)
				)
				test_loss += loss_fn(predicted_eps, eps.to(device)).item()

		test_loss /= len(test_loader)  # Calculate average test loss
		# scheduler.step(test_loss)  # Update learning rate based on train loss

		#print('\nEpoch: {}, Test Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

		print(f"Epoch {epoch}, l.r.={scheduler.get_last_lr()[0]:.5g} | Test_loss={test_loss:.5g}, Train_loss={train_loss:.5g}")
		e_loss.append([train_loss, test_loss])

		if scheduler._last_lr[0] < 1e-6:
			print("\nReached learning rate threshold, stopping training @ epoch ", epoch)
			break
	
	print("Finished training!!")
	torch.save(model.state_dict(), output_model_path)
	print("Saved model: ", output_model_path)

	return e_loss


def plot_loss(loss, save_path, conf_step):
	"""
	Plot and save the train and test loss over epochs.

	Args:
		loss (list): A list of [train_loss, test_loss] for each epoch.
		save_path (str): Path to save the plot.
	"""
	epochs = range(1, len(loss) + 1)
	train_loss = [l[0] for l in loss]
	test_loss = [l[1] for l in loss]

	plt.figure(figsize=(10, 6))
	plt.plot(epochs, train_loss, label="Train Loss", linestyle='-')
	plt.plot(epochs, test_loss, label="Test Loss", linestyle='--')
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Train and Test Loss Over Epochs for config step " + str(conf_step))
	#plt.yscale("log")  # Set y-axis to logarithmic scale
	plt.legend()
	#plt.ylim(0, 1)
	plt.grid(True)
	plt.savefig(save_path)
	plt.close()
	print(f"Loss plot saved to {save_path}")



if __name__ == "__main__":

	# set device (CUDA, MPS, or CPU)
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	print(f"Using {device} device")



	# Define system parameters
	sample_num = 100000
	noise_std = 0.5

	conf_steps = [5e3, 1e4, 1.6e4, 2.3e4, 3.1e4, 4e4, 5e4, 6.1e4, 7.3e4]  # Number of configuration steps
	conf_steps = [5e3]  # Number of configuration steps
	conf_steps = [int(step) for step in conf_steps]
	

	for i in range(len(conf_steps)):

		filepath = f"./NestedSampling/configs/conf_step_{conf_steps[i]}.dat"  # Path to the configuration file
		#x = extract_configuration_from_file(filepath)
		#data = x.clone().detach()

		test_fraction = 0.1 
		train_data  = ConfigurationsDataset(filepath, test_fraction, train=True)
		test_data = ConfigurationsDataset(filepath, test_fraction, train=False)

		batch_size = 128
		max_epochs = 100
		diffusion_steps = 50
		min_beta = 1e-4
		max_beta = 0.02
		init_learning_rate = 1e-3
		output_model_path = f"./NestedSampling/trained/diffusion_model_{conf_steps[i]}.pth"
		loss_plot_path = f"./NestedSampling/resources/loss_plt_{conf_steps[i]}.png"

		print(f"\nTraining model for configuration step {conf_steps[i]}...\n")
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
		plot_loss(loss, loss_plot_path, conf_steps[i])
		print("\n")