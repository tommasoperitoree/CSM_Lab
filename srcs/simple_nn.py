import torch
import torch.nn as nn


class SimpleNN(nn.Module):
	def __init__(self, cond=True, input_dim=2, hidden_dim=64):
		super().__init__()
		input_dim_model = input_dim
		if cond:
			input_dim_model += 2 # for time and conditioning
		else: 
			input_dim_model += 1 # for time only (no conditioning)
		self.cond = cond
		self.net = nn.Sequential(
			nn.Linear(input_dim_model, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, input_dim),
		)

	def forward(self, x, t, c=0): 
		"""
		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
			t (torch.Tensor): Time step tensor of shape (batch_size, 1).
			c (torch.Tensor): Conditioning variable tensor of shape (1).
		"""
		trgt_device = x.device
		tensor_data = [x, t.to(trgt_device)]
		if self.cond:
			tensor_data.append(c)
		input_data = torch.hstack(tensor_data)
		return self.net(input_data)


if __name__ == "__main__":

	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	print(f"Using {device} device")

	model = SimpleNN().to(device)
	print(model)	