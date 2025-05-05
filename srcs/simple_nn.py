import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t, c): 
        """
        Args:
			x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
			t (torch.Tensor): Time step tensor of shape (batch_size, 1).
			c (torch.Tensor): Conditioning variable tensor of shape (1).
		"""
        input_data = torch.hstack([x, t, c])
        return self.net(input_data)


if __name__ == "__main__":

	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	print(f"Using {device} device")

	model = SimpleNN().to(device)
	print(model)	