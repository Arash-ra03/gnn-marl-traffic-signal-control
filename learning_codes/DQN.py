import torch
import torch.nn as nn

#This file is made to make code cleaner for performing hyperparameter search
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # → 512 → 256 → 128 instead of 128 → 64
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.head = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.head(x)