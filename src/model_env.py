import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class StateActionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StateActionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define input, hidden, and output dimensions
input_dim = 5  # 4 for state vector s and 1 for action vector a
hidden_dim = 64
output_dim = 4  # Output state vector s_out

# Instantiate the model, define the loss function and the optimizer
model = StateActionNet(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary
print(model)
