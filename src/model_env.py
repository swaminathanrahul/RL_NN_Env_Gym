import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset

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

def create_model(input_dim=5, hidden_dim=64, output_dim=4):
    model = StateActionNet(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

def load_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    inputs = torch.tensor(data['inputs'], dtype=torch.float32)
    targets = torch.tensor(data['targets'], dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_model(model, criterion, optimizer, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

def model_env():
    # Define input, hidden, and output dimensions
    input_dim = 5  # 4 for state vector s and 1 for action vector a
    hidden_dim = 64
    output_dim = 4  # Output state vector s_out

    # Create the model
    model, criterion, optimizer = create_model(input_dim, hidden_dim, output_dim)

    # Load data
    pickle_file = 'data.pkl'  # Path to your pickle file
    dataloader = load_data(pickle_file)

    # Train the model
    train_model(model, criterion, optimizer, dataloader, epochs=10)

if __name__ == "__main__":
    model_env()
