import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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
    inputs = torch.tensor(data[0], dtype=torch.float32)
    targets = torch.tensor(data[1], dtype=torch.float32)

    print(f'Inputs shape: {inputs.shape}, Targets shape: {targets.shape}')
    print(f'Inputs: {inputs[0]}, Targets: {targets[0]}') 

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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')


import os
import pickle

def load_and_append_pkl_files(dirpath):
    """
    Load all .pkl files in the specified directory and sequentially append their data together.

    Parameters:
    dirpath (str): Path to the directory containing .pkl files.

    Returns:
    combined_data: The combined data from all .pkl files. The type depends on the data in the .pkl files.
    """
    combined_data = None
    
    # Get a list of all .pkl files in the directory
    pkl_files = [f for f in os.listdir(dirpath) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        # Load data from the .pkl file
        with open(os.path.join(dirpath, pkl_file), 'rb') as file:
            data = pickle.load(file)
        
        # Append the data to combined_data
        if combined_data is None:
            combined_data = data
        else:
            if isinstance(combined_data, list):
                combined_data.extend(data)
            elif isinstance(combined_data, (np.ndarray, pd.DataFrame)):
                combined_data = np.append(combined_data, data, axis=0)
            else:
                raise ValueError("Unsupported data type for appending")
    
    return combined_data


def model_env():
    # Define input, hidden, and output dimensions
    input_dim = 5  # 4 for state vector s and 1 for action vector a
    hidden_dim = 64
    output_dim = 4  # Output state vector s_out

    # Create the model
    model, criterion, optimizer = create_model(input_dim, hidden_dim, output_dim)

    # Load data
    # pickle_file = './data/random_agent_data_np.pkl'  # Path to your pickle file
    # pickle_file = './data/learning_agent_nr_training_8.pkl'  # Path to your pickle file
    # dataloader = load_data(pickle_file)


    data = load_and_append_pkl_files('./data/')
    inputs = torch.tensor(data[0], dtype=torch.float32)
    targets = torch.tensor(data[1], dtype=torch.float32)

    print(f'Inputs shape: {inputs.shape}, Targets shape: {targets.shape}')
    print(f'Inputs: {inputs[0]}, Targets: {targets[0]}') 

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    # Train the model
    train_model(model, criterion, optimizer, dataloader, epochs=10)

if __name__ == "__main__":
    model_env()
