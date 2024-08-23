import torch
from torch import nn, optim
import seaborn as sns
import numpy as np
import pandas as pd
import json
from fault_injection import dnn_fi
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your data processing
# Load data
flights = sns.load_dataset("flights")
data = flights["passengers"].values

# Normalize data
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Your data processing
origin_data_num = 12
prd_data_num = len(data) - origin_data_num
input_data = np.zeros((prd_data_num, origin_data_num, 1))
correct_data = np.zeros((prd_data_num, 1))

for i in range(prd_data_num):
    input_data[i] = data[i : i + origin_data_num].reshape(-1, 1)
    correct_data[i] = data[i + origin_data_num : i + origin_data_num + 1]

input_data = torch.tensor(input_data, dtype=torch.float)
correct_data = torch.tensor(correct_data, dtype=torch.float)

dataset = torch.utils.data.TensorDataset(input_data, correct_data)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Your RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x, h=None):
        output, hn = self.rnn(x, h)
        y = self.output(output[:, -1, :])
        return y


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x, h=None):
        output, hn = self.rnn(x, h)
        y = self.output(output[:, -1, :])
        return y


model_rnn = RNNModel(1, 64)
model_rnn = model_rnn.to(device)

print("Original RNN Model:")
print(model_rnn)

loss_fnc = nn.MSELoss()
optimizer = optim.SGD(model_rnn.parameters(), lr=0.01)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for x, t in train_loader:
        x = x.to(device)
        t = t.to(device)
        y = model_rnn(x)
        loss = loss_fnc(y, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss for RNN: {running_loss}")

original_output_rnn = model_rnn(input_data.to(device))
mse_original_rnn = torch.mean((original_output_rnn - correct_data.to(device)) ** 2).item()

model_lstm = LSTMModel(1, 64)
model_lstm = model_lstm.to(device)

print("Original LSTM Model:")
print(model_lstm)

optimizer = optim.SGD(model_lstm.parameters(), lr=0.01)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for x, t in train_loader:
        x = x.to(device)
        t = t.to(device)
        y = model_lstm(x)
        loss = loss_fnc(y, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss for LSTM: {running_loss}")

original_output_lstm = model_lstm(input_data.to(device))
mse_original_lstm = torch.mean((original_output_lstm - correct_data.to(device)) ** 2).item()

# Get the list of layer names in the models
layer_names = [name for name, _ in model_rnn.named_parameters()]

# Define a list to store the results
results = []

settings = [
    {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2]), 'ber': 0.01, 'layer_names': None},
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2]), 'ber': 0.01, 'layer_names': None}
]

for idx, layer in enumerate(layer_names):
    # These lists will hold the error rates for each layer
    mse_rates_rnn = []
    mse_originals_rnn = []
    mse_faultys_rnn = []
    mse_rates_lstm = []
    mse_originals_lstm = []
    mse_faultys_lstm = []

    for setting_idx, setting in enumerate(settings):
        setting['ber'] = 0.005
        setting['layer_names'] = [layer]

        try:
            faulty_model_rnn = dnn_fi(model_rnn, seed=0, **setting)
            faulty_output_rnn = faulty_model_rnn(input_data.to(device))

            mse_original_rnn = torch.mean((original_output_rnn - correct_data.to(device)) ** 2).item()
            mse_faulty_rnn = torch.mean((faulty_output_rnn - correct_data.to(device)) ** 2).item()
            error_rate_rnn = torch.mean((original_output_rnn - faulty_output_rnn) ** 2).item()

            print(f"Original RNN model MSE: {mse_original_rnn:.4f}")
            print(f"Faulty RNN model MSE: {mse_faulty_rnn:.4f}")
            print(f"Fault induced error rate for RNN: {error_rate_rnn:.4f}")

        except Exception as e:
            print(f"An error occurred when injecting faults into the RNN model: {e}")
            mse_original_rnn = torch.mean((original_output_rnn - correct_data.to(device)) ** 2).item()
            mse_faulty_rnn = np.nan
            error_rate_rnn = np.nan

        mse_rates_rnn.append(error_rate_rnn)
        mse_originals_rnn.append(mse_original_rnn)
        mse_faultys_rnn.append(mse_faulty_rnn)

        try:
            faulty_model_lstm = dnn_fi(model_lstm, seed=0, **setting)
            faulty_output_lstm = faulty_model_lstm(input_data.to(device))

            mse_original_lstm = torch.mean((original_output_lstm - correct_data.to(device)) ** 2).item()
            mse_faulty_lstm = torch.mean((faulty_output_lstm - correct_data.to(device)) ** 2).item()
            error_rate_lstm = torch.mean((original_output_lstm - faulty_output_lstm) ** 2).item()

            print(f"Original LSTM model MSE: {mse_original_lstm:.4f}")
            print(f"Faulty LSTM model MSE: {mse_faulty_lstm:.4f}")
            print(f"Fault induced error rate for LSTM: {error_rate_lstm:.4f}")

        except Exception as e:
            print(f"An error occurred when injecting faults into the LSTM model: {e}")
            mse_original_lstm = torch.mean((original_output_lstm - correct_data.to(device)) ** 2).item()
            mse_faulty_lstm = np.nan
            error_rate_lstm = np.nan

        mse_rates_lstm.append(error_rate_lstm)
        mse_originals_lstm.append(mse_original_lstm)
        mse_faultys_lstm.append(mse_faulty_lstm)

    # Store the results for the current layer
    layer_results = {
        'layer': layer,
        'rnn_results': {
            f'Setting {setting_idx + 1}': {
                'error_rate': mse_rates_rnn[setting_idx],
                'original_mse': mse_originals_rnn[setting_idx],
                'faulty_mse': mse_faultys_rnn[setting_idx]
            }
            for setting_idx in range(len(settings))
        },
        'lstm_results': {
            f'Setting {setting_idx + 1}': {
                'error_rate': mse_rates_lstm[setting_idx],
                'original_mse': mse_originals_lstm[setting_idx],
                'faulty_mse': mse_faultys_lstm[setting_idx]
            }
            for setting_idx in range(len(settings))
        },
    }

    results.append(layer_results)

# Save results to a JSON file
with open('layers.json', 'w') as file:
    json.dump(results, file, indent=4)
