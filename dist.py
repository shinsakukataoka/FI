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


# Experiment settings
settings = [
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 4, 4, 4])},
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([4, 4, 4, 2, 2])}
]

num_experiments = 10

results = {
    "rnn": {
        "mse_rates": [],
        "mse_originals": [],
        "mse_faultys": []
    },
    "lstm": {
        "mse_rates": [],
        "mse_originals": [],
        "mse_faultys": []
    }
}

for i in range(num_experiments):
    print(f"Running experiment {i+1}/{num_experiments}")

    # Create models
    model_rnn = RNNModel(1, 64)
    model_rnn = model_rnn.to(device)

    model_lstm = LSTMModel(1, 64)
    model_lstm = model_lstm.to(device)

    # Training for RNN model
    print("Training RNN model...")
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

    # Training for LSTM model
    print("Training LSTM model...")
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

    # Calculate original MSE for RNN model
    original_output_rnn = model_rnn(input_data.to(device))
    mse_original_rnn = torch.mean((original_output_rnn - correct_data.to(device)) ** 2).item()

    # Calculate original MSE for LSTM model
    original_output_lstm = model_lstm(input_data.to(device))
    mse_original_lstm = torch.mean((original_output_lstm - correct_data.to(device)) ** 2).item()

    # Perform fault injection and calculate error rates
    mse_rates_rnn = []
    mse_rates_lstm = []

    for setting in settings:
        try:
            print(f"Testing setting: {setting}")

            # Fault injection for RNN model
            faulty_model_rnn = dnn_fi(model_rnn, seed=0, **setting)
            faulty_output_rnn = faulty_model_rnn(input_data.to(device))

            mse_faulty_rnn = torch.mean((faulty_output_rnn - correct_data.to(device)) ** 2).item()
            error_rate_rnn = torch.mean((original_output_rnn - faulty_output_rnn) ** 2).item()

            print(f"Original RNN model MSE: {mse_original_rnn:.4f}")
            print(f"Faulty RNN model MSE: {mse_faulty_rnn:.4f}")
            print(f"Fault induced error rate for RNN: {error_rate_rnn:.4f}")

            mse_rates_rnn.append(error_rate_rnn)

            # Fault injection for LSTM model
            faulty_model_lstm = dnn_fi(model_lstm, seed=0, **setting)
            faulty_output_lstm = faulty_model_lstm(input_data.to(device))

            mse_faulty_lstm = torch.mean((faulty_output_lstm - correct_data.to(device)) ** 2).item()
            error_rate_lstm = torch.mean((original_output_lstm - faulty_output_lstm) ** 2).item()

            print(f"Original LSTM model MSE: {mse_original_lstm:.4f}")
            print(f"Faulty LSTM model MSE: {mse_faulty_lstm:.4f}")
            print(f"Fault induced error rate for LSTM: {error_rate_lstm:.4f}")

            mse_rates_lstm.append(error_rate_lstm)

        except Exception as e:
            print(f"An error occurred when injecting faults: {e}")

    results["rnn"]["mse_rates"].append(mse_rates_rnn)
    results["rnn"]["mse_originals"].append(mse_original_rnn)
    results["rnn"]["mse_faultys"].append(mse_faulty_rnn)
    results["lstm"]["mse_rates"].append(mse_rates_lstm)
    results["lstm"]["mse_originals"].append(mse_original_lstm)
    results["lstm"]["mse_faultys"].append(mse_faulty_lstm)

# Calculate average results
average_results = {
    "rnn": {
        "mse_rates": np.mean(results["rnn"]["mse_rates"], axis=0).tolist(),
        "mse_originals": np.mean(results["rnn"]["mse_originals"]),
        "mse_faultys": np.mean(results["rnn"]["mse_faultys"])
    },
    "lstm": {
        "mse_rates": np.mean(results["lstm"]["mse_rates"], axis=0).tolist(),
        "mse_originals": np.mean(results["lstm"]["mse_originals"]),
        "mse_faultys": np.mean(results["lstm"]["mse_faultys"])
    }
}

# Convert the results to JSON
json_results = json.dumps(average_results)

# Save the results to a JSON file
with open("results.json", "w") as file:
    file.write(json_results)

print("Results saved to 'results.json'")
