import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from fault_injection import dnn_fi_temp
from keras.datasets import imdb
import numpy as np
import json
from val_config import *

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Define the function for vectorizing the sequences
def vectorize_sequences(sequences, dimension=10000):
    results = torch.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
x_train = vectorize_sequences(train_data).to(device)
x_test = vectorize_sequences(test_data).to(device)
y_train = torch.tensor(train_labels).float().view(-1, 1).to(device)
y_test = torch.tensor(test_labels).float().view(-1, 1).to(device)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10000, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)).view(-1, 1)
        return x

settings = [
    {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'afloat', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'afloat', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}
]

# ber_values = [0.00000001,0.01]
ber_values = np.logspace(np.log10(10 * 10**-10), np.log10(10 * 10**-2), num=50)

# Initialize the model and train it
model = Net().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters())

# Training the model
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=512)

for epoch in range(20):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def loop_through_ber_values():
    results = {}
    for ber in ber_values:
        results[ber] = {}
        for setting in settings:
            # Create a copy of the trained model
            model_with_faults = Net().to(device)
            model_with_faults.load_state_dict(model.state_dict())

            # Inject faults
            model_with_faults = dnn_fi_temp(model_with_faults, seed=0, ber=ber, **setting)

            # Test the model before fault injection
            model.eval()
            with torch.no_grad():
                outputs_before = model(x_test)
                predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                classification_error_before = 1 - accuracy_before

            # Test the model after fault injection
            model_with_faults.eval()
            with torch.no_grad():
                outputs_after = model_with_faults(x_test)
                predictions_after = (outputs_after > 0.5).type(torch.IntTensor).view(-1)
                accuracy_after = accuracy_score(y_test.cpu().numpy().flatten(), predictions_after.cpu().numpy())
                classification_error_after = 1 - accuracy_after

            classification_error_induced = classification_error_after - classification_error_before

            setting_summary = f"{setting['q_type']}-{setting['encode']}-{setting['int_bits']}-{setting['frac_bits']}"
            results[ber][setting_summary] = {
                "accuracy_before": accuracy_before,
                "classification_error_before": classification_error_before,
                "accuracy_after": accuracy_after,
                "classification_error_after": classification_error_after,
                "classification_error_induced": classification_error_induced
            }
    
    output_file = "MLP_loop.json"

    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

def layer_analysis(num_repeats=10):
    ber = 0.005
    setting_indices = [0, 1, 2, 3]  # Indices of all six settings

    layer_names = [name for name, _ in model.named_parameters()]  # Get list of layer names

    results = {}
    for layer_name in layer_names:
        results[layer_name] = {}

        for setting_index in setting_indices:
            setting = settings[setting_index]

            classification_errors = []
            for _ in range(num_repeats):
                model_with_faults = Net().to(device)  # Create a new model for each repeat

                # Load the pre-trained weights into the new model
                model_with_faults.load_state_dict(model.state_dict())

                # Inject faults only in the specified layer
                model_with_faults = dnn_fi(model_with_faults, seed=0, ber=ber, layer_names=[layer_name], **setting)

                # Test the model after fault injection
                model_with_faults.eval()
                with torch.no_grad():
                    outputs_before = model(x_test)
                    predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                    accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                    classification_error_before = 1 - accuracy_before

                # Test the model after fault injection
                model_with_faults.eval()
                with torch.no_grad():
                    outputs_after = model_with_faults(x_test)
                    predictions_after = (outputs_after > 0.5).type(torch.IntTensor).view(-1)
                    accuracy_after = accuracy_score(y_test.cpu().numpy().flatten(), predictions_after.cpu().numpy())
                    classification_error_after = 1 - accuracy_after

                classification_error_induced = classification_error_after - classification_error_before

                classification_errors.append(classification_error_induced)

            if len(classification_errors) > 2:
                classification_errors.remove(min(classification_errors))  # Remove the minimum value
                classification_errors.remove(max(classification_errors))  # Remove the maximum value
            else:
                print("Error: Not enough elements to remove min and max values")

            classification_error_average = np.mean(classification_errors)

            setting_summary = f"{setting['q_type']}-{setting['encode']}-{setting['int_bits']}-{setting['frac_bits']}"
            results[layer_name][setting_summary] = {
                "classification_error_average": classification_error_average
            }

    output_file = 'MLP_layers.json'
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

layer_analysis()