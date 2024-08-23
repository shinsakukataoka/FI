import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
import numpy as np
import json
from fault_injection import dnn_fi
from val_config import *

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Define the function for padding and truncating the sequences
def pad_sequences(sequences, max_length):
    padded_sequences = torch.zeros((len(sequences), max_length)).long()
    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_length]
        padded_sequences[i, :len(sequence)] = torch.tensor(sequence)
    return padded_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the maximum sequence length
max_length = 200

# Data preparation
x_train = pad_sequences(train_data, max_length=max_length).to(device)
x_test = pad_sequences(test_data, max_length=max_length).to(device)

y_train = torch.tensor(train_labels).float().view(-1, 1).to(device)
y_test = torch.tensor(test_labels).float().view(-1, 1).to(device)

# Define the LSTM model
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x.permute(1, 0, 2))
        x = h_n[-1]
        x = self.fc(x)
        x = torch.sigmoid(x).view(-1, 1)
        return x

settings = [
    {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    #{'q_type': 'unsigned', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'afloat', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    #{'q_type': 'unsigned', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'afloat', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}
]

#ber_values = [0.00000001,0.01]
# ber_values = np.logspace(np.log10(10 * 10**-10), np.log10(10 * 10**-2), num=500)

def loop_through_ber_values():
    ber_values = list_10_2
    results = {}
    for ber in ber_values:
        results[ber] = {}
        for setting in settings:
            model = LSTMNet(input_dim=10000, hidden_dim=16, output_dim=1).to(device)

            # Specify the loss function and the optimizer
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

            # Test the model before fault injection
            model.eval()
            with torch.no_grad():
                outputs_before = model(x_test)
                predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                classification_error_before = 1 - accuracy_before

            # Inject faults
            model_with_faults = dnn_fi(model, seed=0, ber=ber, **setting)

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
    if ber_values == list_1_1:
        output_file = 'LSTM_results_list1_1.json'
    elif ber_values == list_1_2:
        output_file = 'LSTM_results_list1_2.json'
    elif ber_values == list_2_1:
        output_file = 'LSTM_results_list2_1.json'
    elif ber_values == list_2_2:
        output_file = 'LSTM_results_list2_2.json'
    elif ber_values == list_3_1:
        output_file = 'LSTM_results_list3_1.json'
    elif ber_values == list_3_2:
        output_file = 'LSTM_results_list3_2.json'
    elif ber_values == list_4_1:
        output_file = 'LSTM_results_list4_1.json'
    elif ber_values == list_4_2:
        output_file = 'LSTM_results_list4_2.json'
    elif ber_values == list_5_1:
        output_file = 'LSTM_results_list5_1.json'
    elif ber_values == list_5_2:
        output_file = 'LSTM_results_list5_2.json'
    elif ber_values == list_6_1:
        output_file = 'LSTM_results_list6_1.json'
    elif ber_values == list_6_2:
        output_file = 'LSTM_results_list6_2.json'
    elif ber_values == list_7_1:
        output_file = 'LSTM_results_list7_1.json'
    elif ber_values == list_7_2:
        output_file = 'LSTM_results_list7_2.json'
    elif ber_values == list_8_1:
        output_file = 'LSTM_results_list8_1.json'
    elif ber_values == list_8_2:
        output_file = 'LSTM_results_list8_2.json'
    elif ber_values == list_9_1:
        output_file = 'LSTM_results_list9_1.json'
    elif ber_values == list_9_2:
        output_file = 'LSTM_results_list9_2.json'
    elif ber_values == list_10_1:
        output_file = 'LSTM_results_list10_1.json'
    elif ber_values == list_10_2:
        output_file = 'LSTM_results_list10_2.json'
    else:
        output_file = 'LSTM_results_default.json'

    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

def layer_analysis(num_repeats=5):
    ber = 0.005
    setting_indices = [0, 1, 2, 3]  # Indices of all four settings

    model = LSTMNet(input_dim=10000, hidden_dim=16, output_dim=1).to(device)  # Move the model definition here
    layer_names = [name for name, _ in model.named_parameters()]  # Get list of layer names

    results = {}
    for layer_name in layer_names:
        results[layer_name] = {}

        for setting_index in setting_indices:
            setting = settings[setting_index]

            classification_errors = []
            for _ in range(num_repeats):
                model = LSTMNet(input_dim=10000, hidden_dim=16, output_dim=1).to(device)  # Reset the model for each repeat

                # Specify the loss function and the optimizer
                criterion = torch.nn.BCELoss()
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

                # Test the model before fault injection
                model.eval()
                with torch.no_grad():
                    outputs_before = model(x_test)
                    predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                    accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                    classification_error_before = 1 - accuracy_before

                # Inject faults only in the specified layer
                model_with_faults = dnn_fi(model, seed=0, ber=ber, layer_names=[layer_name], **setting)

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

    output_file = 'LSTM_layer_analysis_results.json'
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

#settings = [
#    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
#    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 4, 'frac_bits': 4, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
#    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 6, 'frac_bits': 2, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
#    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 12, 'frac_bits': 4, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2,2,2,2,2,2,2,2,2])},
#    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 4, 'frac_bits': 12, 'rep_conf': np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,])},
#    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 8, 'frac_bits': 8, 'rep_conf': np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])},
#    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 9, 'frac_bits': 23, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2])}
#]

import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

def memory_analysis(num_repeats=5):
    ber = 0.001
    setting_indices = [0, 1, 2, 3, 4, 5, 6,7,8]
    model = LSTMNet(input_dim=10000, hidden_dim=16, output_dim=1).to(device)  # Move the model definition here
    layer_names = ['dummy']  # Get list of layer names

    results = {}
    for layer_name in layer_names:
        results[layer_name] = {}
        for setting_index in setting_indices:
            setting = settings[setting_index]
            classification_errors = []
            for _ in range(num_repeats):
                model = LSTMNet(input_dim=10000, hidden_dim=16, output_dim=1).to(device)  # Reset the model for each repeat
                # Specify the loss function and the optimizer
                criterion = torch.nn.BCELoss()
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

                # Test the model before fault injection
                model.eval()
                with torch.no_grad():
                    outputs_before = model(x_test)
                    predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                    accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                    classification_error_before = 1 - accuracy_before
                model_with_faults = dnn_fi(model, seed=0, ber=ber, **setting)
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

    output_file = 'LSTM_memory_analysis_results.json'
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

loop_through_ber_values()