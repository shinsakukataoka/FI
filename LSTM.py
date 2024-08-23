import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
import numpy as np
import json
from fault_injection import dnn_fi_temp
from val_config import *

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def pad_sequences(sequences, max_length):
    padded_sequences = torch.zeros((len(sequences), max_length)).long()
    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_length]
        padded_sequences[i, :len(sequence)] = torch.tensor(sequence)
    return padded_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length = 200

x_train = pad_sequences(train_data, max_length=max_length).to(device)
x_test = pad_sequences(test_data, max_length=max_length).to(device)

y_train = torch.tensor(train_labels).float().view(-1, 1).to(device)
y_test = torch.tensor(test_labels).float().view(-1, 1).to(device)

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(x.permute(1, 0, 2))
        x = h_n[-1]
        x = self.fc(x)
        x = torch.sigmoid(x).view(-1, 1)
        return x

ber_values = np.logspace(np.log10(10 * 10**-10), np.log10(10 * 10**-2), num=1000)

model = LSTMNet(input_dim=10000, hidden_dim=32, output_dim=1, n_layers=2, dropout=0.5).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters())
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
    settings = [
        {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        #{'q_type': 'unsigned', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        #{'q_type': 'unsigned', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}
    ]
    results = {}
    for ber in ber_values:
        results[ber] = {}
        for setting in settings:
            model_copy = LSTMNet(input_dim=10000, hidden_dim=32, output_dim=1, n_layers=2, dropout=0.5).to(device)
            model_copy.load_state_dict(model.state_dict())

            # Test the model before fault injection
            model_copy.eval()
            with torch.no_grad():
                outputs_before = model_copy(x_test)
                predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                classification_error_before = 1 - accuracy_before

            # Inject faults
            model_with_faults = dnn_fi_temp(model_copy, seed=0, ber=ber, **setting)

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
    output_file = "LSTM_loops.json"
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

def loop_through_layers():
    settings = [
        {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}
    ]
    results = {}
    layer_names = [name for name, _ in model.named_parameters()]  # Get list of layer names
    ber = 0.01  # Set the bit error rate to 0.005 for all layers
    for layer in layer_names:
        results[layer] = {}

        for setting in settings:
            model_copy = LSTMNet(input_dim=10000, hidden_dim=32, output_dim=1, n_layers=2, dropout=0.5).to(device)
            model_copy.load_state_dict(model.state_dict())

            # Test the model before fault injection
            model_copy.eval()
            with torch.no_grad():
                outputs_before = model_copy(x_test)
                predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                classification_error_before = 1 - accuracy_before

            # Inject faults in specific layer
            model_with_faults = dnn_fi_temp(model_copy, seed=0, ber=ber, layer_names=[layer], **setting)

            # Test the model after fault injection
            model_with_faults.eval()
            with torch.no_grad():
                outputs_after = model_with_faults(x_test)
                predictions_after = (outputs_after > 0.5).type(torch.IntTensor).view(-1)
                accuracy_after = accuracy_score(y_test.cpu().numpy().flatten(), predictions_after.cpu().numpy())
                classification_error_after = 1 - accuracy_after

            classification_error_induced = classification_error_after - classification_error_before

            setting_summary = f"{setting['q_type']}-{setting['encode']}-{setting['int_bits']}-{setting['frac_bits']}"
            results[layer][setting_summary] = {
                "accuracy_before": accuracy_before,
                "classification_error_before": classification_error_before,
                "accuracy_after": accuracy_after,
                "classification_error_after": classification_error_after,
                "classification_error_induced": classification_error_induced
            }

    output_file = "LSTM_layers.json"
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)
        
def loop_through_settings():
    settings = [
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 4, 'frac_bits': 4, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 6, 'frac_bits': 2, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 8, 'frac_bits': 8, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 9, 'frac_bits': 32, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2])},
    ]
    results = {}
    for ber in ber_values:
        results[ber] = {}
        for setting in settings:
            model_copy = LSTMNet(input_dim=10000, hidden_dim=32, output_dim=1, n_layers=2, dropout=0.5).to(device)
            model_copy.load_state_dict(model.state_dict())

            # Test the model before fault injection
            model_copy.eval()
            with torch.no_grad():
                outputs_before = model_copy(x_test)
                predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
                accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
                classification_error_before = 1 - accuracy_before

            # Inject faults
            model_with_faults = dnn_fi_temp(model_copy, seed=0, ber=ber, **setting)

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
    output_file = "LSTM_settings.json"
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    #loop_through_ber_values()
    # Perform fault injection with different layers
    loop_through_layers()
    # Perform fault injection with different settings
    #loop_through_settings()