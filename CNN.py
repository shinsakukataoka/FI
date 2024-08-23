import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
import numpy as np
import json
from fault_injection import dnn_fi_temp
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

# Define the CNN model
class CNNNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, output_dim, dropout):
        super(CNNNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs) for fs in filter_sizes]
        )
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x => [batch_size, sent_len]
        embedded = self.embedding(x)
        # embedded => [batch_size, sent_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded => [batch_size, emb_dim, sent_len]
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        # conved_n => [batch_size, num_filters, sent_len - filter_sizes[n] + 1]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n => [batch_size, num_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat => [batch_size, num_filters * len(filter_sizes)]
        return torch.sigmoid(self.fc(cat)).view(-1, 1)
    
ber_values = np.logspace(np.log10(10 * 10**-10), np.log10(10 * 10**-2), num=1000)

# Training the model
model = CNNNet(vocab_size=10000, embed_dim=32, num_filters=32, filter_sizes=[3, 4, 5], output_dim=1, dropout=0.5).to(device)
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
            model_copy = CNNNet(vocab_size=10000, embed_dim=32, num_filters=32, filter_sizes=[3, 4, 5], output_dim=1, dropout=0.5).to(device)
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
    output_file = "CNN_loops.json"
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

def loop_through_layers():
    settings = [
        {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        #{'q_type': 'unsigned', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        #{'q_type': 'unsigned', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}
    ]
    results = {}
    layer_names = [name for name, _ in model.named_parameters()]  # Get list of layer names
    ber = 0.01  # Set the bit error rate to 0.005 for all layers
    for layer in layer_names:
        results[layer] = {}

        for setting in settings:
            model_copy = CNNNet(vocab_size=10000, embed_dim=32, num_filters=32, filter_sizes=[3, 4, 5], output_dim=1, dropout=0.5).to(device)
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

    output_file = "CNN_layers.json"
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)
        
def loop_through_settings():
    settings = [
        {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'unsigned', 'encode': 'bitmask', 'int_bits': 4, 'frac_bits': 4, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'bitmask', 'int_bits': 6, 'frac_bits': 2, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'signed', 'encode': 'dense', 'int_bits': 8, 'frac_bits': 8, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2])},
        {'q_type': 'afloat', 'encode': 'dense', 'int_bits': 9, 'frac_bits': 32, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2, 2, 2])},
    ]
    results = {}
    for setting in settings:
        setting_copy = {key: value for key, value in setting.items() if key != 'rep_conf'}
        results[json.dumps(setting_copy)] = {}
        model_copy = CNNNet(vocab_size=10000, embed_dim=32, num_filters=32, filter_sizes=[3, 4, 5], output_dim=1, dropout=0.5).to(device)
        model_copy.load_state_dict(model.state_dict())

        # Test the model before fault injection
        model_copy.eval()
        with torch.no_grad():
            outputs_before = model_copy(x_test)
            predictions_before = (outputs_before > 0.5).type(torch.IntTensor).view(-1)
            accuracy_before = accuracy_score(y_test.cpu().numpy().flatten(), predictions_before.cpu().numpy())
            classification_error_before = 1 - accuracy_before

        # Inject faults
        model_with_faults = dnn_fi_temp(model_copy, seed=0, ber=0.005, **setting)

        # Test the model after fault injection
        model_with_faults.eval()
        with torch.no_grad():
            outputs_after = model_with_faults(x_test)
            predictions_after = (outputs_after > 0.5).type(torch.IntTensor).view(-1)
            accuracy_after = accuracy_score(y_test.cpu().numpy().flatten(), predictions_after.cpu().numpy())
            classification_error_after = 1 - accuracy_after

        classification_error_induced = classification_error_after - classification_error_before

        setting_summary = f"{setting['q_type']}-{setting['encode']}-{setting['int_bits']}-{setting['frac_bits']}"
        results[setting_summary] = {
            "accuracy_before": accuracy_before,
            "classification_error_before": classification_error_before,
            "accuracy_after": accuracy_after,
            "classification_error_after": classification_error_after,
            "classification_error_induced": classification_error_induced
        }

    output_file = "CNN_settings.json"
    # Save results to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    #loop_through_ber_values()
    # Perform fault injection with different layers
    loop_through_layers()
    # Perform fault injection with different settings
    #loop_through_settings()