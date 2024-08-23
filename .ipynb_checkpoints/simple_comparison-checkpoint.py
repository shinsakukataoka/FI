from keras.datasets import imdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Define the MLP model
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(10000, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)).view(-1, 1)
        return x

# Define the LSTM model
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = self.dropout(h_n[-1]) # Apply dropout on the last hidden state
        x = self.fc(h_n)
        x = torch.sigmoid(x).view(-1, 1)
        return x

# Define the function for vectorizing the sequences
def vectorize_sequences(sequences, dimension=10000):
    results = torch.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Define the function for padding and truncating the sequences
def pad_sequences(sequences, max_length):
    padded_sequences = torch.zeros((len(sequences), max_length)).long()
    for i, sequence in enumerate(sequences):
        sequence = sequence[:max_length]
        padded_sequences[i, :len(sequence)] = torch.tensor(sequence)
    return padded_sequences

# Data preparation for MLP
x_train_mlp = vectorize_sequences(train_data)
x_test_mlp = vectorize_sequences(test_data)
y_train_mlp = torch.tensor(train_labels).float().view(-1, 1)
y_test_mlp = torch.tensor(test_labels).float().view(-1, 1)

# Data preparation for LSTM
max_length = 200
x_train_lstm = pad_sequences(train_data, max_length=max_length)
x_test_lstm = pad_sequences(test_data, max_length=max_length)
y_train_lstm = torch.tensor(train_labels).float().view(-1, 1)
y_test_lstm = torch.tensor(test_labels).float().view(-1, 1)

# Train MLP
model_mlp = MLPNet()
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model_mlp.parameters())
train_data_mlp = TensorDataset(x_train_mlp, y_train_mlp)
train_loader_mlp = DataLoader(train_data_mlp, batch_size=512)

for epoch in range(10):
    for inputs, targets in train_loader_mlp:
        optimizer.zero_grad()
        outputs = model_mlp(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Train LSTM
model_lstm = LSTMNet(input_dim=10000, hidden_dim=256, output_dim=1, num_layers=2, dropout=0.5)
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model_lstm.parameters(), lr=0.001)
train_data_lstm = TensorDataset(x_train_lstm, y_train_lstm)
train_loader_lstm = DataLoader(train_data_lstm, batch_size=50)

for epoch in range(4):
    for inputs, targets in train_loader_lstm:
        optimizer.zero_grad()
        outputs = model_lstm(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), max_norm=5)
        optimizer.step()

# Test MLP
model_mlp.eval()
with torch.no_grad():
    outputs_mlp = model_mlp(x_test_mlp)
    predictions_mlp = (outputs_mlp > 0.5).type(torch.IntTensor).view(-1)
    accuracy_mlp = accuracy_score(y_test_mlp.numpy().flatten(), predictions_mlp.numpy())

# Test LSTM
model_lstm.eval()
with torch.no_grad():
    outputs_lstm = model_lstm(x_test_lstm)
    predictions_lstm = (outputs_lstm > 0.5).type(torch.IntTensor).view(-1)
    accuracy_lstm = accuracy_score(y_test_lstm.numpy().flatten(), predictions_lstm.numpy())

print(f"Accuracy of MLP: {accuracy_mlp}")
print(f"Accuracy of LSTM: {accuracy_lstm}")