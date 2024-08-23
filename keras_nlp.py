import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from fault_injection import dnn_fi
from keras.datasets import imdb
import numpy as np

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

model = Net().to(device)

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

# Currently I am oly using settings[0] but in the futrue loop throgh them
    
settings = [
    {'q_type': 'signed', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'unsigned', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'afloat', 'encode': 'bitmask', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'unsigned', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'afloat', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}
]

setting = settings[0]

# Currently I am not using this but in the future loop through this

ber_values = [0.001,0.005,0.01]
ber_value = ber_values[0]

# Inject faults
model_with_faults = dnn_fi(model, seed=0, ber = ber, **setting)

# Test the model after fault injection
model_with_faults.eval()
with torch.no_grad():
    outputs_after = model_with_faults(x_test)
    predictions_after = (outputs_after > 0.5).type(torch.IntTensor).view(-1)
    accuracy_after = accuracy_score(y_test.cpu().numpy().flatten(), predictions_after.cpu().numpy())
    classification_error_after = 1 - accuracy_after

classification_error_induced = classification_error_after - classification_error_before

print("Before fault injection:")
print("Accuracy: {:.2f}".format(accuracy_before))
print("Classification error: {:.2f}".format(classification_error_before))

print("\nAfter fault injection:")
print("Accuracy: {:.2f}".format(accuracy_after))
print("Classification error: {:.2f}".format(classification_error_after))

print("\nClassification error induced by fault injection: {:.2f}".format(classification_error_induced))