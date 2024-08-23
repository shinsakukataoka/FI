# Importing the necessary libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from fault_injection import dnn_fi_temp
import json
from MNIST import Net  # Assuming your model architecture is in a file named 'model.py'

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model and move it to the defined device
model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))

# Define the test set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)

# Evaluation function
def evaluate_model(model):
    model.eval()
    test_loss = 0
    correct = 0
    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# Evaluate the original model
orig_loss, orig_accuracy = evaluate_model(model)
print('Original model: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(orig_loss, orig_accuracy))

# Parameters for dnn_fi_temp
params = {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}

# Generate the BER values
#ber_values_1 = np.logspace(-9, -5, num=50)
#ber_values_2 = np.logspace(-5, -2, num=250)
#ber_values = np.concatenate((ber_values_1, ber_values_2))
ber_values = np.logspace(-9, -2, num=500)

results = {}

# Loop over BER values
for ber in ber_values:
    faulty_model = dnn_fi_temp(model, **params, ber=ber)
    faulty_model.to(device)  # Ensure the faulty model is on the correct device
    faulty_loss, faulty_accuracy = evaluate_model(faulty_model)
    print('Faulty model (BER = {}): Average loss: {:.4f}, Accuracy: {:.2f}%'.format(ber, faulty_loss, faulty_accuracy))
    results[str(ber)] = {
        'average_loss': faulty_loss,
        'accuracy': faulty_accuracy
    }

# Save the results to a JSON file
with open('ber_evaluation_results.json', 'w') as f:
    json.dump(results, f)
