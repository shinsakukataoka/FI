import torch
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from fault_injection import dnn_fi
from models import Net, SimpleNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('~/tmp', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/tmp', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define and train model
model = SimpleNet().to(device)

# Move model to CPU for quantization
model.to('cpu')

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Move model back to original device for further training
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        log_ps = model(images)
        loss = criterion(log_ps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss}")

def calculate_accuracy(loader, model):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Move data and model to CPU for evaluation
device = torch.device('cpu')
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
model.to(device)

original_accuracy = calculate_accuracy(testloader, model)
print(f"Original model accuracy: {original_accuracy:.2f}%")

quantized_accuracy = calculate_accuracy(testloader, quantized_model)
print(f"Quantized model accuracy: {quantized_accuracy:.2f}%")

# Move model back to GPU for fault injection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


settings = [
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])},
    {'q_type': 'signed', 'encode': 'dense', 'int_bits': 9, 'frac_bits': 23, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])},
]

for setting in settings:
    print(f"Testing setting: {setting}")
    faulty_model = dnn_fi(model, ber=0.05, seed=0, **setting)
    faulty_accuracy = calculate_accuracy(testloader, faulty_model)
    print(f"Faulty model accuracy: {faulty_accuracy:.2f}%")
    print(f"Accuracy drop due to fault injection: {original_accuracy - faulty_accuracy:.2f}%")