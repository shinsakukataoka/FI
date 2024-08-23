import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from fault_injection import dnn_fi
from models import Net, MNISTNet, CIFAR10Net  
from torchvision import datasets, transforms
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loader(name, path, batch_size):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    if name == "MNIST":
        trainset = datasets.MNIST(path, download=True, train=True, transform=transform)
    else:
        trainset = datasets.CIFAR10(path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader

models = {
    "MNISTNet": MNISTNet(),
    "CIFAR10Net": CIFAR10Net()
}

datasets = [
    {"name": "MNIST", "loader": get_data_loader("MNIST", '~/tmp/MNIST', 64)},
    # {"name": "CIFAR10", "loader": get_data_loader("CIFAR10", '~/tmp/CIFAR10', 64)}
]

compatibility_matrix = {
    "MNISTNet": ["MNIST"],
    "CIFAR10Net": ["CIFAR10"]
}

def train_model(model, trainloader, criterion, optimizer, epochs, device):
    model = model.to(device)
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

    return model

def predict(model, test_images):
    output = model(test_images)
    preds = torch.argmax(output, dim=1)
    return preds

# Use np.logspace to generate 50 values between 1E-9 and 1E-1 on a log scale
bers = np.logspace(-9, -1, 4)

results = []
error_rates = {}

for ber in bers:
    for dataset in datasets:
        for model_name, model in models.items():
            if dataset["name"] not in compatibility_matrix[model_name]:
                print(f"Skipping incompatible model-data pair: {model_name}-{dataset['name']}")
                continue

            print(f"Testing {model_name} with {dataset['name']}...")

            dataloader = dataset["loader"]
            test_images, test_labels = next(iter(dataloader))
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            # Prepare model and train it
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.003)
            model = train_model(model, dataloader, criterion, optimizer, 5, device)

            # Predict with the original model
            original_preds = predict(model, test_images)

            # Fault injection parameters
            seed = 0
            int_bits = 2
            frac_bits = 6
            rep_conf = np.array([4,4,4,4])
            q_type = 'signed'
            encode = 'dense'

            # Perform fault injection on the model
            faulty_model = dnn_fi(model, seed=seed, int_bits=int_bits, frac_bits=frac_bits, rep_conf=rep_conf, q_type=q_type, encode=encode, ber = ber)

            # Predict with the faulty model
            faulty_preds = predict(faulty_model, test_images)

            # Calculate the error rate
            error_rate = torch.mean((original_preds != faulty_preds).float()).item() * 100
            
            error_rates[(model_name, dataset["name"])] = error_rate

            # Save the results in a dictionary
            result = {
                "model": model_name,
                "dataset": dataset["name"],
                "ber": ber,
                "error_rate": error_rate
            }
            results.append(result)
            

# Convert the results to a Pandas DataFrame for easier analysis and plotting
results_df = pd.DataFrame(results)

# Now, let's plot the results using matplotlib
for (model, dataset), group in results_df.groupby(["model", "dataset"]):
    plt.loglog(group["ber"], group["error_rate"], label=f"{model} on {dataset}")

plt.xlabel("Bit Error Rate (BER)")
plt.ylabel("Classification Error Rate (%)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("ber_vs_error_rate.png")

for (model_name, dataset_name), error_rate in error_rates.items():
    print(f"Classification error rate for {model_name} on {dataset_name}: {error_rate:.2f}%")