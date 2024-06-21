import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from GAM import GAM, ModelArgs


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load training data from jsonl file
train_data_path = "./training_train_data.jsonl"
test_data_path = "./training_test_data.jsonl"

num_tokens = 10
train_data = load_data(train_data_path)
train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_data = load_data(test_data_path)
test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# train setup
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAM(ModelArgs())
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(f"Number of trainable parameters: {count_parameters(model)}")

# train
for epoch in range(num_epochs):
    model.train()
    for i, data_element in enumerate(train_loader):
        input = data_element[0].to(device)
        target = torch.cat((input[0][1:], torch.tensor([10]))).unsqueeze(0)
        output = model(input, target)
        loss = model.last_loss

        if torch.isnan(loss).any():
            print(f"NaN loss detected at batch {i} of epoch {epoch}")
            print("Output:", output)
            print("Target:", target)
            continue

        if (i + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}")
            # print("Target sample:", target[:10])

            # predictions = []
            # for j in range(10):
            #     predictions.append(torch.argmax(output[0][j][:10], dim=-1))

            # print("Predicted:", predictions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
