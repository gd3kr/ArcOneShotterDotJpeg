import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

from model_v2 import Model, ModelArgs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_router_gradients(model):
    print("Router Gradients:")
    for name, param in model.router.named_parameters():
        if param.grad is not None:
            print(f"{name}:")
            print(f"  Shape: {param.grad.shape}")
            print(f"  Mean: {param.grad.mean().item():.5f}")
            print(f"  Std: {param.grad.std().item():.5f}")
            print(f"  Min: {param.grad.min().item():.5f}")
            print(f"  Max: {param.grad.max().item():.5f}")
        else:
            print(f"{name}: No gradient")
        print("------------------------")

class DictionaryDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'data': torch.tensor(item['data'], dtype=torch.long),
            'x_pos': torch.tensor(item['x_pos'], dtype=torch.long),
            'y_pos': torch.tensor(item['y_pos'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }

def collate_fn(batch):
    # This function will be used to collate the data in each batch
    return {
        'data': torch.stack([item['data'] for item in batch]),
        'x_pos': torch.stack([item['x_pos'] for item in batch]),
        'y_pos': torch.stack([item['y_pos'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }

# Load training data from jsonl file
train_data_path = "./global_processed_train_data.jsonl"
test_data_path = "./global_processed_test_data.jsonl"

# Load the datasets
train_dataset = DictionaryDataset('global_processed_train_data.jsonl')
test_dataset = DictionaryDataset('global_processed_test_data.jsonl')

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


# train setup
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(ModelArgs())
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(f"Number of trainable parameters: {count_parameters(model)}")

# train
for epoch in range(num_epochs):
    model.train()
    for i, data_element in enumerate(train_loader):
        input = data_element["data"].to(device)
        target = torch.cat((input[0][1:], torch.tensor([10]))).unsqueeze(0)
        x_pos = data_element["x_pos"].to(device)
        y_pos = data_element["y_pos"].to(device)
        attn_mask = data_element["attention_mask"].to(device)
        output = model(input, x_pos, y_pos, attn_mask, target)
        loss = model.last_loss
        # importance = model.last_importance

        print(loss.item())

        if torch.isnan(loss).any():
            print(f"NaN loss detected at batch {i} of epoch {epoch}")
            print("Output:", output)
            print("Target:", target)
            break

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
