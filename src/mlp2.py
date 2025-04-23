import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import product

base_path = "./dataset/4"
train_path = os.path.join(base_path, "train_dataset.csv")
val_path = os.path.join(base_path, "validation_dataset.csv")
test_path = os.path.join(base_path, "test_dataset.csv")

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('label', axis=1).values.T 
    y = df['label'].values.reshape(1, -1) 
    return torch.tensor(X.T, dtype=torch.float32), torch.tensor(y.T, dtype=torch.long)

X_train, y_train = load_data(train_path)
X_val, y_val = load_data(val_path)
X_test, y_test = load_data(test_path)

architectures = {
    "Pequena": [50, 64, 4],
    "Média": [50, 128, 64, 4],
    "Grande": [50, 256, 128, 64, 4]
}

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation='relu', dropout=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                if dropout:
                    layers.append(nn.Dropout(p=0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, X_val, y_val, optimizer_name='SGD',
                activation='relu', lr=0.01, weight_decay=0.0, max_epochs=100, patience=20):
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train.squeeze())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val.squeeze())

            train_preds = torch.argmax(output, dim=1)
            val_preds = torch.argmax(val_output, dim=1)
            train_acc = (train_preds == y_train.squeeze()).float().mean().item()
            val_acc = (val_preds == y_val.squeeze()).float().mean().item()

            history['train_loss'].append(loss.item())
            history['val_loss'].append(val_loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    model.load_state_dict(best_model_state)
    return model, history, history['val_acc'][-1], history['val_loss'][-1]

optimizers = {'SGD': 0.01, 'Adam': 0.001}
activations = ['relu', 'tanh']
regularizations = {
    'none': {'dropout': False, 'weight_decay': 0.0},
    'dropout': {'dropout': True, 'weight_decay': 0.0},
    'l2': {'dropout': False, 'weight_decay': 0.001}
}

results = []

for arch_name, arch in architectures.items():
    for opt_name, lr in optimizers.items():
        for act in activations:
            for reg_name, reg_params in regularizations.items():
                print(f">>> {arch_name} | {opt_name} | {act} | {reg_name}")
                model = MLP(arch, activation=act, dropout=reg_params['dropout'])
                model, history, val_acc, val_loss = train_model(
                    model, X_train, y_train, X_val, y_val,
                    optimizer_name=opt_name,
                    activation=act,
                    lr=lr,
                    weight_decay=reg_params['weight_decay'],
                    max_epochs=100,
                    patience=20
                )
                with torch.no_grad():
                    test_output = model(X_test)
                    test_preds = torch.argmax(test_output, dim=1)
                    test_acc = (test_preds == y_test.squeeze()).float().mean().item()
                results.append({
                    'arquitetura': arch_name,
                    'otimizador': opt_name,
                    'ativação': act,
                    'regularização': reg_name,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'test_acc': test_acc
                })

df_results = pd.DataFrame(results)
print(df_results.sort_values(by='test_acc', ascending=False).to_string(index=False))