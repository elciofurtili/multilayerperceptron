import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

base_path = "./dataset/2"
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

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, X_val, y_val,
                lr=0.01, max_epochs=100, patience=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

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

            print(f"[{epoch+1}] Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break

    model.load_state_dict(best_model_state)
    return model, history

layer_sizes = [50, 16, 8, 4]  
activation = 'relu'
learning_rate = 0.01

model = MLP(layer_sizes, activation=activation)
model, history = train_model(model, X_train, y_train, X_val, y_val,
                             lr=learning_rate, max_epochs=100, patience=20)

with torch.no_grad():
    model.eval()
    test_output = model(X_test)
    test_preds = torch.argmax(test_output, dim=1)
    test_acc = (test_preds == y_test.squeeze()).float().mean().item()
    print(f"Acurácia no conjunto de teste: {test_acc:.4f}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title("Perda")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title("Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acc")
plt.legend()

plt.tight_layout()
plt.show()