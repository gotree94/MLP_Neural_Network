"""
train_mnist_mlp.py
==================
Train a 3-layer MLP on MNIST dataset using PyTorch.
Output: trained weights and biases as NumPy arrays.

Network: 784 (input) → 64 (hidden1) → 32 (hidden2) → 10 (output)
Activation: ReLU (hidden layers)
Loss: CrossEntropyLoss
Optimizer: Adam

Usage:
    python train_mnist_mlp.py

Outputs:
    models/mnist_mlp_weights.npz  — trained parameters (FP32)
    models/mlp_traced.pt          — TorchScript model for export
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import argparse


# =========================================================================
#  MLP Model Definition
# =========================================================================
class MLP(nn.Module):
    """3-layer Multi-Layer Perceptron for MNIST digit recognition.

    Architecture:
        Input (784) → Linear(784→64) → ReLU →
        Linear(64→32) → ReLU →
        Linear(32→10) → Output (logits)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten (batch, 1, 28, 28) → (batch, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)            # logits (CrossEntropyLoss includes softmax)
        return x


# =========================================================================
#  Training Configuration
# =========================================================================
class Config:
    batch_size = 64
    epochs = 10
    lr = 0.001
    seed = 42
    val_split = 0.1               # 10% of training set for validation
    model_dir = "models"


# =========================================================================
#  Training Functions
# =========================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def export_weights(model, save_dir):
    """Extract weights and biases as NumPy arrays and save to .npz file."""
    os.makedirs(save_dir, exist_ok=True)

    w1 = model.fc1.weight.detach().cpu().numpy()  # (64, 784)
    b1 = model.fc1.bias.detach().cpu().numpy()     # (64,)
    w2 = model.fc2.weight.detach().cpu().numpy()   # (32, 64)
    b2 = model.fc2.bias.detach().cpu().numpy()      # (32,)
    w3 = model.fc3.weight.detach().cpu().numpy()   # (10, 32)
    b3 = model.fc3.bias.detach().cpu().numpy()      # (10,)

    np.savez(os.path.join(save_dir, "mnist_mlp_weights.npz"),
             w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    # Also dump as Python-parseable text for debugging
    dump_txt(os.path.join(save_dir, "fc1_weight.txt"), w1)
    dump_txt(os.path.join(save_dir, "fc1_bias.txt"), b1)
    dump_txt(os.path.join(save_dir, "fc2_weight.txt"), w2)
    dump_txt(os.path.join(save_dir, "fc2_bias.txt"), b2)
    dump_txt(os.path.join(save_dir, "fc3_weight.txt"), w3)
    dump_txt(os.path.join(save_dir, "fc3_bias.txt"), b3)

    print(f"[INFO] Weights exported to {save_dir}/")
    print(f"  FC1: {w1.shape}  ({w1.size * 4 / 1024:.1f} KB)")
    print(f"  FC2: {w2.shape}  ({w2.size * 4 / 1024:.1f} KB)")
    print(f"  FC3: {w3.shape}  ({w3.size * 4 / 1024:.1f} KB)")


def dump_txt(filepath, array):
    """Dump NumPy array to human-readable text file."""
    np.savetxt(filepath, array, fmt="%.8f")


# =========================================================================
#  Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])

    # Load MNIST
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Model, loss, optimizer
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"[INFO] MLP parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Training for {args.epochs} epochs...\n")

    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion,
                                 optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(
                Config.model_dir, "best_model.pth"))

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"Loss: {train_loss:.4f}  "
              f"Test Acc: {test_acc:.2f}%  "
              f"{'★ BEST' if test_acc == best_acc else ''}")

    print(f"\n[INFO] Best test accuracy: {best_acc:.2f}%")

    # Export weights
    model.load_state_dict(torch.load(
        os.path.join(Config.model_dir, "best_model.pth")))
    export_weights(model, Config.model_dir)

    # Export TorchScript for potential use in Vitis AI
    scripted_model = torch.jit.script(model.cpu())
    scripted_model.save(os.path.join(Config.model_dir, "mlp_traced.pt"))
    print(f"[INFO] TorchScript model saved.")

    # Run a single inference demo
    print("\n[INFO] Single inference demo:")
    demo_inference(model.cpu(), test_dataset)


def demo_inference(model, dataset):
    """Run a single inference and print the result."""
    model.eval()
    image, label = dataset[0]  # first test image
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()

    print(f"  Image label: {label}")
    print(f"  Predicted:   {pred}")
    print(f"  Confidence:  {probabilities[0][pred].item():.4f}")
    print(f"  Probabilities: {probabilities[0].numpy().round(3)}")


if __name__ == "__main__":
    main()
