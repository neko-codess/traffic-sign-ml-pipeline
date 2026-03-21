import torch
import os
import torch.nn as nn
import torchvision.models as models
from src.dataset import get_dataloaders


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers — we don't want to change what ResNet already knows
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer for our 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


def train(num_epochs=10, batch_size=32, learning_rate=0.001):
    # Setup
    os.makedirs('models', exist_ok=True) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader = get_dataloaders(batch_size)
    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           # Step 3: clear old gradients
            outputs = model(images)         # Step 1: forward pass
            loss = criterion(outputs, labels)  # Step 2: calculate loss
            loss.backward()                 # Step 4: backward pass
            optimizer.step()               # Step 5: update weights

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "models/resnet18_cifar10.pth")
    print("Model saved to models/resnet18_cifar10.pth")


if __name__ == "__main__":
    train(num_epochs=10)