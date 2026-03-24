import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.dataset import get_dataloaders

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(path='models/resnet18_cifar10.pth'):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def evaluate():
    device = torch.device('cpu')
    print("Running evaluation on CPU...")

    _, val_loader = get_dataloaders()
    model = load_model().to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

            if i % 50 == 0:
                print(f"  Batch {i}/{len(val_loader)}...")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix — ResNet18 CIFAR10')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("\nSaved to confusion_matrix.png")


if __name__ == "__main__":
    evaluate()