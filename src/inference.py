import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(path='models/resnet18_cifar10.pth'):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict(image_path, model_path='models/resnet18_cifar10.pth'):
    # Load model
    model = load_model(model_path)
    transform = get_inference_transforms()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()

    predicted_class = CLASSES[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class} ({confidence:.1f}% confidence)")
    print("\nAll class probabilities:")
    for cls, prob in sorted(zip(CLASSES, probabilities.tolist()),
                            key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 40)
        print(f"  {cls:<12} {prob*100:5.1f}%  {bar}")

    return predicted_class, confidence


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        print("Usage: python -m src.inference <path_to_image>")
        sys.exit(1)
    predict(image_path)