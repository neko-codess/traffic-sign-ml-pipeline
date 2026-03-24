from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import io

app = FastAPI()

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(path='models/resnet18_cifar10.pth'):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


# Load once at startup — not on every request
model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/")
def root():
    return {"message": "Traffic Sign Classifier API", "status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Preprocess
    tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()

    return JSONResponse({
        "prediction": CLASSES[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "all_probabilities": {
            cls: round(prob * 100, 2)
            for cls, prob in zip(CLASSES, probabilities.tolist())
        }
    })