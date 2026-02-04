import torch
import torch.nn.functional as F
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
from model import build_model 

app = FastAPI(title="ABDULLAH AI - VERSION 2")

# --- CONFIGURATION ---
MODEL_PATH = "best_densenet121.pth"
DEVICE = "cpu" 

# --- LOAD MODEL ---
model = build_model(num_classes=2) 
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

model.to(DEVICE)
model.eval()

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor) 
        probabilities = F.softmax(outputs, dim=1)
        abnormal_prob = probabilities[0][1].item()
        
    return {
        "filename": file.filename,
        "abnormality_probability": f"{abnormal_prob:.4f}",
        "diagnosis": "Abnormal" if abnormal_prob > 0.5 else "Normal"
    }
