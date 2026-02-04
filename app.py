import torch
import torch.nn.functional as F
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
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
transform_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- GRAD-CAM HELPER FUNCTIONS ---
def get_gradcam(model, image_tensor):
    # 1. Hook into the last convolutional layer (DenseNet121 features)
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Target layer: The last block of 'features' in DenseNet
    target_layer = model.features[-1]

    # Register hooks
    hook_handle_fwd = target_layer.register_forward_hook(forward_hook)
    hook_handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # 2. Forward Pass
    output = model(image_tensor)
    pred_idx = output.argmax(dim=1).item()

    # 3. Backward Pass (Calculate Gradients)
    model.zero_grad()
    score = output[0, pred_idx]
    score.backward()

    # Get stored gradients and activations
    grads = gradients[0].cpu().data.numpy()[0]  # [1024, 7, 7]
    acts = activations[0].cpu().data.numpy()[0]  # [1024, 7, 7]

    # Clean up hooks
    hook_handle_fwd.remove()
    hook_handle_bwd.remove()

    # 4. Generate Heatmap
    # Average gradients spatially (Global Average Pooling)
    weights = np.mean(grads, axis=(1, 2))

    # Multiply activations by weights
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU (ignore negative values)
    cam = np.maximum(cam, 0)

    # Normalize 0-1
    cam = cam / (np.max(cam) + 1e-8)  # Add epsilon to avoid div by zero

    return cam, output


def overlay_heatmap(heatmap, original_image):
    # Resize heatmap to match original image
    heatmap = Image.fromarray(np.uint8(255 * heatmap))
    heatmap = heatmap.resize(original_image.size, resample=Image.BICUBIC)

    # Apply colormap (Jet is standard for heatmaps)
    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(np.array(heatmap) / 255.0)  # Returns RGBA
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)  # Drop Alpha
    heatmap_img = Image.fromarray(heatmap_colored)

    # Blend images (50% original, 50% heatmap)
    overlay = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=0.5)
    return overlay


# --- API ENDPOINTS ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Load Image
    image_data = await file.read()
    original_image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # 2. Preprocess
    tensor = transform_preprocess(original_image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad = True  # IMPORTANT for Grad-CAM

    # 3. Get Prediction & Grad-CAM
    heatmap_raw, output = get_gradcam(model, tensor)

    # 4. Calculate Probabilities
    probabilities = F.softmax(output, dim=1)
    abnormal_prob = probabilities[0][1].item()

    # 5. Create Overlay Image
    overlay_img = overlay_heatmap(heatmap_raw, original_image)

    # 6. Convert Overlay to Base64 string to send over JSON
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "filename": file.filename,
        "abnormality_probability": f"{abnormal_prob:.4f}",
        "diagnosis": "Abnormal" if abnormal_prob > 0.5 else "Normal",
        "gradcam_image": img_str  # <--- NEW FIELD
    }