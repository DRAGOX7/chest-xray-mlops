import torch
import torch.nn.functional as F
import io
import base64
import numpy as np
import traceback  # <--- NEW: To track errors
import matplotlib

matplotlib.use('Agg')  # Keep this!
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
from model import build_model

app = FastAPI(title="ABDULLAH AI - DEBUG MODE")

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
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Target layer: The last block of 'features' in DenseNet
    target_layer = model.features[-1]

    hook_handle_fwd = target_layer.register_forward_hook(forward_hook)
    hook_handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_idx = output.argmax(dim=1).item()

    model.zero_grad()
    score = output[0, pred_idx]
    score.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    hook_handle_fwd.remove()
    hook_handle_bwd.remove()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)

    return cam, output


def overlay_heatmap(heatmap, original_image):
    heatmap = Image.fromarray(np.uint8(255 * heatmap))
    heatmap = heatmap.resize(original_image.size, resample=Image.BICUBIC)

    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(np.array(heatmap) / 255.0)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored)

    overlay = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=0.5)
    return overlay


# --- API ENDPOINTS ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Load Image
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # 2. Preprocess
        tensor = transform_preprocess(original_image).unsqueeze(0).to(DEVICE)
        tensor.requires_grad = True

        # 3. Get Prediction & Grad-CAM
        heatmap_raw, output = get_gradcam(model, tensor)

        # 4. Calculate Probabilities
        probabilities = F.softmax(output, dim=1)
        abnormal_prob = probabilities[0][1].item()

        # 5. Create Overlay Image
        overlay_img = overlay_heatmap(heatmap_raw, original_image)

        # 6. Convert Overlay to Base64
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "filename": file.filename,
            "abnormality_probability": f"{abnormal_prob:.4f}",
            "diagnosis": "Abnormal" if abnormal_prob > 0.5 else "Normal",
            "gradcam_image": img_str,
            "error": None  # No error!
        }

    except Exception as e:
        # CATCH THE ERROR AND RETURN IT
        return {
            "filename": "error",
            "abnormality_probability": "0.0",
            "diagnosis": "Error",
            "gradcam_image": "",
            "error": str(e),
            "traceback": traceback.format_exc()
        }