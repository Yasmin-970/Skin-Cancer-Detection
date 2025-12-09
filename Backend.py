import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from PIL import Image
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Working Directory: {BASE_DIR}")

def load_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2),
    )

    model_path = os.path.join(BASE_DIR, "final_best_melanoma_model.pth")
    print(f"🔍 Looking for model at: {model_path}")
    
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print("Model Loaded Successfully!")
            return model
        else:
            print("ERROR: Model file not found! Please check the file name.")
            return None
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded properly. Check server console."}
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        classes = ["Benign", "Malignant"]
        return {
            "prediction": classes[predicted.item()], 
            "confidence": float(confidence.item())
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def read_index():
    return RedirectResponse(url="/static/Home Page.html")

static_dir = os.path.join(BASE_DIR, "static")
print(f"🔍 Looking for static files at: {static_dir}")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print("Static folder found and mounted.")
else:
    print("ERROR: 'static' folder not found!")

if __name__ == "__main__":
    print("Starting server...")
    try:
        uvicorn.run("Backend:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print("Reload mode failed, switching to standard mode...")
        uvicorn.run(app, host="127.0.0.1", port=8000)