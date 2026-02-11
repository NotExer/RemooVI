import os
import io
import time
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
# U2NETP is the lightweight version (~4.7 MB)
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = os.path.join(os.environ.get("U2NET_HOME", "/tmp"), "u2netp.onnx")

# --- Model Management ---
def download_model(url, path):
    if os.path.exists(path):
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading model from {url} to {path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def get_session():
    # Download model if not exists
    download_model(MODEL_URL, MODEL_PATH)
    # Create ONNX Runtime session
    return ort.InferenceSession(MODEL_PATH)

# Initialize session globally to reuse across requests
# In a true serverless cold start, this will run once per instance
try:
    session = get_session()
except Exception as e:
    print(f"Error initializing model: {e}")
    session = None

# --- Image Processing ---
def preprocess(image: Image.Image):
    # Resize to 320x320 (standard for U2Net)
    input_image = image.resize((320, 320), Image.Resampling.BILINEAR)
    
    # Convert to numpy array and normalize
    img_array = np.array(input_image).astype(np.float32)
    
    # Normalize: (x - mean) / std
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    img_array /= 255.0
    img_array -= np.array([0.485, 0.456, 0.406])
    img_array /= np.array([0.229, 0.224, 0.225])
    
    # Transpose to (Batch, Channel, Height, Width)
    img_array = img_array.transpose((2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def postprocess(prediction, original_image: Image.Image):
    # Prediction is (1, 1, 320, 320)
    # Remove batch and channel dims -> (320, 320)
    mask = prediction[0][0]
    
    # Normalize to 0-255
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask * 255).astype(np.uint8)
    
    # Convert to PIL Image
    mask_image = Image.fromarray(mask, mode='L')
    
    # Resize mask to original image size
    mask_image = mask_image.resize(original_image.size, Image.Resampling.BILINEAR)
    
    # Create new image with transparent background
    empty = Image.new("RGBA", original_image.size, 0)
    image_with_alpha = original_image.convert("RGBA")
    
    # Composite: use mask to paste original onto empty
    return Image.composite(image_with_alpha, empty, mask_image)

# --- FastAPI Application ---
app = FastAPI(root_path="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Remoover API (Lightweight) is running"}

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    if session is None:
         return {"error": "Model not initialized"}

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Inference
    input_data = preprocess(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    result = session.run([output_name], {input_name: input_data})
    
    # Post-process
    output_image = postprocess(result[0], image)
    
    # Return result
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
