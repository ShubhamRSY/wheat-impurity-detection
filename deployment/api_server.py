import io, json, base64, time
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import cv2

app = FastAPI(title="Wheat Quality Inspection", version="2.0.0")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/wheat_segmentation_v2.pt"
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

CLASSES = ["Background", "Wheat", "Wheat_Bran", "Straw", "Weed", "Gravel", "Glass"]
IMP_IDS = [2, 3, 4, 5, 6]
SZ = 256

PRED_TOTAL = Counter("wheat_predictions_total", "Total predictions")
PRED_DUR = Histogram("wheat_predict_duration_seconds", "Prediction latency", buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
IR_GAUGE = Gauge("wheat_impurity_rate", "Estimated impurity rate")
CONTAM = Counter("wheat_contaminated_samples_total", "Count of contaminated samples")


def preprocess(img):
    img = cv2.resize(img, (SZ, SZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def decode_mask(mask):
    clr = np.array([[0, 0, 0], [46, 204, 113], [231, 76, 60], [243, 156, 18],
                    [155, 89, 182], [52, 152, 219], [26, 188, 156]], dtype=np.uint8)
    return clr[mask]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Not an image")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image")
    oh, ow = image.shape[:2]
    tensor = preprocess(image)
    start = time.time()
    with torch.no_grad():
        pred = model(tensor)["out"].argmax(dim=1).squeeze(0).cpu().numpy()
    dur = time.time() - start
    PRED_TOTAL.inc()
    PRED_DUR.observe(dur)
    pred_full = cv2.resize(pred.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
    _, enc = cv2.imencode(".png", decode_mask(pred_full))
    imp_px = sum((pred_full == c).sum() for c in IMP_IDS)
    ir = float(imp_px / pred_full.size)
    IR_GAUGE.set(ir)
    if ir > 0.05:
        CONTAM.inc()
    cc = {}
    for i, n in enumerate(CLASSES):
        cnt = int((pred_full == i).sum())
        if cnt > 0:
            cc[n] = {"pixels": cnt, "percentage": float(cnt / pred_full.size)}
    return JSONResponse({
        "impurity_rate": ir,
        "class_composition": cc,
        "segmentation_mask": "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode(),
        "image_dimensions": {"width": ow, "height": oh},
        "quality_assessment": "contaminated" if ir > 0.05 else "clean",
        "latency": dur,
        "model": "deeplabv3_mobilenetv3"
    })


@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(DEVICE), "model": "DeepLabV3-MobileNetV3"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
