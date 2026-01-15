"""Flask Backend for Waste Classification."""

from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import logging
from PIL import Image
import io
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "comparison_results_pytorch")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (224, 224)

CLASS_TO_BIN = {
    'battery': 'HAZARDOUS',
    'biological': 'BIO',
    'glass': 'GLASS',
    'metal_plastic': 'PLASTIC',
    'paper': 'PAPER',
    'textile': 'MIXED',
    'trash': 'MIXED'
}

BIN_COLORS = {
    'GLASS': '#4CAF50',     # Green
    'PLASTIC': '#FFC107',   # Yellow
    'PAPER': '#2196F3',     # Blue
    'BIO': '#8B4513',       # Brown
    'MIXED': '#212121',     # Black
    'HAZARDOUS': '#F44336'  # Red
}

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

device = None
class_names = []
yolo_model = None
loaded_models = {}


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ============================================================================
# RESOURCE LOADING
# ============================================================================


def load_resources():
    global device, class_names, yolo_model, loaded_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Load YOLO
    try:
        from ultralytics import YOLO
        print("[INFO] Loading YOLOv8 model...")
        yolo_model = YOLO("yolov8n.pt")
        print("[INFO] YOLOv8 model loaded.")
    except Exception as e:
        print(f"[WARN] Could not load YOLO model: {e}")
        yolo_model = None

    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    else:
        print(f"[WARN] class_names.json not found")
        return

    num_classes = len(class_names)

    # Load Feature Extractor (EfficientNet without classifier)
    print("[INFO] Loading EfficientNetB0 feature extractor...")
    feature_extractor = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT)
    feature_extractor.classifier = nn.Identity()
    feature_extractor = feature_extractor.to(device).eval()
    loaded_models["_feature_extractor"] = feature_extractor

    # Load MLP
    mlp_path = os.path.join(RESULTS_DIR, "mlp_classifier.pth")
    if os.path.exists(mlp_path):
        print(f"[INFO] Loading MLP from {mlp_path}")
        mlp = MLP(1280, num_classes).to(device)
        mlp.load_state_dict(torch.load(
            mlp_path, map_location=device, weights_only=True))
        mlp.eval()
        loaded_models["MLP"] = mlp
        print("[INFO] MLP loaded.")

    # Load SimpleCNN
    cnn_path = os.path.join(RESULTS_DIR, "simple_cnn.pth")
    if os.path.exists(cnn_path):
        print(f"[INFO] Loading SimpleCNN from {cnn_path}")
        cnn = SimpleCNN(num_classes).to(device)
        cnn.load_state_dict(torch.load(
            cnn_path, map_location=device, weights_only=True))
        cnn.eval()
        loaded_models["SimpleCNN"] = cnn
        print("[INFO] SimpleCNN loaded.")

    # Load MobileNet
    mobilenet_path = os.path.join(RESULTS_DIR, "mobilenet.pth")
    if os.path.exists(mobilenet_path):
        print(f"[INFO] Loading MobileNetV2 from {mobilenet_path}")
        mobilenet = models.mobilenet_v2(weights=None)
        mobilenet.classifier[1] = nn.Linear(
            mobilenet.last_channel, num_classes)
        mobilenet.load_state_dict(torch.load(
            mobilenet_path, map_location=device, weights_only=True))
        mobilenet = mobilenet.to(device).eval()
        loaded_models["MobileNetV2"] = mobilenet
        print("[INFO] MobileNetV2 loaded.")

    # Load EfficientNet (best model from training)
    effnet_path = os.path.join(RESULTS_DIR, "efficientnet_best.pth")
    if not os.path.exists(effnet_path):
        effnet_path = os.path.join(RESULTS_DIR, "efficientnet.pth")  # fallback
    if os.path.exists(effnet_path):
        print(f"[INFO] Loading EfficientNetB0 from {effnet_path}")
        effnet = models.efficientnet_b0(weights=None)
        effnet.classifier[1] = nn.Linear(
            effnet.classifier[1].in_features, num_classes)
        effnet.load_state_dict(torch.load(
            effnet_path, map_location=device, weights_only=True))
        effnet = effnet.to(device).eval()
        loaded_models["EfficientNetB0"] = effnet
        print("[INFO] EfficientNetB0 loaded.")

    print(f"[INFO] Loaded {len(loaded_models) - 1} classification models.")


load_resources()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def allowed_file(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def predict_with_model(model_name, pil_img):
    """Run prediction with a specific model."""
    transform = get_transform()
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    model = loaded_models[model_name]

    with torch.no_grad():
        if model_name == "MLP":
            features = loaded_models["_feature_extractor"](img_tensor)
            logits = model(features)
            logits = logits / 2.0  # Temperature scaling
        else:
            logits = model(img_tensor)

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return probs


def predict_ensemble(pil_img, models=["MLP", "EfficientNetB0", "MobileNetV2", "SimpleCNN"]):
    """Ensemble prediction using hard voting (majority vote)."""
    model_predictions = {}
    votes = {}

    for model_name in models:
        if model_name in loaded_models:
            try:
                probs = predict_with_model(model_name, pil_img)
                top_idx = int(np.argmax(probs))
                top_class = class_names[top_idx]
                top_conf = float(probs[top_idx])

                # Store vote
                if top_class not in votes:
                    votes[top_class] = []
                votes[top_class].append((model_name, top_conf))

                # Store individual model predictions for debug
                model_predictions[model_name] = {
                    "class": top_class,
                    "confidence": float(top_conf * 100)
                }
            except Exception as e:
                print(f"[WARN] {model_name} failed: {e}")

    if not votes:
        raise ValueError("No models available for prediction")

    # Find winner by vote count, then by sum of confidences
    winner_class = None
    winner_score = -1

    for cls, vote_list in votes.items():
        vote_count = len(vote_list)
        avg_conf = sum(c for _, c in vote_list) / vote_count
        score = vote_count + avg_conf  # Prioritize vote count, then confidence

        if score > winner_score:
            winner_score = score
            winner_class = cls

    # Build probability array (simple: 1.0 for winner, 0 for others)
    avg_probs = np.zeros(len(class_names))
    winner_idx = class_names.index(winner_class)

    # Use average confidence of winning votes as the final confidence
    winning_votes = votes[winner_class]
    avg_probs[winner_idx] = sum(
        c for _, c in winning_votes) / len(winning_votes)

    # Agreement: all models voted for same class
    all_voted_classes = list(votes.keys())
    agreement = len(all_voted_classes) == 1

    return avg_probs, agreement, model_predictions


# ============================================================================
# ROUTES
# ============================================================================
@app.route("/health")
def health():
    model_list = [k for k in loaded_models.keys() if not k.startswith("_")]
    return jsonify({
        "ok": True,
        "device": str(device),
        "models": model_list,
        "classes": class_names
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models", methods=["GET"])
def list_models():
    model_list = [k for k in loaded_models.keys() if not k.startswith("_")]
    return jsonify({"models": model_list})


@app.route("/predict_all", methods=["POST"])
def predict_all():
    """Runs prediction on ALL loaded models for each detected object."""
    if not class_names:
        return jsonify({"error": "Models not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w_img, h_img = pil_img.size

        # 1. Detect objects with YOLO (improved settings)
        detections = []
        MIN_AREA = 0.01  # Minimum 1% of image area
        MIN_CONF = 0.20  # Minimum confidence

        if yolo_model:
            results = yolo_model(pil_img, verbose=False,
                                 conf=MIN_CONF, imgsz=960, agnostic_nms=False)
            boxes = results[0].boxes
            IGNORED_CLASSES = [60]  # dining table

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Skip ignored COCO classes
                if cls_id in IGNORED_CLASSES:
                    continue

                # Skip low confidence
                if conf < MIN_CONF:
                    continue

                # Skip too small boxes
                area = ((x2 - x1) * (y2 - y1)) / (w_img * h_img)
                if area < MIN_AREA:
                    continue

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "area": area,
                    "score": conf * area  # Combined score for ranking
                })

            # Select up to 3 best boxes (highest score = conf * area)
            if detections:
                detections = sorted(
                    detections, key=lambda d: d["score"], reverse=True)[:3]

        # Fallback: whole image
        if not detections:
            detections.append({
                "box": [0, 0, w_img, h_img],
                "confidence": 1.0,
                "fallback": True
            })

        # 2. For each detection, compare ALL models
        final_results = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = pil_img.crop((x1, y1, x2, y2))
            box_norm = [x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img]

            # Compare all models on this crop
            model_results = {}

            for model_name in loaded_models.keys():
                if model_name.startswith("_"):
                    continue

                try:
                    probs = predict_with_model(model_name, crop)
                    top3_indices = np.argsort(probs)[::-1][:3].tolist()
                    predictions = [
                        {"class": class_names[i], "confidence": float(
                            probs[i]) * 100}
                        for i in top3_indices
                    ]
                    model_results[model_name] = predictions
                except Exception as e:
                    model_results[model_name] = {"error": str(e)}

            final_results.append({
                "box": [x1, y1, x2, y2],
                "box_norm": box_norm,
                "comparison": model_results
            })

        return jsonify({"detections": final_results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint using best model (MLP).

    Uses hybrid approach:
    - YOLO detection if confident (conf >= 0.5, area >= 5%)
    - Otherwise: center crop pyramid for robust classification
    """
    if "MLP" not in loaded_models:
        return jsonify({"error": "MLP model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file field 'file'"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        image_bytes = f.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w_img, h_img = pil_img.size

        # =================================================================
        # STEP 1: YOLO detection (ENABLED for better crop quality)
        # =================================================================
        use_yolo = True
        yolo_crop = None
        yolo_box = None

        if use_yolo and yolo_model:
            # Run YOLO with lower threshold to catch more objects
            results = yolo_model(pil_img, verbose=False,
                                 conf=0.15, imgsz=640, agnostic_nms=True)

            # https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8
            # Filter for relevant classes (bottles, cups, bowls, etc.)
            # COCO classes: 39=bottle, 40=wine glass, 41=cup, 42=fork, 43=knife, 44=spoon, 45=bowl,
            # 46-55=food (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake) -> BIO
            # 64=mouse, 65=remote, 66=keyboard, 67=cell phone -> E-WASTE
            # 73=book (paper), 75=vase, 76=scissors, 79=toothbrush
            RELEVANT_CLASSES = [
                39, 40, 41, 42, 43, 44, 45,  # Kitchen/Dining
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # Food (Bio)
                64, 65, 66, 67,  # Electronics
                73, 75, 76, 79   # Misc (Book, Vase, Scissors, Toothbrush)
            ]

            best_det = None
            max_area = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    # Calculate area
                    w_box = xyxy[2] - xyxy[0]
                    h_box = xyxy[3] - xyxy[1]
                    area = w_box * h_box

                    # Prefer relevant classes, otherwise take largest object
                    if cls_id in RELEVANT_CLASSES or area > max_area:
                        max_area = area
                        best_det = (xyxy, conf, cls_id)

            if best_det:
                xyxy, conf, cls_id = best_det
                padding = 20
                x1 = max(0, int(xyxy[0] - padding))
                y1 = max(0, int(xyxy[1] - padding))
                x2 = min(w_img, int(xyxy[2] + padding))
                y2 = min(h_img, int(xyxy[3] + padding))

                yolo_crop = pil_img.crop((x1, y1, x2, y2))
                yolo_box = [x1, y1, x2, y2]

        # Build crop candidates
        crops = []

        if yolo_crop is not None:
            crops.append(("yolo", yolo_crop, yolo_box))
            crops.append(("full", pil_img, [0, 0, w_img, h_img]))
        else:
            # Pyramid strategy
            crops.append(("full", pil_img, [0, 0, w_img, h_img]))
            w, h = pil_img.size
            for scale in [0.9, 0.75, 0.6]:
                cw, ch = int(w * scale), int(h * scale)
                x1 = (w - cw) // 2
                y1 = (h - ch) // 2
                padding = int(min(cw, ch) * 0.1)
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(w, x1 + cw + padding)
                y2_pad = min(h, y1 + ch + padding)
                crop = pil_img.crop((x1_pad, y1_pad, x2_pad, y2_pad))
                crops.append(
                    (f"center_{int(scale*100)}", crop, [x1_pad, y1_pad, x2_pad, y2_pad]))

        # Select best crop
        best_result = None
        best_conf = -1

        for tag, candidate_crop, candidate_box in crops:
            try:
                probs = predict_with_model("MLP", candidate_crop)
                top_prob = float(np.max(probs))

                if tag == "yolo":
                    top_prob *= 1.1

                if top_prob > best_conf:
                    best_conf = top_prob
                    best_result = (tag, candidate_crop, candidate_box)
            except Exception:
                continue

        if best_result is None:
            # Fallback to full image
            method = "full_image"
            crop = pil_img
            box = [0, 0, w_img, h_img]
        else:
            method, crop, box = best_result

        # Final classification
        try:
            avg_probs, agreement, model_preds = predict_ensemble(crop)
            top_idx = int(np.argmax(avg_probs))
            top_class = class_names[top_idx]
            top_prob = float(avg_probs[top_idx])
        except Exception:
            avg_probs = predict_with_model("MLP", crop)
            top_idx = int(np.argmax(avg_probs))
            top_class = class_names[top_idx]
            top_prob = float(avg_probs[top_idx])
            agreement = True
            model_preds = {}

        # Alternatives
        alt_indices = np.argsort(avg_probs)[::-1][1:4].tolist()
        top3_payload = [
            {"class": class_names[i], "prob": float(avg_probs[i])}
            for i in alt_indices
        ]

        x1, y1, x2, y2 = box
        box_norm = [x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img]

        # Bin mapping
        bin_type = CLASS_TO_BIN.get(top_class, 'GENERAL')
        bin_color = BIN_COLORS.get(bin_type, '#6B7280')

        # Confidence warning
        warning = None
        if not agreement:
            warning = "Models disagree on classification"
        elif top_prob < 0.7:
            warning = "Low confidence prediction"

        # Debug crop
        debug_buf = io.BytesIO()
        crop.save(debug_buf, format="JPEG", quality=70)
        crop_b64 = base64.b64encode(debug_buf.getvalue()).decode("utf-8")

        result = {
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "box_norm": box_norm,
            "class": top_class,
            "confidence": round(100.0 * top_prob, 2),
            "bin": bin_type,
            "binColor": bin_color,
            "top3": top3_payload,
            "method": method,
            "debug_crop_b64": crop_b64,
            "ensemble_agreement": agreement,
            "model_predictions": model_preds
        }

        if warning:
            result["warning"] = warning

        return jsonify({"detections": [result]})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
