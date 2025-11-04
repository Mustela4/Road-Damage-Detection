import os
import io
import base64
import random
import time
from typing import Dict

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor
import requests


APP_TITLE = "Road Damage Detection"

# 10 ảnh mẫu để test
SAMPLE_DIR = os.getenv("SAMPLE_DIR", r"D:\DL\Dataset\test_inference\images")

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    os.path.join(os.getcwd(), "checkpoints", "best_model_checkpoint_epoch_30.pth")
)


NUM_CLASSES = int(os.getenv("NUM_CLASSES", "9"))

CONF_THRESH = float(os.getenv("CONF_THRESH", "0.55"))
IOU_THRESH  = float(os.getenv("IOU_THRESH",  "0.60"))
TOPK        = int(os.getenv("TOPK", "50"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device):
    """
    Build Faster R-CNN model bằng hàm build_model trong finetune.py
    và nạp checkpoint đã train.
    """
    from finetune import build_model  # dùng đúng model của repo bạn
    model = build_model(NUM_CLASSES)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # hỗ trợ cả dạng state_dict thuần hoặc dict có "model_state_dict"
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    model.to(device).eval()
    return model


def fetch_image_from_url(url: str) -> Image.Image:
    """
    Tải ảnh từ URL (http/https) -> PIL Image RGB.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=12)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """
    Convert PIL Image -> data URL base64 để trả ra frontend.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def postprocess_nms(boxes, scores, labels):
    """
    Lọc box theo CONF_THRESH + NMS, giới hạn TOPK.
    """
    keep = scores >= CONF_THRESH
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if boxes.numel() == 0:
        return boxes, scores, labels

    keep_n = nms(boxes, scores, IOU_THRESH)
    boxes, scores, labels = boxes[keep_n], scores[keep_n], labels[keep_n]

    if len(scores) > TOPK:
        top_idx = torch.topk(scores, TOPK).indices
        boxes, scores, labels = boxes[top_idx], scores[top_idx], labels[top_idx]
    return boxes, scores, labels


def draw_boxes(pil_img: Image.Image, boxes, scores, labels) -> Image.Image:
    """
    Vẽ bbox + score lên ảnh, trả về ảnh mới.
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for b, s, c in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        caption = f"{int(c)}:{float(s):.2f}"
        # vẽ nền caption
        try:
            t_left, t_top, t_right, t_bottom = draw.textbbox((x1, max(0, y1 - 18)), caption, font=font)
            draw.rectangle([t_left - 4, t_top - 2, t_right + 4, t_bottom + 2], fill=(255, 0, 0))
        except Exception:
            pass
        draw.text((x1, max(0, y1 - 18)), caption, fill=(255, 255, 255), font=font)
    return img


def run_inference(pil_img: Image.Image, model) -> Dict:
    """
    Chạy inference 1 ảnh PIL -> trả base64 + bbox info.
    """
    t0 = time.time()
    with torch.no_grad():
        tensor = to_tensor(pil_img).to(DEVICE)  # (C,H,W), 0..1
        outputs = model([tensor])[0]
    infer_ms = int((time.time() - t0) * 1000)

    boxes = outputs["boxes"].detach().cpu()
    scores = outputs["scores"].detach().cpu()
    labels = outputs["labels"].detach().cpu()

    boxes, scores, labels = postprocess_nms(boxes, scores, labels)
    vis_img = draw_boxes(pil_img, boxes, scores, labels)

    dets = []
    for b, s, c in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        dets.append({"box": [x1, y1, x2, y2], "score": float(s), "label": int(c)})

    return {
        "image_b64": pil_to_base64(vis_img),
        "detections": dets,
        "meta": {
            "width": pil_img.width,
            "height": pil_img.height,
            "inference_ms": infer_ms,
            "num": len(dets),
        },
    }


# =========================
# Flask app
# =========================

app = Flask(__name__, template_folder="templates")

try:
    MODEL = load_model(CHECKPOINT_PATH, DEVICE)
    MODEL_READY = True
    MODEL_MSG = f"Model loaded from: {CHECKPOINT_PATH}"
except Exception as e:
    MODEL = None
    MODEL_READY = False
    MODEL_MSG = f"Failed to load model: {e}"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", app_title=APP_TITLE, model_ready=MODEL_READY, model_msg=MODEL_MSG)


@app.route("/detect", methods=["POST"])
def detect():
    """
    Nhận JSON: {"imageUrl": "..."} -> trả JSON chứa ảnh + bbox.
    """
    if not MODEL_READY:
        return jsonify({"ok": False, "error": MODEL_MSG}), 500

    data = request.get_json(silent=True) or {}
    image_url = data.get("imageUrl")
    if not image_url:
        return jsonify({"ok": False, "error": "Missing 'imageUrl'"}), 400

    try:
        pil_img = fetch_image_from_url(image_url)
        result = run_inference(pil_img, MODEL)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/test-samples", methods=["POST"])
def test_samples():
    """
    Chạy thử 10 ảnh ngẫu nhiên trong SAMPLE_DIR.
    """
    if not MODEL_READY:
        return jsonify({"ok": False, "error": MODEL_MSG}), 500

    if not os.path.isdir(SAMPLE_DIR):
        return jsonify({"ok": False, "error": f"SAMPLE_DIR not found: {SAMPLE_DIR}"}), 400

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_imgs = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(exts)]
    if not all_imgs:
        return jsonify({"ok": False, "error": f"No images in: {SAMPLE_DIR}"}), 400

    random.shuffle(all_imgs)
    picks = all_imgs[:10]

    items = []
    for p in picks:
        try:
            pil_img = Image.open(p).convert("RGB")
            result = run_inference(pil_img, MODEL)
            items.append({"filename": os.path.basename(p), **result})
        except Exception as e:
            items.append({"filename": os.path.basename(p), "error": str(e)})

    return jsonify({"ok": True, "count": len(items), "items": items})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
