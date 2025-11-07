# 🛣️ Road Damage Detection (RDD2022) – Faster R-CNN ResNet50

## Overview
This project detects and classifies road surface damages using **Faster R-CNN (ResNet-50 FPN)**, trained on the **RDD2022 dataset**.  
It supports full training, evaluation (mAP), and real-time inference via a Flask web app.

---

## Input / Output
**Input:**  
- Images and bounding box annotations from RDD2022.
**Output:**  
- Best checkpoint: `checkpoints/best_model_checkpoint_epoch_30.pth`  
- Flask app for visualizing predictions (upload or via URL).

---

## Why ResNet-50?
After obtaining the dataset, we found a related repo using the same dataset:  
🔗 https://github.com/andreas-roennestad/Resnet50-FPN-detection-RDD2022  
Their results:  
> “Fine-tuning was performed on the Norway subset with 1, 2, 3 trainable backbone layers, and results were 0.18, 0.21, and 0.31 respectively.”

Hence, we tested **the same backbone** on the **same dataset** to compare performance.  
Although our final mAP ≈ **0.20** (lower than their of 0.31), we tackled a **harder task** (9 classes including background, vs their 4).  
Moreover, we achieved **mAP 0.20 by around epoch 15**, while their 0.31 was reached at epoch 40 — and our mAP was still improving at epoch 31 with no saturation yet.

---

## Results
| Metric | Meaning | Result |
|:-------|:--------|:------:|
| **mAP** | COCO standard | 0.2039 |
| **mAP@50** | IoU > 0.5 | 0.4432 |
| **Recall** | mar_100 | 0.3846 |

---

## Dataset
Official dataset:  
🔗 https://datasetninja.com/road-damage-detector

---

## Workflow
 Load Dataset → Transform → Faster R-CNN (ResNet-50)
→ Train (main.py) → Evaluate (mAP) → Save checkpoint → Flask Demo


---

## How to Run
pip install -r requirements.txt
git clone https://github.com/pytorch/vision.git

# Train or Resume
python main.py
python main.py --resume --checkpoint checkpoints/best_model_checkpoint_epoch_31.pth

# Run Web App
python app.py

