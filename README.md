# 🌊 Flood Area Segmentation using U-Net + ResNet34

A deep learning-based image segmentation project for identifying flood-affected regions in satellite or aerial imagery using a U-Net model with a ResNet34 encoder.

---

## 🧠 Model Overview

This project implements a semantic segmentation model to detect flooded areas from RGB satellite/drone images. The architecture is based on U-Net, enhanced with a ResNet34 encoder pretrained on ImageNet for improved feature extraction.

- **Model**: U-Net  
- **Encoder**: ResNet34 (pretrained on ImageNet)  
- **Framework**: PyTorch  
- **Library**: `segmentation_models_pytorch`  
- **Loss Function**: Focal Loss  
- **Metrics**: Dice Score, IoU, Accuracy  
- **Deployment**: Streamlit Web App

---

## 📈 Performance Metrics

| Metric       | Score   |
|--------------|---------|
| Dice Score   | 0.81    |
| IoU          | 0.72    |
| Accuracy     | 87%     |

---

## 💻 Features of the Streamlit Dashboard

- 📤 Upload satellite/drone images (JPG/PNG)
- 🧠 Predict flood segmentation masks in real time
- 📊 Display estimated flood coverage percentage
- ⚠️ Alert users based on severity thresholds:
  - **> 40%** → 🚨 Evacuation recommended  
  - **30–40%** → ⚠️ Partial flooding  
  - **< 30%** → ✅ Safe  
- 📥 Download predicted mask as PNG
- ✅ Upload ground truth mask (optional) to evaluate:
  - Dice Score
  - IoU
  - Accuracy

---
