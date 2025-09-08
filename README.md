# Pneumonia Detection using YOLOv8

This project demonstrates **Pneumonia Detection from chest X-ray images** using a YOLOv8 model. The goal is to classify X-ray images as **Normal** or **Pneumonia** using object detection techniques.  

> **Note:** This project is for learning purposes only and is **not intended for medical diagnosis**.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [YOLOv8 Training](#yolov8-training)  
4. [Inference / Testing](#inference--testing)  
5. [Folder Structure](#folder-structure)  
6. [Dependencies](#dependencies)  
7. [How to Run](#how-to-run)  
8. [Results](#results)  
9. [Skills Learned](#skills-learned)  
10. [References](#references)

---

## Project Overview

- Trained a **YOLOv8 model** to detect Pneumonia in chest X-ray images.  
- Classifies images as either **Normal** or **Pneumonia**.  
- Demonstrates the full workflow from **dataset preparation → model training → inference**.

---

## Dataset

- Model trained on a **publicly available Pneumonia dataset**, such as the **Kaggle Chest X-ray dataset**: [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- Dataset contains X-ray images divided into `train`, `val`, and `test` folders, with subfolders for `NORMAL` and `PNEUMONIA`.

**Tip:** For GitHub, include only a few sample images in `sample_images/` for demonstration.

---

## YOLOv8 Training

1. Set up **Google Colab** with GPU runtime.  
2. Install ultralytics YOLOv8:

```bash
!pip install ultralytics
```

3. Prepare dataset folder structure:

```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
```

4. Train the YOLOv8 model:

```python
from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  

# Train the model
model.train(
    data="path/to/data.yaml",  # dataset YAML file
    epochs=50,
    imgsz=224,
    batch=16,
    project="runs/train",
    name="pneumonia_detection"
)
```

- `best.pt` will be saved in:

```
runs/train/pneumonia_detection/weights/best.pt
```

---

## Inference / Testing

After training, run inference on new X-ray images:

```bash
python inference.py
```

- The script can process **single images** or **all images in a folder**.  
- Output images with predictions are saved in `runs/classify/`.  

---

## Folder Structure for GitHub

```
Pneumonia-Detection-YOLOv8/
├── best.pt
├── sample_images/
├── inference.py
├── requirements.txt
└── README.md
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

- `ultralytics` → YOLOv8 library  
- `opencv-python` → Image visualization  

---

## How to Run

1. Clone the repository:

```bash
git clone <your-repo-link>
cd Pneumonia-Detection-YOLOv8
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run inference:

```bash
python inference.py
```

- Predictions will be saved in `runs/classify/`.  

---

## Results

| Image | Prediction |
|-------|------------|
| gr1.jpeg | PNEUMONIA 0.89, NORMAL 0.11 |
| img.jpeg | PNEUMONIA 0.73, NORMAL 0.27 |
<img width="1041" height="489" alt="result 1" src="https://github.com/user-attachments/assets/1159ee4a-7ef8-4852-ad91-2aa9e389a9c3" />

---

## Skills Learned

- Dataset preparation & annotation  
- YOLOv8 training workflow  
- Running inference on new images  
- Handling Colab & Google Drive integration  
- Visualization of results  

---

## References

1. [YOLOv8 Documentation](https://docs.ultralytics.com/)  
2. [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
3. [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)  

> 💡 Only `best.pt` and sample images are included for GitHub. Full dataset is not uploaded.

