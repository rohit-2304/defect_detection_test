# YOLO11 Nano Defect Detection for Metal Spare Parts

An optimized object detection pipeline using **YOLO11 Nano (`yolo11n.pt`)** to identify surface defects in metal spare parts (Dents, Scratches, Chip-offs, surface damage, and unclear surfaces).

This project features:
1. **Mathematical Label Cleaning**: Automatically converts Roboflow's mixed polygon segmentations to standard bounding boxes to ensure standard detection training and anchor alignment.
2. **OpenCV Bounding-Box Aware Augmentation**: Dynamically copies underrepresented/missing classes from other splits and applies 10 custom physical transforms (flips, contrast, blur, and lighting adjustments) to balance classes and expand training data from 12 to 75 images.
3. **GPU Accelerated Pipeline**: Auto-detects and leverages GPU (NVIDIA GTX 1650) for fast convergence.
4. **End-to-End Evaluation**: Evaluates on unseen test splits, computes precision, recall, and mAP, and saves annotated visual results showing detected defects.

---

## 📊 Performance Breakthrough

By cleaning annotation shapes and balancing class representations, the model achieved the following results on the unseen **test split**:

| Metric | Original Run (Mixed/Imbalanced) | Optimized Run (Cleaned + Augmented) |
| :--- | :--- | :--- |
| **Precision** | 82.84% | **92.30%** |
| **Recall** | 33.33% | **100.00%** |
| **mAP50** | 18.12% | **99.50%** |
| **mAP50-95** | 16.89% | **79.60%** |

---

## 📂 Project Directory Structure

```text
├── data/                  # Dataset directories (train, valid, test)
├── output/                # Training logs, weights, metrics, and annotated images
│   └── yolo11n/
│       ├── train_results/     # Best/last model weights (.pt), curves, and validation plots
│       ├── test_annotated/    # Annotated test images with visual bounding boxes
│       └── test_metrics.json  # JSON report of precision, recall, mAP50, and class mappings
├── data_local.yaml        # Dataset configuration file with absolute paths
├── augment.py             # Polygon-to-bounding-box cleaner and 10x OpenCV augmentations
├── train.py               # Retraining pipeline with tuned online augmentations (100 epochs)
├── test.py                # Testing pipeline (mAP evaluation and visual annotations)
└── README.md              # Project documentation
```

---

## 🚀 How to Run

### 1. Requirements
Ensure you have the required libraries installed:
```bash
pip install ultralytics opencv-python numpy
```

### 2. Run Data Augmentation and Retrain
To mathematically clean all annotations and train the model for 100 epochs, run:
```bash
python train.py
```
*Note: `train.py` automatically executes `augment.py` under the hood before initializing training.*

### 3. Run Inference and Evaluation
To evaluate the model on the test split and generate annotated images in `output/yolo11n/test_annotated/`:
```bash
python test.py
```
# defect_detection_test
