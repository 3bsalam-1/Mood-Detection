# 😊😢 Mood Detection CNN

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-D00000.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-80%25-brightgreen.svg)](#model-performance)

A professional **deep learning project** that detects facial mood expressions (Happy vs Sad) from images using a **Convolutional Neural Network (CNN)**. Features real-time prediction via camera or file upload with an intuitive GUI.

## ✨ Features

- 🎯 **High-Accuracy Classification** - 80%+ accuracy on binary mood detection (Happy/Sad)
- 📸 **Real-Time Camera Detection** - Live video feed with instant mood prediction
- 📁 **File Upload Support** - Predict mood from image files (JPG, PNG, BMP)
- 🖥️ **Modern GUI** - Clean dark/light theme interface with CustomTkinter
- 🚀 **GPU Optimized** - Automatic GPU detection and memory management
- 📊 **Data Augmentation** - Prevents overfitting on small datasets
- 🔧 **Easy Training** - Simple notebook-based training pipeline
- 📈 **TensorBoard Logging** - Monitor training metrics in real-time

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **TensorFlow** | 2.16+ | Deep learning framework |
| **Keras** | 3.0+ | Neural network API |
| **OpenCV** | 4.8+ | Image processing & camera capture |
| **CustomTkinter** | 5.0+ | Modern GUI framework |
| **Matplotlib** | 3.7+ | Training visualization |
| **Pillow** | 10.0+ | Image display |

## 📁 Project Structure

```
Image-Classification/
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore patterns
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Packaging metadata
├── Dockerfile                       # Docker image definition
│
# (No top-level `scripts/` directory — run GUI using the module entrypoint)
# Run GUI locally with: `python -m src.gui.app`
├── src/                             # Core package
│   ├── __init__.py
│   ├── cli.py                       # CLI: train / predict
│   ├── train.py                     # Training pipeline (scriptable)
│   ├── inference.py                 # Inference utilities (MoodPredictor)
│   └── gui/                         # GUI package
│       └── app.py                   # GUI application implementation
│
├── notebooks/                       # Notebooks for experiments
│   └── train.ipynb                  # Training notebook
│
├── models/                          # Trained models
│   └── mood.h5                      # Trained binary classifier
│
├── mood/                            # Dataset
│   ├── happy/                        # happy images
│   └── sad/                          # sad images
│
└── m_logs/                          # TensorBoard logs
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager
- (Optional) NVIDIA GPU

### Installation

```bash
# 1. Clone repository
git clone https://github.com/3bsalam-1/Image-Classification.git
cd Image-Classification

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Enable GPU support
pip install tensorflow[and-cuda]
```

## 📖 Usage

### Training the Model

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter notebook notebooks/train.ipynb
```
Execute all cells for interactive training with visualizations and experiments.

**Option B: CLI / Python Script**
```bash
# Using the CLI (recommended):
python -m src.cli train --epochs 50

# Or run the training script directly:
python src/train.py
```

**Expected Output:**
```
Dataset loaded: 9 batches
Starting training...
Epoch 1/50 - loss: 0.689, accuracy: 0.562, val_loss: 0.683, val_accuracy: 0.583
...
=== Test Set Results ===
Precision: 0.85
Recall: 0.78
Accuracy: 0.81
Loss: 0.4921
```

### Real-Time Prediction GUI

```bash
# Run GUI locally (module entrypoint)
python -m src.gui.app
```

**Note:** The repository does not include `scripts/gui_app.py` (the `Dockerfile` references it). If you plan to use Docker, either add a `scripts/gui_app.py` that imports `src.gui.app:main` or change the `Dockerfile` `CMD` to `python -m src.gui.app`.

**Features:**
- 📸 Live camera detection
- 📁 Image file selection
- 😊 Happy / 😢 Sad classification
- Dark/Light theme support

### Python API

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('models/mood.h5')

# Predict from image
image = cv2.imread('image.jpg')
image_resized = cv2.resize(image, (256, 256)) / 255.0
prediction = model.predict(np.expand_dims(image_resized, 0))[0][0]

mood = 'Happy' if prediction <= 0.5 else 'Sad'
confidence = (1 - prediction) if prediction <= 0.5 else prediction
print(f"Mood: {mood}, Confidence: {confidence:.2%}")
```

## 🤖 Model Architecture

### Network Design

```
Input: 256×256 RGB Image
       ↓
[Data Augmentation]
RandomFlip(0.5) + Rotation(0.15) + Zoom(0.15)
       ↓
[Conv Block 1] 32 filters → MaxPool → Dropout(0.25)
[Conv Block 2] 64 filters → MaxPool → Dropout(0.25)
[Conv Block 3] 128 filters → MaxPool → Dropout(0.25)
       ↓
[Dense 1] 512 units → Dropout(0.4)
[Dense 2] 256 units → Dropout(0.4)
[Dense 3] 128 units → Dropout(0.3)
       ↓
[Output] 1 unit → Sigmoid
       ↓
Probability: [0, 1]
```

### Key Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Loss | BinaryCrossentropy | Binary classification |
| Activation (Output) | Sigmoid | Probability in [0,1] |
| Optimizer | Adam (lr=0.0005) | Stable convergence |
| Regularization | L2(0.0001) | Prevents overfitting |
| Data Augmentation | Yes | Small dataset (440 images) |
| Class Weights | {0: 1.1, 1: 0.9} | Balance classes |
| Early Stopping | Patience=7 | Avoid overfitting |
| Max Epochs | 50 | With early stopping |

## 📊 Model Performance

### Test Set Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.0% |
| **Precision** | 0.85 |
| **Recall** | 0.78 |
| **Loss (BCE)** | 0.4921 |

### Dataset Info

| Aspect | Value |
|--------|-------|
| Total Images | 440 |
| Happy | 214 (49%) |
| Sad | 226 (51%) |
| Balance Ratio | 1.06:1 ✅ |
| Image Size | 256×256 |
| Corruption | 0% ✅ |
| Split | 70/20/10 |


## 🐛 Troubleshooting

### No GPU Detected
```bash
pip install tensorflow[and-cuda]
# Or with conda:
conda install tensorflow-gpu
```

### Module Not Found
```bash
pip install -r requirements.txt
```

### Model File Missing
```bash
# Train first using the CLI or script
python -m src.cli train
# or
python src/train.py
```

### GUI Window Won't Open
```bash
pip install customtkinter --upgrade
```

### Camera Permission Issues
- Grant camera access in OS settings
- Try different USB port for external camera

## 📚 Adding New Images

1. Place happy images in `mood/happy/`
2. Place sad images in `mood/sad/`
3. Retrain: `python -m src.cli train`

Supported formats: JPG, PNG, BMP

## 📝 GPU Performance

| Component | CPU | GPU |
|-----------|-----|-----|
| Training Time/Epoch | 1-5 min | 5-15 sec |
| Speedup | 1x | **10-50x** ✅ |

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Ahmed Abdulsalam**
- GitHub: [@3bsalam-1](https://github.com/3bsalam-1)
- Email: ahmed.abdulsalam@example.com

## 🙏 Acknowledgments

- [TensorFlow](https://tensorflow.org/) - Deep learning framework
- [Keras](https://keras.io/) - Neural networks API
- [OpenCV](https://opencv.org/) - Computer vision
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern GUI
- [Matplotlib](https://matplotlib.org/) - Visualization
- [Pandas](https://pandas.pydata.org/) - Data processing

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by Ahmed Abdulsalam

</div>
