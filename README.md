<div align="center">

# 🧠 Brain Tumor MRI Classification with Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sheicky/Brain_Tumor_AI/blob/main/brain_tumor.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

An advanced deep learning system for automated brain tumor classification from MRI scans, achieving 96.89% accuracy.

[Key Features](#key-features) • [Demo](#live-demo) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Results](#results)

</div>

---

## 🌟 Key Features

- **Dual Model Architecture**
  - Transfer Learning with Xception (96.89% accuracy)
  - Custom CNN (95.65% accuracy)
- **Interactive Web Interface**
  - Real-time predictions
  - Saliency map visualization
  - Confidence scores
- **AI-Powered Explanations**
  - Medical context generation
  - Region-specific analysis
- **Production-Ready**
  - RESTful API support
  - Docker containerization
  - Batch processing capability

## 🎯 Overview

The system classifies brain MRI scans into four categories with exceptional accuracy:

| Tumor Type | Description                       | Model Performance |
| ---------- | --------------------------------- | ----------------- |
| Glioma     | Most common malignant brain tumor | 97% accuracy      |
| Meningioma | Usually benign, slow-growing      | 90% accuracy      |
| Pituitary  | Occurs in pituitary gland         | 99% accuracy      |
| No Tumor   | Healthy brain scan                | 100% accuracy     |

## 🔬 Dataset

The project utilizes the Brain Tumor MRI Dataset from Kaggle:

- **Size**: 7,023 MRI images
- **Resolution**: High-quality .jpg format
- **Distribution**: Balanced across classes
- **Augmentation**: Applied for robust training

## 🏗️ Architecture

### Transfer Learning Model (Xception)

```python
model = Sequential([
    Xception(include_top=False, weights='imagenet'),
    GlobalMaxPooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(4, activation='softmax')
])
```

### Custom CNN Architecture

- 6 convolutional layers
- Batch normalization
- Skip connections
- Dropout regularization

## 📊 Performance Metrics

### Xception Model

```
Accuracy: 96.89%
Precision: 0.97
Recall: 0.96
F1-Score: 0.97
```

### Custom CNN

```
Accuracy: 95.65%
Precision: 0.96
Recall: 0.95
F1-Score: 0.96
```

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/sheicky/Brain_Tumor_AI.git
cd Brain_Tumor_AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

## 🚀 Usage
 
**Access via Browser**

- Run the code on google colab
- Open `http://localhost:8501`
- Upload MRI scan
- View predictions and analysis

## 📱 Web Interface Features

- **Upload**: Drag & drop MRI scans
- **Model Selection**: Choose between Xception and Custom CNN
- **Visualization**:
  - Saliency maps
  - Confidence scores
  - Region highlighting
- **Analysis**: AI-generated medical explanations


## 📈 Results & Metrics

Our models achieve exceptional performance:

<div align="center">

| Metric    | Xception | Custom CNN |
| --------- | -------- | ---------- |
| Accuracy  | 96.89%   | 95.65%     |
| Precision | 0.97     | 0.96       |
| Recall    | 0.96     | 0.95       |
| F1-Score  | 0.97     | 0.96       |

</div>


## 📧 Contact

Sheick - [@sheicky](https://github.com/sheicky)

Project Link: [https://github.com/sheicky/Brain_Tumor_AI](https://github.com/sheicky/Brain_Tumor_AI)

---

<div align="center">
Made with ❤️ by Sheick | Advancing Medical Imaging with AI
</div>
