# Brain Tumor MRI Classification with Deep Learning

This project implements an advanced deep learning system for classifying brain tumors from MRI scans. It leverages two complementary approaches - transfer learning with the Xception architecture and a custom CNN model - to achieve high accuracy in tumor classification.

## Overview

The system can classify brain MRI scans into four categories:
- Glioma
- Meningioma  
- No Tumor
- Pituitary

Key features:
- Transfer learning using pre-trained Xception model
- Custom CNN architecture optimized for MRI analysis
- Interactive Streamlit web interface
- Saliency map visualization for model interpretability
- AI-generated explanations of model predictions

## Dataset

The project uses the Brain Tumor MRI Dataset from Kaggle:
- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Size**: Over 7,000 MRI images
- **Classes**: 4 categories (glioma, meningioma, no tumor, pituitary)
- **Format**: High quality .jpg images of brain MRI scans

## Model Architecture

### Transfer Learning Model (Xception)
- Pre-trained Xception base model
- Additional layers:
  - Global Max Pooling
  - Dropout (0.3)
  - Dense layer (128 units, ReLU)
  - Dropout (0.25)
  - Output layer (4 units, Softmax)

### Custom CNN Model
- Optimized convolutional architecture
- Multiple conv-pool blocks
- Batch normalization
- Dropout regularization
- Dense layers for classification

## Performance

Both models achieve strong classification performance:

**Xception Model**:
- Training Accuracy: 99.15%
- Validation Accuracy: 94.58%
- Test Accuracy: 96.89%

**Custom CNN**:
- Training Accuracy: 97.05%
- Validation Accuracy: 95.03% 
- Test Accuracy: 95.65%

## Web Interface

The project includes an interactive Streamlit web application that allows users to:

1. Upload brain MRI scans
2. Choose between Xception and Custom CNN models
3. Get real-time predictions with confidence scores
4. View saliency maps highlighting regions of interest
5. Read AI-generated explanations of model predictions

### Interface Features
- Clean, intuitive design
- Real-time processing
- Visualization of model attention via saliency maps
- Confidence scores for all classes
- Natural language explanations

## Project Structure 🗂️

```
Brain_Tumor_AI/
├── models/
│   ├── xception_model.weights.h5
│   └── cnn_model.weights.h5
├── notebooks/
│   └── brain_tumor.ipynb
├── app/
│   ├── app.py
│   └── requirements.txt
├── saliency_maps/
└── README.md
```

## Installation & Usage 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Brain_Tumor_AI.git
cd Brain_Tumor_AI
```



2. Run the Streamlit app:
```bash
streamlit run app/app.py
```

4. Access the web interface at `http://localhost:8501`

## Model Training 🧠

To train the models:

1. Download the dataset from Kaggle
2. Open and run `brain_tumor.ipynb` in Google Colab or Jupyter
3. Model weights will be saved to the `models/` directory

## Technologies Used 💻

- TensorFlow/Keras
-Scikit-learn
- Streamlit
- OpenCV
- NumPy
- Pandas
- Google Gemini AI (for explanations)
- Plotly (for visualizations)

## Results & Metrics 📊

The project achieves exceptional performance in tumor classification:

- High accuracy across all tumor types
- Robust performance on unseen data
- Real-time inference capabilities
- Interpretable predictions with saliency maps

Detailed metrics including precision, recall, and confusion matrices are available in the project notebook.

## Future Improvements 🔮

- [ ] Implement ensemble methods for improved accuracy
- [ ] Add support for 3D MRI sequences
- [ ] Enhance visualization capabilities
- [ ] Deploy model to cloud platform
- [ ] Add batch processing functionality
- [ ] Improve explanation quality with medical context
- [ ] Integrate additional pre-trained architectures



<div align="center">
Made with ❤️  by Sheick 
</div>
