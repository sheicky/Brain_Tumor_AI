# Brain Tumor MRI Classification

This project aims to classify brain tumors using MRI scans with deep learning models. It leverages two different approaches: transfer learning with the Xception model and a custom-built CNN.

## Dataset

The dataset used for this project is the Brain Tumor MRI Dataset from Kaggle:

* **Dataset Link:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset consists of MRI scans classified into four categories:

* Glioma
* Meningioma
* No Tumor
* Pituitary

## Models

Two models are employed for classification:

1. **Transfer Learning with Xception:** A pre-trained Xception model is fine-tuned on the brain tumor dataset for classification. This approach utilizes the knowledge gained by the model on a large dataset (ImageNet) and adapts it for brain tumor classification.

2. **Custom CNN:** A custom-built Convolutional Neural Network (CNN) is designed and trained specifically for brain tumor classification. This model architecture is tailored to the characteristics of the MRI images and aims to achieve optimal performance.


## Project Structure

The project's code structure can be visualized as follows:

              +-----------------+
                                |   Data Loading  |
                                +--------+--------+
                                         |
                                         v
                    +-----------------+-----------------+
                    | Xception Model | Custom CNN Model |
                    +--------+--------+--------+--------+
                             |                 |
                             v                 v
               +------------+------------+------------+
               |   Training | Validation |   Testing  |
               +------------+------------+------------+
                             |
                             v
                   +-----------------+
                   | Streamlit App  |
                   +--------+--------+
                             |
                             v
             +------------+------------+
             | Prediction | Explanation |
             +------------+------------+


    This graph illustrates the flow of data and the different components of the project's code. Data loading is performed first, followed by the training, validation, and testing of the two models (Xception and Custom CNN). The trained models are then integrated into the Streamlit app, allowing users to upload images for prediction and visualization.

Here's a brief description of the key files:

* **xception_model.weights.h5:** Contains the saved weights of the fine-tuned Xception model.
* **cnn_model.weights.h5:** Contains the saved weights of the custom CNN model.
* **app.py:** A Streamlit app that allows users to upload MRI images for prediction and visualization.
* **saliency_maps:** A directory to store the generated saliency maps.

## Usage

1. **Set up Environment:**
2. **Run the Streamlit App:**
3. **Upload Image:** Browse and select an MRI image for classification.

4. **View Results:**  The app displays the predicted class along with class probabilities. 
   * Saliency Map: The highlighted areas in the saliency map indicate regions the model focused on for prediction.
   * Explanation: The AI-generated explanation offers potential reasons for the model's prediction.

## Streamlit App

The Streamlit app provides an interactive interface for users to:

* Upload MRI images for prediction.
* Select the desired model (Xception or Custom CNN).
* View predicted tumor type with confidence scores.
* Visualize saliency maps highlighting areas of focus for prediction.
* Read an AI-generated explanation of the model's prediction.

## Results

The project achieves high accuracy on the brain tumor dataset. Detailed evaluation metrics, including accuracy, precision, recall, and the confusion matrix, are available in the project report.

## Future Work

* Exploring other deep learning architectures for improved performance.
* Experimenting with hyperparameter optimization techniques.
* Integrating model interpretability tools for better understanding predictions.
* Deploying the app to the cloud for wider accessibility.
