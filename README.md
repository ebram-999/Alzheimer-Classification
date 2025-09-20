# Alzheimer-Classification

This repository contains a Jupyter Notebook (`AD_PD_Control.ipynb`) for classifying Alzheimer's Disease (AD), Parkinson's Disease (PD), and Control (healthy) subjects based on image data. The project utilizes deep learning techniques, specifically Convolutional Neural Networks (CNNs) and Deep Neural Networks (DNNs) with ResNet features.

## Project Overview

The goal of this project is to develop a robust image classification model to distinguish between different neurological conditions. The notebook demonstrates the process from data loading and preprocessing to model training, evaluation, and visualization of results.

## Dataset

The dataset consists of image files categorized into three classes: AD, CONTROL, and PD. The notebook includes code to count the number of images in the training and testing folders for each class.

- **Train folder:**
  - AD: 2561 images
  - CONTROL: 3010 images
  - PD: 906 images

- **Test folder:**
  - AD: 639 images
  - CONTROL: 662 images
  - PD: 61 images

## Technologies and Libraries

- **TensorFlow/Keras**: For building and training deep learning models.
- **Scikit-learn**: For evaluation metrics like classification reports.
- **Matplotlib**: For visualizing training history and other plots.
- **Numpy**: For numerical operations.
- **OS**: For file system operations.

## Model Architecture

The project explores two main model architectures:

1.  **Deep Neural Network (DNN) with ResNet Features**: 
    - ResNet50 is used as a feature extractor. 
    - The extracted features are then fed into a sequential DNN model with Dense layers and Dropout for regularization. 
    - SMOTE (Synthetic Minority Over-sampling Technique) is applied to address class imbalance.

2.  **Convolutional Neural Network (CNN)**: 
    - A custom CNN model is built with Conv2D, MaxPooling2D, GlobalAveragePooling2D, and Dense layers. 
    - Early stopping is implemented to prevent overfitting.

## Training and Evaluation

The notebook details the training process for both DNN and CNN models, including:

- Data loading and preprocessing using `tf.keras.preprocessing.image_dataset_from_directory`.
- Configuration of image size (224x224) and batch size (32).
- Model compilation with `adam` optimizer and `sparse_categorical_crossentropy` loss.
- Training with `EarlyStopping` callback.
- Evaluation using `model.evaluate` and `classification_report`.

## Results

The notebook provides accuracy and classification reports for both models. For example, the CNN model achieved an accuracy of approximately 48.6% on the test set.

## Usage

To run the notebook:

1.  Clone the repository.
2.  Ensure you have the necessary image dataset structured in `train` and `test` directories, with subdirectories for each class (AD, CONTROL, PD).
3.  Install the required Python libraries (TensorFlow, scikit-learn, matplotlib, numpy).
4.  Open and run the `AD_PD_Control.ipynb` notebook in a Jupyter environment.

```python
# Example of data loading
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "path/to/your/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "path/to/your/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
```

## Future Work

- Explore more advanced deep learning architectures.
- Implement data augmentation techniques.
- Fine-tune hyperparameters for improved performance.
- Investigate interpretability methods to understand model predictions.


