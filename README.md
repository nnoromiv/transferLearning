# Transfer Learning: Feature Extraction and Fine-Tuning a Pre-Trained Model

## Overview

This project focuses on using transfer learning with the VGG16 model to classify images from the CIFAR-10 dataset. The implementation includes two approaches:

1. **Feature Extraction:** Using the pre-trained convolutional base of VGG16 and adding a custom classifier.
2. **Fine-Tuning:** Unfreezing some layers of the pre-trained model and training them along with the classifier.

## Requirements

Before running the notebook, ensure the following dependencies are installed:

- Python 3.x
- TensorFlow 2.15
- Keras
- NumPy
- Matplotlib

## Instructions

### Setting Up the Environment

- Run the notebook in **Vs Code** with the **GPU runtime enabled** for faster training.

### Feature Extraction

- Load the **VGG16** model with `include_top=False` to remove its classifier.
- Freeze all convolutional layers to retain pre-trained weights.
- Add a custom **fully connected classifier** with dropout layers to prevent overfitting.
- Train the model on **CIFAR-10** images.
- Adjust dropout layers and neurons to **achieve validation accuracy > 0.65** without overfitting.

### Fine-Tuning

- Load VGG16 again and initially freeze all layers except the classifier.
- Train for a few epochs to adjust the classifier.
- Unfreeze some top convolutional layers and train again with a reduced learning rate.
- Add **EarlyStopping** to stop training when validation accuracy stops improving.
- Fine-tune until **validation accuracy > 0.8** and the difference between training and validation accuracy is <0.1.

## Model Training

- Training is performed in two stages: first **feature extraction**, then **fine-tuning**.
- **Callbacks:**
  - `ModelCheckpoint` saves the best model.
  - `EarlyStopping` prevents unnecessary training.
- The final trained model is saved as **vgg16_model_fineTuning.keras**.

## Output

- Training history is saved in `history_featureExtraction.npy` and `history_fineTuning_2.npy`.
- Accuracy graphs help visualize model performance.

## Results

- The best model should achieve **>0.8 validation accuracy** with no significant overfitting.
- The final model can be used for CIFAR-10 image classification.

## Usage

1. Run the notebook in **Vs Code**.
2. Ensure **GPU runtime** is enabled.
3. Modify the classifier layers if needed to improve performance.
4. Train the model using **feature extraction** and **fine-tuning**.
5. Save the best-performing model.

## References

- **CIFAR-10 Dataset:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **VGG16 Model:** [https://keras.io/api/applications/vgg/](https://keras.io/api/applications/vgg/)

Created by `Dr Ana Matran-Fernandez` (<amatra@essex.ac.uk>)
