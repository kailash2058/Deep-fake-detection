# Deep Fake Detection Model

## Overview
This repository contains code for training and evaluating a deep learning model for detecting deep fake videos. The model is based on the Mesonet-4 architecture and is trained using a dataset of real and deep fake videos.

## Introduction
Deep fake videos have become a significant concern due to their potential to spread misinformation and manipulate public opinion. Detecting deep fake videos is crucial for maintaining trust and integrity in digital content. This project aims to develop a deep learning model capable of accurately distinguishing between real and fake videos.

## Model Architecture
The deep fake detection model is based on the Mesonet-4 architecture, which consists of convolutional and pooling layers followed by fully connected layers. The architecture is designed to extract features from video frames and classify them as real or fake. Here's a summary of the model architecture:

- Input Layer: Accepts input images with dimensions (256, 256, 3).
- Convolutional Layers: Multiple convolutional layers with batch normalization and max-pooling operations.
- Fully Connected Layers: Dense layers with dropout and LeakyReLU activation.
- Output Layer: Single neuron with sigmoid activation for binary classification.

The model is trained using the Adam optimizer and binary cross-entropy loss function.

## Dataset
The deep fake detection model is trained and evaluated using a dataset of real and deep fake videos. The dataset is divided into training and validation sets, with images resized to (256, 256) pixels. Data augmentation techniques such as rotation, shifting, and flipping are applied to the training set to improve model generalization.

## Training and Evaluation
The model is trained for 25 epochs using the training set, and its performance is evaluated on the validation set. Model training progress is monitored using accuracy metrics, and the trained model is saved for future use. The model's performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Results
After training and evaluation, the model achieves an accuracy of approximately 84.67% on the validation set. Additionally, the precision, recall, and F1-score metrics indicate the model's effectiveness in detecting deep fake videos.

## Visualization
The repository includes visualizations of the model's performance, such as accuracy curves across epochs and receiver operating characteristic (ROC) curves. These visualizations provide insights into the model's training progress and its ability to discriminate between real and fake videos.

## Usage
To use the deep fake detection model, follow these steps:
1. Clone the repository to your local machine.
2. Prepare your dataset of real and deep fake videos.
3. Train the model using the provided code, adjusting parameters as needed.
4. Evaluate the trained model on a separate validation set to assess its performance.
5. Use the trained model to detect deep fake videos in new datasets or applications.

## Dependencies
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
