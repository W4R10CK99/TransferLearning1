# Image Classification using MobileNet

This project demonstrates how to build and train an image classification model using TensorFlow and Keras. The model is based on MobileNet, a lightweight convolutional neural network architecture suitable for mobile and embedded devices.

## Overview

The goal of this project is to classify images into two categories: cats and non-cats. The model is trained on a dataset containing images of cats and various other objects. Transfer learning is employed by using the pre-trained MobileNet model as the convolutional base and adding custom layers on top for classification.

## Requirements

To run this project, you need the following dependencies:
- Python 3.x
- TensorFlow 2.x
- Keras
- Pillow (PIL)
- NumPy

You can install the required packages using pip:

```bash
pip install tensorflow keras pillow numpy
```

## Usage

1. Clone the repository to your local machine:

  ```bash
  git clone https://github.com/your_username/image-classification.git
  cd image-classification
  ```
2. Open the Jupyter Notebook file image_classification.ipynb in Jupyter Notebook or any compatible environment.
3. Follow the instructions in the notebook to run each code cell and train the image classification model.

## Dataset

The Dataset used in this project is taken from metal plate scans which have either metal plate pictures or asset pictures.

## Model Evaluation
The trained model is evaluated on a separate test set to measure its performance in terms of accuracy.

