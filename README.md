# FaceTracker: Real-Time Face Detection and Bounding Box Prediction

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Image Augmentation and Label Transfer](#image-augmentation-and-label-transfer)
   - 3.1 [Capturing Images](#1-capturing-images)
   - 3.2 [Label File Handling](#2-label-file-handling)
   - 3.3 [Loading and Displaying Images](#3-loading-and-displaying-images)
   - 3.4 [Data Augmentation](#4-data-augmentation)
   - 3.5 [Labeling and Coordinates](#5-labeling-and-coordinates)
   - 3.6 [Saving Augmented Images and Annotations](#6-saving-augmented-images-and-annotations)
   - 3.7 [Error Handling](#7-error-handling)
4. [Folder Structure](#folder-structure)
5. [Image Preprocessing](#image-preprocessing)
6. [Label Parsing](#label-parsing)
7. [Data Augmentation](#data-augmentation)
8. [Model Architecture](#model-architecture)
9. [Training and Validation](#training-and-validation)
10. [Evaluation](#evaluation)
11. [Real-Time Face Detection](#real-time-face-detection)
12. [Key Features](#key-features)
13. [Conclusion](#conclusion)

## Overview
This project implements a FaceTracker model for face detection and bounding box prediction using deep learning. The model utilizes a VGG16 backbone pre-trained on ImageNet, with custom layers for face classification and bounding box regression. The training pipeline includes image preprocessing, augmentation, and label handling for object detection tasks. The model is trained on augmented data, with separate datasets for training, validation, and testing. After training, it can be used for real-time face detection via webcam, drawing bounding boxes around detected faces. The project focuses on processing images, annotations, and performing data augmentation for training the model.
## Requirements

To run the code, you will need the following dependencies:

- **OpenCV**: For image capturing and processing.
- **TensorFlow**: For image loading and manipulation.
- **Albumentations**: For image augmentation.
- **Matplotlib**: For visualizing images.
- **JSON**: For handling annotation data.

Install the required libraries using:
```bash
pip install opencv-python tensorflow albumentations matplotlib
```

## Image Augmentation and Label Transfer

### 1. **Capturing Images**
The script captures 80 images from the webcam using OpenCV, storing them in a directory named `data/images`. Each image is saved with a unique filename generated using UUID to avoid overwriting.

### 2. **Label File Handling**
For each image, label files (stored in JSON format) are transferred from a central directory to their corresponding folder in `train`, `test`, and `val` datasets. If an existing label file is found, it is moved to the appropriate location.

### 3. **Loading and Displaying Images**
Using TensorFlow, the images are loaded from the `data/images` directory. The images are decoded and prepared for display, and a subset of images (in batches of 4) is visualized using Matplotlib.

### 4. **Data Augmentation**
Albumentations is used for performing a series of augmentation operations on the images. These operations include:
   - Random cropping
   - Horizontal and vertical flipping
   - Random brightness/contrast adjustment
   - Random gamma adjustment
   - RGB shifting

These augmentations help create more varied training data, making the model more robust.

### 5. **Labeling and Coordinates**
For each image, bounding box coordinates are extracted from the corresponding label files. These coordinates are normalized based on the image's resolution. Augmented images are then generated with new bounding box coordinates.

### 6. **Saving Augmented Images and Annotations**
For each image in the `train`, `test`, and `val` datasets, 60 augmented versions are created. The augmented images and their corresponding JSON label files (with updated bounding box coordinates) are saved in an `aug_data` directory. If no label file exists, default empty bounding boxes are applied.

### 7. **Error Handling**
The script includes error handling to ensure that augmentation is performed correctly, and the program continues execution even in case of errors. If no label file exists for an image, a default annotation with empty bounding boxes is generated.

## Folder Structure

The directory structure is as follows:
``` bash
data/
│
├── images/                # Captured images (original)
│
├── train/                 # Training dataset images and labels
│   ├── images/
│   └── labels/
│
├── test/                  # Test dataset images and labels
│   ├── images/
│   └── labels/
│
├── val/                   # Validation dataset images and labels
│   ├── images/
│   └── labels/
│
aug_data/                  # Augmented data (images and labels)
│
└── labels/                # Store original label files
```

## Image Preprocessing
The images are loaded from the dataset and resized to 120x120 pixels for consistent input to the model. The pixel values are normalized to the range [0,1] for better model performance. The images are stored in `aug_data/train/images`, `aug_data/test/images`, and `aug_data/val/images` for training, testing, and validation respectively.

## Label Parsing
The dataset also includes label files in JSON format stored in `aug_data/train/labels`, `aug_data/test/labels`, and `aug_data/val/labels`. These files contain information about the class (e.g., whether there is a face or not) and the bounding box coordinates. These labels are loaded and processed to match the images.

## Data Augmentation
The images undergo several augmentation techniques, such as:
- Resizing
- Normalization
These augmentations help improve the model’s robustness and generalization.

## Model Architecture
The **FaceTracker** model consists of:
- A **VGG16** backbone without the top classification layers (pre-trained on ImageNet) to extract features from the images.
- Two output heads:
  - **Classification head**: Outputs a binary classification (face or no face).
  - **Bounding box regression head**: Outputs the coordinates of the bounding box around the detected face.

The model is compiled with the Adam optimizer and two loss functions:
- **Binary Crossentropy Loss**: For the classification of the face.
- **Localization Loss**: To calculate the difference between the predicted and true bounding box coordinates.

## Training and Validation
The model is trained using the augmented data, and the performance is evaluated on the validation dataset. TensorBoard is used for monitoring the loss and other metrics during training.

## Evaluation
The model is evaluated on the test dataset, and results are visualized with bounding boxes drawn around detected faces. If the model predicts a face with a high confidence score, a bounding box is drawn around the face on the image.

## Real-Time Face Detection
Once the model is trained, it is saved to a file (`facetracker.h5`). In real-time, the webcam feed is captured, and the model predicts the presence of a face and its bounding box coordinates. If a face is detected, it is highlighted with a bounding box drawn on the webcam frame. The program continues to display this output until the user presses the 'q' key to quit.

## Key Features
- **Image Preprocessing**: Includes image loading, resizing, and normalization.
- **Data Augmentation**: Performs image augmentations for better model robustness.
- **Face Detection**: Detects faces and draws bounding boxes on real-time webcam images.
- **Model Architecture**: Uses VGG16 for feature extraction, with custom heads for classification and bounding box regression.
- **Training**: Optimized using Adam with binary crossentropy and localization loss.
- **Real-Time Prediction**: Uses a webcam to perform real-time face tracking.

## Conclusion
The **FaceTracker** project provides a comprehensive solution for face detection and bounding box prediction. It leverages the power of deep learning, using a pre-trained **VGG16** model to efficiently extract features and predict bounding boxes for face detection. The real-time face detection capabilities make it an ideal solution for live video analysis, and the data augmentation techniques ensure that the model is robust and generalizes well across diverse scenarios. This project demonstrates how image augmentation, label handling, and efficient training pipelines can work together to create a high-performance object detection model.

