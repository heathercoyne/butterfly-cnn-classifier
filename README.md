# ButterflyNet – CNN Butterfly Image Classifier

This project implements a convolutional neural network (CNN) for butterfly image classification using **PyTorch**.  
The goal of the project was to iteratively design and improve a deep learning model through architectural changes and hyperparameter tuning.

The project was completed as part of an **Artificial Intelligence course project**.

---

# Project Overview

The objective of this project is to classify butterfly species from images using a deep learning model.

The development process follows an **experimental workflow**, where the model architecture is gradually improved across several versions:

1. Baseline CNN model  
2. Deeper CNN with additional convolution layers  
3. CNN with downsampling, batch normalization, and dropout  
4. CNN with pooling and optimized hyperparameters  

Each version of the model is implemented as a separate training script.

The final model achieves **85.12% classification accuracy** on the test dataset.

---

# Model Development Process

## Baseline Model

The first model uses a simple CNN architecture consisting of:

- two convolution layers
- ReLU activation
- a fully connected classification layer

Data preprocessing includes:

- random cropping
- horizontal flipping
- color jitter
- normalization

The dataset is split into:

- 70% training
- 15% validation
- 15% testing

Baseline model performance:

**Accuracy: 0.5863**

---

## Version 2 – Deeper CNN (`train2.0`)

The second version increases the network depth to **five convolution layers**.

Additional improvements:

- optional stronger data augmentation
- deeper feature extraction

However, simply increasing network depth did not significantly improve performance due to **training instability and overfitting**.

---

## Version 3 – Downsampling + BatchNorm + Dropout (`train2.1`)

The third version introduces several important improvements:

- stride-based downsampling
- batch normalization after convolution layers
- dropout regularization in the classifier

These changes stabilize training and improve model generalization.

Best performance in this stage:

**Accuracy: 0.8185**

---

## Version 4 – Pooling and Hyperparameter Optimization (`train2.2`)

The final version adds optional pooling layers:

- **max pooling**
- **average pooling**

Experiments show that **max pooling performs slightly better** for this dataset.

Additional improvements include:

- momentum tuning
- weight decay optimization
- learning rate adjustments

Final model performance:

**Accuracy: 0.8512**

---

# Results

## Baseline Model
Accuracy: **0.5863**

<img width="495" height="495" alt="image" src="https://github.com/user-attachments/assets/333bf26f-7cab-47a9-9e8f-7e9b21086e02" />


---

## Improved Model (`train2.1`)
Accuracy: **0.8185**

<img width="492" height="492" alt="image" src="https://github.com/user-attachments/assets/d8fb1206-ba9e-424c-8e29-1a22b6117c5d" />


---

## Final Model (`train2.2`)
Accuracy: **0.8512**

<img width="532" height="532" alt="image" src="https://github.com/user-attachments/assets/1bdb5f2a-e6fa-43f6-a4d1-fa9de7201d43" />


The confusion matrices illustrate how classification accuracy improves as the model architecture becomes more sophisticated.

Most remaining errors occur between visually similar butterfly species.

---

# Repository Structure
butterfly-classifier
│
├ train.py
├ train2.0.py
├ train2.1.py
├ train2.2.py
│
├ images
│ baseline_confusion_matrix.png
│ train21_confusion_matrix.png
│ train22_confusion_matrix.png
│
├ requirements.txt
├ .gitignore
└ README.md

---

# Installation

Install dependencies using:

pip install -r requirements.txt


---

# Running the Models

Baseline training:
python train.py

Improved models:
python train2.0.py
python train2.1.py
python train2.2.py

---

# Dataset

The butterfly dataset used for training is **not included in this repository due to size limitations**.

The code expects the dataset to be placed in the following directory:
ButterflyClassificationDataset/

---

# Key Techniques Used

- Convolutional Neural Networks (CNN)
- Data augmentation
- Batch normalization
- Dropout regularization
- Max pooling vs average pooling comparison
- Hyperparameter tuning
- Confusion matrix analysis

---

# Author

Heather Xin Coyne 
Tsinghua University  
Artificial Intelligence Course Project
