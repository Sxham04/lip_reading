# Deep Learning Lip Reading Pipeline

## Overview
This repository contains an end-to-end deep learning pipeline for automated lip reading. It processes raw video footage of speakers and translates their lip movements into text sequences without the need for audio. The project leverages a hybrid Spatio-Temporal neural network, combining 3D Convolutional Neural Networks (3D-CNN) and Bidirectional LSTMs (Bi-LSTM), trained using Connectionist Temporal Classification (CTC) Loss.

## Features
* **Automated Video Preprocessing:** Dynamically extracts a 112x112 Region of Interest (ROI) focused on the lips, converts frames to grayscale, and applies CLAHE for contrast enhancement.
* **Memory-Efficient Data Generators:** Utilizes dynamic batch processing to load massive 4D video tensors without crashing system RAM.
* **Spatio-Temporal Architecture:** 3D-CNNs extract geometric lip motion, while Bi-LSTMs evaluate contextual phonetic patterns over time.
* **Alignment-Free Training:** CTC Loss allows the model to learn dynamically from unsegmented videos without requiring manual, frame-by-frame text mapping.
* **Strict Evaluation Metrics:** Calculates exact Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance computations.

## Tech Stack
* **Deep Learning Framework:** TensorFlow & Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Handling & Math:** NumPy, Pandas
* **Visualization:** Matplotlib

## Dataset
This project utilizes the [LipReading Dataset](https://www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset) hosted on Kaggle. It consists of short video clips of speakers accompanied by forced-alignment text files that provide the exact phonetic timestamps.

The data ingestion pipeline processes this dataset using the following structure:
* **Videos (`.mpg`):** Raw video data. The preprocessing script automatically standardizes these clips to exactly 75 frames to ensure uniform tensor dimensions for the 3D-CNN.
* **Alignments (`.align`):** Structured text documents containing the ground-truth phonetic timeline. These are parsed to filter out silence (`sil`/`sp`) and extract the exact character sequence required for the CTC loss calculation.

## Model Architecture
1. **Input Stage:** 75 Frames, 112x112 spatial resolution, 1 channel (Grayscale).
2. **3D-CNN & MaxPooling:** Slides 3D mathematical filters across height, width, and time to extract visual features, aggressively down-sampling to preserve core geometric shapes.
3. **Reshape Layer:** Flattens spatial tensors into a sequence format (75 x 14700) for the recurrent layers.
4. **Bi-LSTM:** Processes the sequence in both forward and backward directions to grasp complete linguistic context.
5. **Dropout Layer:** Applies 50% regularization to prevent overfitting on specific lighting or skin tones.
6. **Dense Output:** Softmax activation evaluating a 28-character vocabulary (plus the CTC blank token).

## Evaluation
The model's accuracy is benchmarked against rigorous NLP standards using the Levenshtein Distance matrix:
* **Character Error Rate (CER):** Evaluates the percentage of individual letters incorrectly predicted.
* **Word Error Rate (WER):** Evaluates the percentage of whole words incorrectly predicted.
