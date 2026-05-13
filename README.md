# Deep Learning Lip Reading Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An end-to-end deep learning pipeline that translates lip movements in silent video into text sequences. The model uses a hybrid Spatio-Temporal architecture combining 3D Convolutional Neural Networks (3D-CNN) and Bidirectional LSTMs (Bi-LSTM), trained using Connectionist Temporal Classification (CTC) Loss.

---

## Project Structure

```
lip_reading/
├── main_nb.ipynb         # Full pipeline: preprocessing, training, and evaluation
├── metadata.csv          # Dataset metadata and file mappings
├── .gitignore
└── README.md
```

> Trained model weights and the video dataset are not included in this repository. See the [Dataset](#dataset) and [Pretrained Weights](#pretrained-weights) sections below.

---

## Features

- **Automated Video Preprocessing:** Dynamically extracts a 112×112 Region of Interest (ROI) focused on the lips, converts frames to grayscale, and applies CLAHE for contrast enhancement.
- **Memory-Efficient Data Generators:** Uses dynamic batch processing to load massive 4D video tensors without loading the full dataset into memory.
- **Spatio-Temporal Architecture:** 3D-CNNs capture geometric lip motion features, while Bi-LSTMs model contextual phonetic patterns over time.
- **Alignment-Free Training:** CTC Loss allows the model to learn from unsegmented videos without requiring manual, frame-by-frame text annotation.
- **Rigorous Evaluation:** Calculates Character Error Rate (CER) and Word Error Rate (WER) using the Levenshtein Distance algorithm.

---

## Tech Stack

| Purpose | Library |
|---|---|
| Deep Learning | TensorFlow & Keras |
| Computer Vision | OpenCV (`cv2`) |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Sxham04/lip_reading.git
cd lip_reading
```

**2. Install dependencies**
```bash
pip install tensorflow opencv-python numpy pandas matplotlib
```

**3. Download the dataset** (see [Dataset](#dataset) section below) and place it in a `data/` folder at the project root.

**4. Open and run the notebook**
```bash
jupyter notebook main_nb.ipynb
```

---

## Dataset

This project uses the [LipReading Dataset](https://www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset) hosted on Kaggle. It consists of short video clips of speakers accompanied by forced-alignment text files with exact phonetic timestamps.

The pipeline expects the following structure inside `data/`:
```
data/
├── videos/        # Raw .mpg video clips
└── alignments/    # .align files with ground-truth phonetic timelines
```

**Preprocessing steps applied automatically:**
- **Videos (`.mpg`):** Standardized to exactly 75 frames for uniform 3D-CNN input dimensions.
- **Alignments (`.align`):** Silence tokens (`sil`, `sp`) are filtered out; the remaining character sequence is used for CTC loss computation.

---

## Model Architecture

| Stage | Details |
|---|---|
| Input | 75 frames × 112×112 spatial resolution × 1 channel (grayscale) |
| 3D-CNN + MaxPooling | Extracts spatio-temporal visual features; progressively down-samples across height, width, and time |
| Reshape | Flattens spatial tensors to a sequence of shape `(75, 14700)` for recurrent layers |
| Bi-LSTM | Processes the sequence in both directions to capture full linguistic context |
| Dropout (50%) | Reduces overfitting |
| Dense + Softmax | Outputs over a 28-character vocabulary (a–z, space, CTC blank token) |

---

## Pretrained Weights

Trained model weights are hosted separately due to file size constraints.

| Model | Notes | Download |
|---|---|---|
| `lip_reading_v1.keras` | Baseline model | [Download](https://huggingface.co/sxham04/lip_reading/resolve/main/lip_reading_v1.keras) |
| `best_lip_model.keras` | Best single-run checkpoint | [Download](https://huggingface.co/sxham04/lip_reading/resolve/main/best_lip_model.keras) |
| `lip_model_finetuned.keras` | Fine-tuned on expanded data | [Download](https://huggingface.co/sxham04/lip_reading/resolve/main/lip_model_finetuned.keras) |
| `lip_model_finetuned_v2.keras` | Final fine-tuned version | [Download](https://huggingface.co/sxham04/lip_reading/resolve/main/lip_model_finetuned_v2.keras) |

To use a pretrained model, download the `.keras` file and load it in the notebook:
```python
from tensorflow import keras
model = keras.models.load_model("best_lip_model.keras")
```

---

## Evaluation

Model performance is measured using standard speech recognition metrics computed via Levenshtein Distance:

| Metric | Description |
|---|---|
| **CER** (Character Error Rate) | 32% |
| **WER** (Word Error Rate) | 32.2% |

> **Results:** *(Add your best CER/WER scores here once evaluated — this is the most important section for demonstrating the model works.)*

---

## License

This project is licensed under the MIT License.
