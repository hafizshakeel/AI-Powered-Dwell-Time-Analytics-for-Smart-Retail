# 🔄Training Pipeline Module

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, modular pipeline for training YOLOv8 on custom dataset to detect people in retail or other similar environments.

<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" width="800">
</p> -->

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Pipeline Workflow](#-pipeline-workflow)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Customization](#-customization)
- [Results & Evaluation](#-results--evaluation)
- [Troubleshooting](#-troubleshooting)

## 🔭 Overview

This Training Pipeline provides an end-to-end solution for object detection on custom dataset using YOLOv8 models. This pipeline handles everything from data acquisition to model evaluation, making it easy to train high-performance models with minimal effort.

## ✨ Features

- 🔄 **Automated Workflow**: Complete pipeline from data ingestion to model evaluation
- 📊 **Data Validation**: Ensures dataset structure meets YOLOv8 requirements
- 🧠 **Transfer Learning**: Leverages pre-trained YOLOv8 weights for faster convergence
- 📈 **Performance Metrics**: Comprehensive evaluation with precision, recall, and mAP
- 🛠️ **Configurable**: Easily adjust training parameters via configuration files
- 📁 **Artifact Management**: Organized storage of datasets, models, and evaluation results

## 🗂️ Project Structure

```
customer_vision/
├── components/               # Core pipeline components
│   ├── data_ingestion.py     # Dataset download and preparation
│   ├── data_validation.py    # Dataset structure validation
│   ├── model_trainer.py      # YOLOv8 model training
│   └── model_evaluation.py   # Model performance evaluation
├── constants/                # Configuration constants
├── entity/                   # Data classes for configuration and artifacts
├── exception/                # Custom exception handling
├── logger/                   # Logging utilities
├── pipeline/                 # Pipeline orchestration
│   └── training_pipeline.py  # Main pipeline implementation
└── utils/                    # Helper utilities
```

## 🔄 Pipeline Workflow

The training pipeline follows these key steps:

1. **Data Ingestion** 📥
   - Downloads dataset from configured URL
   - Extracts and organizes data in the required format
   - Verifies dataset integrity

2. **Data Validation** ✅
   - Validates dataset structure (train/valid splits)
   - Checks for required files (images, labels, data.yaml)
   - Ensures proper configuration for customer detection

3. **Model Training** 🧠
   - Initializes YOLOv8 with pre-trained weights
   - Configures training parameters (epochs, batch size, image size)
   - Trains model on the prepared dataset
   - Saves training artifacts (model weights, plots, metrics)

4. **Model Evaluation** 📊
   - Evaluates model on validation dataset
   - Calculates precision, recall, mAP metrics
   - Generates confusion matrix and performance visualizations
   - Validates model against minimum performance thresholds

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/hafizshakeel/AI-Powered-Dwell-Time-Analytics-for-Smart-Retail.git
cd customer_vision

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Complete Pipeline

To execute the full training pipeline:

```bash
python -m customer_vision.pipeline.training_pipeline
```

### Running Individual Components

You can also run specific components of the pipeline:

```python
from customer_vision.pipeline.training_pipeline import TrainPipeline

# Initialize the pipeline
pipeline = TrainPipeline()

# Run data ingestion only
data_ingestion_artifact = pipeline.start_data_ingestion()

# Run data validation only
data_validation_artifact = pipeline.start_data_validation(data_ingestion_artifact)

# Run model training only
model_trainer_artifact = pipeline.start_model_trainer()

# Run model evaluation only
model_evaluation_artifact = pipeline.start_model_evaluation(model_trainer_artifact)
```

## ⚙️ Configuration

The pipeline is highly configurable through constants defined in `constants/training_pipeline/__init__.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DOWNLOAD_URL` | Roboflow URL | Source dataset URL |
| `MODEL_TRAINER_PRETRAINED_WEIGHT_NAME` | `yolov8s.pt` | Pre-trained weights |
| `MODEL_TRAINER_NO_EPOCHS` | `1` | Training epochs |
| `MODEL_TRAINER_BATCH_SIZE` | `8` | Batch size |
| `MODEL_TRAINER_IMAGE_SIZE` | `416` | Input image size |
| `MODEL_EVALUATION_THRESHOLD` | `0.25` | Detection confidence threshold |
| `MODEL_EVALUATION_IOU_THRESHOLD` | `0.5` | IoU threshold for NMS |
| `MODEL_EVALUATION_MIN_MAP_THRESHOLD` | `0.5` | Minimum mAP for acceptance |

## 🔧 Customization

### Using Your Own Dataset

1. Prepare your dataset in YOLOv8 format:
   - `train/images/` - Training images
   - `train/labels/` - Training labels (YOLO format)
   - `valid/images/` - Validation images
   - `valid/labels/` - Validation labels (YOLO format)
   - `data.yaml` - Dataset configuration

2. Update the `DATA_DOWNLOAD_URL` in constants or provide a local path

### Modifying Model Architecture

To use a different YOLOv8 model variant:

1. Change `MODEL_TRAINER_PRETRAINED_WEIGHT_NAME` to one of:
   - `yolov8n.pt` (nano)
   - `yolov8s.pt` (small)
   - `yolov8m.pt` (medium)
   - `yolov8l.pt` (large)
   - `yolov8x.pt` (extra large)

## 📊 Results & Evaluation

After training, results are stored in the artifacts directory:

```
artifacts/
├── data_ingestion/           # Raw and processed dataset
├── data_validation/          # Validation reports
├── model_trainer/            # Trained model weights and plots
│   ├── plots/                # Performance visualizations
│   │   ├── confusion_matrix.png
│   │   ├── results.png       # Training metrics plot
│   │   └── val_batch0_pred.jpg  # Example predictions
│   ├── weights/              # Model weights
│   └── results.csv           # Training metrics
└── model_evaluation/         # Evaluation results and metrics
```

## 🔍 Troubleshooting

### Common Issues

- **Dataset Download Fails**: Check internet connection and URL validity
- **Training Errors**: Ensure CUDA is available for GPU training
- **Low Performance**: Try increasing epochs or adjusting batch size
- **Out of Memory**: Reduce batch size or use a smaller model variant

For additional help, check the logs in the `logs/` directory.

 