# Code Overview and Reproducibility Guide

This project implements a full deep learning pipeline for defect detection and localization using the Severstal Steel Defect Detection dataset. The workflow progresses from dataset inspection → manifest generation → model training → visualization → final reporting.

---

# Code Structure and File Summaries

## Code/analysis/

### build_results_summary.py
Aggregates final model results into a single summary table for reporting.

**Outputs:**
- Code/outputs/results/final_results_summary.csv

### make_report_plots.py
Generates report-ready figures such as model performance comparisons.

### make_training_plots.py
Creates training curve plots for segmentation models.

**Outputs:**
- multiclass_training_curve.png  
- binary_training_curve.png  

### threshold_sweep.py
Evaluates segmentation performance across different probability thresholds to optimize Dice score.

---

## Code/data/

### build_classification_manifest.py
Builds a dataset manifest for classification (defect vs no-defect).

**Outputs:**
- Code/outputs/manifests/classification_manifest.csv

### build_segmentation_manifest.py
Builds a manifest of labeled images for segmentation tasks.

**Outputs:**
- Code/outputs/manifests/segmentation_manifest.csv

### severstal_segmentation_dataset.py
PyTorch dataset for multiclass segmentation (4-channel masks).

### severstal_binary_segmentation_dataset.py
PyTorch dataset for binary segmentation (defect vs background mask).

---

## Code/models/

### train_binary_classifier.py
TensorFlow-based baseline CNN classifier for defect detection.

### train_binary_classifier_torch.py
PyTorch implementation of binary classifier using GPU acceleration.

**Outputs:**
- binary_cnn_results.csv

### train_unet_segmentation.py
Trains multiclass U-Net (4-class segmentation).

**Outputs:**
- unet_segmentation_results.csv  
- unet_training_history.csv  
- saved model checkpoint  

### train_unet_binary_segmentation.py
Trains binary U-Net (defect vs background).

**Outputs:**
- binary_unet_segmentation_results.csv  
- binary_unet_training_history.csv  
- saved model checkpoint  

---

## Code/utils/

### inspect_severstal.py
Visualizes sample images and decoded masks for sanity checking.

### summarize_severstal_dataset.py
Prints dataset statistics including class distribution and image counts.

### visualize_unet_predictions.py
Generates qualitative prediction overlays for multiclass U-Net.

**Outputs:**
- Code/outputs/figures/example_*.png

### visualize_binary_unet_predictions.py
Generates qualitative prediction overlays for binary U-Net.

**Outputs:**
- Code/outputs/figures/binary_example_*.png

---

# Reproducing Results

## 0. Dataset Setup

Download the dataset from Kaggle:  
**Severstal Steel Defect Detection**

Place files in:


data/raw/
train.csv
train_images/
test_images/


---

## 1. Activate Environment


source .venv/bin/activate


---

## 2. Inspect Dataset


python -m Code.utils.summarize_severstal_dataset
python -m Code.utils.inspect_severstal


---

## 3. Build Manifests


python -m Code.data.build_classification_manifest
python -m Code.data.build_segmentation_manifest


---

## 4. Train Classification Model


python -m Code.models.train_binary_classifier_torch


---

## 5. Train Multiclass Segmentation Model


python -m Code.models.train_unet_segmentation


---

## 6. Train Binary Segmentation Model (Best Model)


python -m Code.models.train_unet_binary_segmentation


---

## 7. Generate Prediction Visualizations


python -m Code.utils.visualize_unet_predictions
python -m Code.utils.visualize_binary_unet_predictions


---

## 8. Generate Plots and Final Summary


python -m Code.analysis.make_training_plots
python -m Code.analysis.make_report_plots
python -m Code.analysis.build_results_summary


---