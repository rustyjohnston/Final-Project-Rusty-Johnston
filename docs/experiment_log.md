# Experiment Log

## 2026-04-23

### Objective
Create first Severstal dataset inspection script and verify RLE mask decoding.

### Files Changed
- `Code/utils/inspect_severstal.py`

### Environment
- Mac for editing in PyCharm
- AWS for execution/training
- Python 3.12 on AWS
- Python 3.11 on Mac

### Actions
- Downloaded and extracted the Severstal Steel Defect Detection dataset on AWS
- Loaded `train.csv`
- Decoded one RLE mask
- Overlaid the decoded mask on the original steel image
- Saved `overlay_example.png`

### Results
- Overlay aligned correctly with a visible defect region
- Confirmed image/mask orientation and RLE decoding are correct

### Notes
- The current visualization uses an opaque red mask
- Future visualizations should use transparent overlays and multi-class overlays

### Next Step
- Upgrade the inspection script to support transparent overlays and multi-class mask visualization

## 2026-04-23

### Objective
Upgrade Severstal inspection visualization to support transparent overlays and multi-class masks per image.

### Files Changed
- `Code/utils/inspect_severstal.py`

### Environment
- Mac for editing in PyCharm
- AWS for execution
- Python 3.12 on AWS

### Actions
- Updated the inspection script to use relative project paths
- Added transparent mask blending instead of opaque overwrite
- Grouped labels by image to combine all available class masks
- Added color coding for all four defect classes
- Saved both the original image and a multi-class overlay image

### Results
- Created improved visualization outputs for inspection and debugging
- Confirmed the pipeline can aggregate multiple class masks for a single image
- Produced report-friendly figures for dataset understanding

### Notes
- These overlays are more useful than opaque masks because the underlying steel texture remains visible
- This is a better basis for report figures and segmentation sanity checks

### Next Step
- Build a classification dataset pipeline for defect vs. no-defect and class-level summaries
## 2026-04-23

### Objective
Summarize the Severstal dataset structure and verify whether negative training images are represented in the label CSV.

### Files Changed
- `Code/utils/summarize_severstal_dataset.py`

### Environment
- Mac for editing in PyCharm
- AWS for execution
- Python 3.12 on AWS

### Actions
- Counted image files in `train_images`
- Compared file inventory against `train.csv`
- Measured number of labeled images and rows
- Summarized how many defect classes appear per image

### Results
- Confirmed severe class imbalance, especially domination by Class 3
- Began verifying whether no-defect images are missing from the processed CSV
- Produced a dataset inventory baseline to guide classification dataset design

### Notes
- The processed CSV appears to contain only defect-positive rows
- Need to verify whether negative examples exist on disk but are absent from the CSV

### Next Step
- Define binary classification labels using file inventory and CSV presence

## 2026-04-23

### Objective
Build classification manifest combining labeled and unlabeled images.

### Files Changed
- `Code/data/build_classification_manifest.py`

### Actions
- Enumerated all image files from disk
- Merged with defect labels from train.csv
- Created binary and multi-label targets
- Saved classification_manifest.csv

### Results
- Confirmed presence of ~5900 no-defect images
- Built dataset suitable for classification training

### Notes
- Dataset is highly imbalanced (class 3 dominant)
- Binary classification is now straightforward

### Next Step
- Train first binary CNN baseline
## 2026-04-23

### Objective
Train first binary CNN baseline for defect detection.

### Files Changed
- `Code/models/train_binary_classifier.py`

### Environment
- Mac for editing in PyCharm
- AWS for execution
- Python 3.12 on AWS
- TensorFlow 2.21.0
- CPU execution forced due to cuDNN mismatch

### Actions
- Installed TensorFlow on AWS
- Attempted GPU execution
- Encountered cuDNN version mismatch
- Disabled CUDA for TensorFlow execution
- Trained a simple CNN for 5 epochs
- Evaluated on held-out test set

### Results
- Binary F1 Score: 0.8440
- Binary Accuracy: 0.8341
- Best validation accuracy observed: approximately 0.8131

### Notes
- This establishes a working baseline for defect/no-defect classification
- Dataset is reasonably balanced for binary classification
- GPU was intentionally disabled for TensorFlow due to cuDNN mismatch
- Future improvement options include augmentation, class-specific multi-label classification, or switching to PyTorch for GPU use

### Next Step
- Save model metrics/results to a tracked output file and then move toward multi-label classification or segmentation.
## 2026-04-23

### Objective
Train binary CNN baseline using PyTorch with GPU acceleration and implement tracked results logging.

### Files Changed
- `Code/models/train_binary_classifier_torch.py`
- `Code/outputs/results/binary_cnn_results.csv`

### Environment
- Mac for editing in PyCharm
- AWS for execution
- Python 3.12 on AWS
- PyTorch 2.5.1 with CUDA 12.1
- GPU: NVIDIA A10G

### Actions
- Installed PyTorch with CUDA support on AWS
- Verified GPU availability
- Implemented binary CNN classifier in PyTorch
- Trained model for 5 epochs on full dataset
- Evaluated on held-out test set
- Logged results to tracked CSV artifact

### Results
- Accuracy: 0.8039
- F1 Score: 0.8183
- Precision: 0.8043
- Recall: 0.8327
- Confusion Matrix:
  - TN: 911
  - FP: 270
  - FN: 223
  - TP: 1110

### Notes
- GPU training successfully enabled via PyTorch
- Training significantly faster than CPU-based TensorFlow run
- Results slightly lower than TF baseline, but pipeline is now GPU-capable
- Dataset appears reasonably balanced for binary classification

### Next Step
- Add confusion matrix visualization and result plots
- Begin transition to segmentation (U-Net) using Dice metric

## 2026-04-23

### Objective
Create segmentation manifest for defect images.

### Files Changed
- `Code/data/build_segmentation_manifest.py`
- `Code/outputs/manifests/segmentation_manifest.csv`

### Environment
- Mac for editing
- AWS for execution
- PyTorch environment with CUDA available

### Actions
- Filtered dataset to only defect-positive rows
- Aggregated labels to one row per image
- Created multi-label indicators per class
- Saved segmentation manifest for training

### Results
- Built dataset index for segmentation pipeline
- Identified distribution of number of defect classes per image
- Confirmed multi-class defect presence

### Notes
- Segmentation dataset includes only defect images
- Class imbalance persists (class 3 dominant)
- Manifest will be used for U-Net training

### Next Step
- Build mask loader and dataset class for segmentation training

## 2026-04-23

### Objective
Create segmentation manifest for defect images.

### Files Changed
- `Code/data/build_segmentation_manifest.py`
- `Code/outputs/manifests/segmentation_manifest.csv`

### Environment
- Mac for editing
- AWS for execution
- Python 3.12 on AWS
- PyTorch/CUDA environment available

### Actions
- Filtered dataset to defect-positive images
- Aggregated rows to one record per image
- Added per-class presence indicators
- Saved segmentation manifest for U-Net training

### Results
- Total defect images: 6666
- Images with 1 defect class: 6239
- Images with 2 defect classes: 425
- Images with 3 defect classes: 2
- Class 1 images: 897
- Class 2 images: 247
- Class 3 images: 5150
- Class 4 images: 801

### Notes
- Most segmentation examples contain only one defect class
- Class 3 dominates the segmentation dataset
- Class imbalance should be considered when evaluating Dice by class

### Next Step
- Build PyTorch segmentation dataset and mask decoder for U-Net training.

## 2026-04-23

### Objective
Build PyTorch segmentation dataset and mask decoder.

### Files Changed
- `Code/data/severstal_segmentation_dataset.py`

### Actions
- Implemented RLE decoding for masks
- Created PyTorch Dataset for segmentation
- Built 4-channel mask output (one per class)
- Verified correct tensor shapes

### Results
- Image tensor shape: [1, 128, 128]
- Mask tensor shape: [4, 128, 128]
- Successfully generated segmentation-ready dataset

### Notes
- Masks are multi-channel (not single-label)
- Ready for multi-class segmentation training
- Data pipeline now complete for U-Net

### Next Step
- Implement U-Net model and Dice loss

## 2026-04-23

### Objective
Train first PyTorch U-Net baseline for defect segmentation.

### Files Changed
- `Code/models/train_unet_segmentation.py`
- `Code/outputs/results/unet_segmentation_results.csv`

### Environment
- Mac for editing
- AWS for execution
- Python 3.12 on AWS
- PyTorch with CUDA
- GPU: NVIDIA A10G

### Actions
- Implemented a small U-Net architecture
- Used 4-channel masks, one channel per defect class
- Trained for 5 epochs
- Evaluated validation Dice coefficient
- Saved tracked results to CSV

### Results
- Best validation Dice: 0.3147
- Best epoch: 1
- Later epochs decreased in Dice, suggesting overfitting, threshold sensitivity, or class-imbalance effects

### Notes
- This establishes a working segmentation baseline
- Checkpoint was saved locally on AWS but not committed due to model artifact size
- Future work should improve Dice using threshold tuning, augmentation, class weighting, or longer controlled training

### Next Step
- Generate prediction overlays from the trained U-Net checkpoint for qualitative segmentation evaluation.

## 2026-04-23

### Objective
Generate qualitative U-Net segmentation prediction figures.

### Files Changed
- `Code/utils/visualize_unet_predictions.py`
- `Code/outputs/figures/example_0.png`
- `Code/outputs/figures/example_1.png`
- `Code/outputs/figures/example_2.png`
- `Code/outputs/figures/example_3.png`
- `Code/outputs/figures/example_4.png`

### Environment
- Mac for editing
- AWS for execution
- PyTorch/CUDA on NVIDIA A10G

### Actions
- Loaded the trained U-Net baseline checkpoint
- Ran inference on sample Severstal images
- Created side-by-side figures showing original image, ground-truth mask, and predicted mask

### Results
- Saved five qualitative prediction figures for report use

### Notes
- These figures help interpret the Dice score visually
- Future improvement should compare successful and failed segmentation examples

### Next Step
- Improve segmentation Dice using threshold tuning and/or longer controlled training.

## 2026-04-23

### Objective
Evaluate segmentation prediction threshold sensitivity for Dice score.

### Files Changed
- `Code/analysis/threshold_sweep.py`

### Environment
- AWS execution
- PyTorch/CUDA on NVIDIA A10G

### Actions
- Loaded the trained U-Net baseline checkpoint
- Evaluated Dice over thresholds from 0.10 to 0.90
- Used a small 50-image sample for fast threshold analysis

### Results
- Best sampled threshold: 0.40
- Best sampled Dice: 0.2354
- Threshold results:
  - 0.10: 0.2071
  - 0.20: 0.2254
  - 0.30: 0.2306
  - 0.40: 0.2354
  - 0.50: 0.2296
  - 0.60: 0.2172
  - 0.70: 0.2024
  - 0.80: 0.1582
  - 0.90: 0.0806

### Notes
- Sampled threshold Dice was lower than training-script validation Dice, likely because the sweep used the first 50 dataset images rather than the same validation split.
- Threshold tuning alone does not currently solve the weak segmentation performance.
- Need to evaluate using a consistent validation split before reporting final threshold results.

### Next Step
- Update U-Net training/evaluation to use a consistent validation split and save validation prediction figures.


## 2026-04-23

### Objective
Create final results summary artifact for report preparation.

### Files Changed
- `Code/analysis/build_results_summary.py`
- `Code/outputs/results/final_results_summary.csv`
- `Code/outputs/results/unet_segmentation_results.csv`
- `Code/outputs/results/unet_training_history.csv`

### Environment
- AWS execution
- PyTorch/CUDA on NVIDIA A10G

### Actions
- Re-ran U-Net baseline to refresh clean segmentation results.
- Fixed malformed results CSV by regenerating the U-Net results file.
- Built a final summary table combining binary classification and segmentation metrics.
- Committed tracked results artifacts to GitHub.

### Results
- Binary classification F1: 0.8183
- Binary classification accuracy: 0.8039
- Segmentation Dice coefficient: 0.3253

### Notes
- The final summary CSV provides a concise report-ready metrics table.
- Binary classification performance is acceptable for a baseline.
- Segmentation performance is a valid first U-Net baseline but should be described as needing improvement.
- The project now has tracked code, logs, metrics, and qualitative figures.

### Next Step
- Prepare report figures and final project narrative.


## 2026-04-23

### Objective
Train improved binary U-Net segmentation model for defect localization.

### Files Changed
- `Code/models/train_unet_binary_segmentation.py`
- `Code/outputs/results/binary_unet_segmentation_results.csv`

### Environment
- AWS execution
- PyTorch/CUDA on NVIDIA A10G

### Actions
- Reformulated segmentation as binary defect localization
- Combined all defect classes into one mask
- Trained U-Net for 20 epochs
- Logged best binary Dice result to tracked CSV artifact

### Results
- Best binary validation Dice: 0.5597
- Previous 4-class U-Net Dice: 0.3253
- Binary formulation improved segmentation performance by approximately 0.2344 Dice points

### Notes
- Binary defect localization better matches the practical inspection task of identifying defect regions
- Multi-class segmentation remains future work due to strong class imbalance
- Performance is below Kaggle-winning solutions, which used more advanced architectures, preprocessing, augmentation, thresholding, and competition-specific optimization

### Next Step
- Generate final binary segmentation prediction figures for report.