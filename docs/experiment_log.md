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