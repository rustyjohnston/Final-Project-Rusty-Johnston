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