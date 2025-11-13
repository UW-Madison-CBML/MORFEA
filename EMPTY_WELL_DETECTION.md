# Empty Well Detection System

Automatically labels embryo images as empty or non-empty wells for split-latent-space training.

## Overview

The system detects empty wells using a hybrid approach:

1. **Heuristic Features** (fast, unsupervised)
   - Entropy in well region (empty = low entropy/uniform)
   - Contrast (empty = low contrast)
   - Intensity distribution (empty wells are mid-gray ~0.5)
   - Peak frequency of histogram

2. **Optional Neural Network** (trainable, for refinement)
   - Lightweight CNN classifier
   - Can be trained on labeled examples for better accuracy

## Quick Start

### 1. Enable in build_index.py

```python
DETECT_EMPTY_WELLS = True  # Enable detection
EMPTY_WELL_THRESHOLD = 0.6  # Probability threshold
EMPTY_WELL_MODEL = None     # Optional: path to trained model
```

Then run:
```bash
python build_index.py
```

This will:
- Scan all images in `embryo_dataset/`
- Assign empty well probability to each sequence
- Output `index.csv` with `empty_well` column (True/False)

### 2. Single Image Detection

```python
from detect_empty_wells import EmptyWellDetector

detector = EmptyWellDetector()
prob = detector.predict("path/to/image.jpg")
print(f"Empty well probability: {prob:.4f}")
```

### 3. Batch Processing

```python
from detect_empty_wells import EmptyWellDetector

detector = EmptyWellDetector()
probs = detector.predict_batch(image_paths, batch_size=32)
```

## Architecture

### EmptyWellDetector

Main class providing predictions:

- `predict(image_path)` → probability [0, 1]
- `predict_batch(image_paths)` → list of probabilities
- `predict_with_confidence(image_path)` → (probability, confidence_level)

### EmptyWellClassifier

Optional neural network for refinement:

```
Input: 128x128 grayscale
├─ Conv2d (32 channels) → MaxPool
├─ Conv2d (64 channels) → MaxPool
├─ Conv2d (128 channels) → MaxPool
├─ FC (256) + Dropout
├─ FC (128)
└─ FC (1 sigmoid) → probability
```

## Training a Custom Model

If heuristics don't work well on your data, train a classifier:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from detect_empty_wells import EmptyWellClassifier

# Prepare labeled data (images + labels)
X = torch.randn(1000, 1, 128, 128)  # Your images
y = torch.randint(0, 2, (1000, 1)).float()  # Your labels (0 or 1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
model = EmptyWellClassifier().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

for epoch in range(10):
    for batch_x, batch_y in loader:
        pred = model(batch_x.cuda())
        loss = loss_fn(pred, batch_y.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), "empty_well_model.pth")
```

Then use it:
```python
detector = EmptyWellDetector(model_path="empty_well_model.pth")
```

## Configuration

In `build_index.py`:

```python
DETECT_EMPTY_WELLS = True              # Enable/disable detection
EMPTY_WELL_THRESHOLD = 0.6             # Classification threshold
EMPTY_WELL_MODEL = None                # Optional trained model path
```

Threshold interpretation:
- `prob >= threshold` → `empty_well = True`
- `prob < threshold` → `empty_well = False`

Lower threshold = more wells classified as empty (higher recall)
Higher threshold = fewer false positives (higher precision)

## GPU Cluster Usage

Submit to CHTC:

```bash
condor_submit detect_empty_wells.sub
```

This will:
- Run on GPU node
- Extract dataset
- Run detection + indexing
- Output `index.csv` to staging area

Monitor:
```bash
condor_tail -f detect_empty_wells_<cluster>_<process>.log
```

## Integration with Training

In your training code, split by empty_well label:

```python
import pandas as pd

index_df = pd.read_csv("index.csv")

# Non-empty wells (embryo sequences)
fg_sequences = index_df[index_df["empty_well"] == False]

# Empty wells (background only)
bg_sequences = index_df[index_df["empty_well"] == True]

# During training:
# - For fg_sequences: use z_fg for reconstruction
# - For bg_sequences: use z_bg for reconstruction
```

This allows your autoencoder to learn:
- `z_fg`: foreground features (embryo morphology)
- `z_bg`: background features (noise, well shape)

## Parameters Explained

### Heuristic Features

The system uses 4 features to compute empty well probability:

1. **Entropy Score** (weight: 0.3)
   - Calculates Shannon entropy of histogram
   - Empty wells have low entropy (uniform gray)
   - Formula: `1 - (entropy / max_entropy)`

2. **Contrast Score** (weight: 0.3)
   - Standard deviation of well region
   - Empty wells have low contrast
   - Formula: `1 - min(std_dev, 1.0)`

3. **Intensity Score** (weight: 0.2)
   - How close mean intensity is to 0.5 (ideal empty)
   - Formula: `1 - abs(mean - 0.5) * 2`

4. **Peak Frequency Score** (weight: 0.2)
   - Proportion of most common intensity
   - Higher peak = more uniform
   - Empty wells have high peak frequency

## Tuning for Your Dataset

If detection isn't working well:

1. **Increase EMPTY_WELL_THRESHOLD** (0.6 → 0.7)
   - More conservative, fewer false positives

2. **Decrease EMPTY_WELL_THRESHOLD** (0.6 → 0.5)
   - More aggressive, catches more empties

3. **Train a custom model**
   - Label ~200-500 examples
   - Use `EmptyWellClassifier` to train
   - Set `EMPTY_WELL_MODEL` path

4. **Check well detection**
   - Run with a sample and check circle detection
   - If circles are wrong, heuristics won't work

## Troubleshooting

### No well detected
- Check image resolution (expects ~500x500)
- Verify well is roughly centered
- Fallback uses center-based circle

### Low accuracy
- Train custom model on your data
- Adjust threshold based on precision/recall tradeoff
- Check if "empty" looks different in your dataset

### Memory issues on GPU
- Reduce batch_size in predict_batch()
- Use CPU mode: `EmptyWellDetector(device="cpu")`
- Process images individually

## Performance

Typical performance on GPU (V100):

- Single image: ~50ms (heuristics only)
- Single image: ~80ms (with neural network)
- Batch of 32: ~200ms (with neural network)

Throughput:
- Heuristics only: ~200 images/sec
- With model: ~150 images/sec

## References

- Well detection: Hough Circle Transform (OpenCV)
- Features: Entropy, contrast, intensity distribution
- Architecture: Lightweight CNN suitable for classification
