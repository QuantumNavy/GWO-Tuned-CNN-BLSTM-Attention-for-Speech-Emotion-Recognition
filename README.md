# GWO-Tuned CNN-BLSTM-Attention for Speech Emotion Recognition

> **A self-optimizing deep learning pipeline** that automatically discovers the best architecture and hyperparameters for speech emotion recognition — no manual tuning required.

---

## Overview

This project applies **Speech Emotion Recognition (SER)** on the [RAVDESS](https://zenodo.org/record/1188976) dataset using a hybrid deep learning model: a **Convolutional Neural Network (CNN)** for spatial feature extraction, a **Bidirectional LSTM (BLSTM)** for temporal modelling, and a **soft Attention mechanism** for context-weighted classification.

What makes this pipeline *self-configuring* is the **Grey Wolf Optimizer (GWO)** — a bio-inspired metaheuristic algorithm — which automatically searches for optimal hyperparameters (learning rate, dropout, filter sizes, LSTM hidden units, optimizer type, batch size) before the final training run begins.

The dataset is expanded **5× via audio augmentation** (noise injection, time shifting, pitch shifting, time stretching) to improve generalization on the relatively small RAVDESS corpus.

---

## Key Features

- **Hybrid Architecture** — CNN → BLSTM → Attention → Softmax, designed for audio spectrogram classification
- **Grey Wolf Optimizer (GWO)** — metaheuristic hyperparameter search; no manual grid search needed
- **5× Audio Augmentation** — four augmentation strategies applied per sample to combat data scarcity
- **Mel-Spectrogram Representation** — 128×128 log-mel spectrograms as visual audio fingerprints
- **End-to-End in PyTorch** — GPU-accelerated, fully reproducible (seeded)
- **RAVDESS Compatible** — built around the standard 8-emotion RAVDESS filename convention

---

## Architecture

```
Input (1 × 128 × 128 Mel-Spectrogram)
        │
        ▼
┌───────────────────────────────┐
│  CNN Block × 3                │
│  Conv2d → BatchNorm → ReLU    │
│  → MaxPool2d                  │
│  Filters: 32 → 64 → 128      │
└───────────────┬───────────────┘
                │  Reshape to sequence
                ▼
┌───────────────────────────────┐
│  Bidirectional LSTM           │
│  hidden_dim = 128 (× 2 dirs) │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Soft Attention               │
│  Linear → Softmax → Weighted  │
│  Sum over time steps          │
└───────────────┬───────────────┘
                │
                ▼
         Dropout → FC → Logits
                │
                ▼
        Emotion Class (8)
```

> Filter sizes, LSTM hidden units, and dropout are **automatically tuned by GWO**.

---

## Grey Wolf Optimizer

GWO mimics the social hierarchy and hunting behaviour of grey wolves. In this pipeline:

- Each **wolf** represents a candidate set of hyperparameters
- **Fitness** is measured as validation accuracy after a quick 3-epoch training run
- The **alpha wolf** (best solution) guides the pack toward better regions of the search space over several iterations
- After GWO converges, the best parameters are used for a full **50-epoch** final training run

**Search Space:**

| Hyperparameter | Range |
|---|---|
| Learning Rate | 1e-4 → 1e-2 (log-uniform) |
| Weight Decay | 1e-6 → 1e-3 (log-uniform) |
| Dropout Rate | 0.2 → 0.5 |
| LSTM Hidden Units | 64 / 96 / 128 / 160 |
| CNN Filter Sets | 16 / 32 / 48 / 64 / 96 / 128 |
| Optimizer | Adam / AdamW / RMSprop |
| Batch Size | 16 / 32 / 64 |

---

## Audio Augmentation (5×)

Each original `.wav` file produces 5 samples total:

| # | Augmentation | Details |
|---|---|---|
| 1 | **Original** | Raw audio, no modification |
| 2 | **Gaussian Noise** | σ = 0.005 additive white noise |
| 3 | **Time Shift** | Random ±25% circular shift |
| 4 | **Pitch Shift** | ±2 semitones via librosa |
| 5 | **Time Stretch** | 0.8× or 1.2× playback rate |

All augmented signals are converted to **128×128 log-Mel spectrograms** entirely in RAM to avoid I/O bottlenecks with mounted drives.

---

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 24 professional actors (12 male, 12 female)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Filename format: `03-01-**05**-02-02-01-12.wav` → emotion ID in position 3 (1-indexed)
- Download: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

---

## Requirements

```
Python >= 3.8
torch >= 1.12
torchaudio
librosa
numpy
scikit-learn
scikit-image
matplotlib
```

Install dependencies:

```bash
pip install torch torchaudio librosa numpy scikit-learn scikit-image matplotlib
```

---

## Usage

### 1. Set the dataset path

Edit this line in the script to point to your downloaded RAVDESS folder:

```python
audio_dir = "/content/drive/MyDrive/audio_speech_actors_01-24"
```

### 2. Run the pipeline

```bash
python ser_gwo_cnn_blstm.py
```

The pipeline will:

1. Walk the dataset directory and load all `.wav` files
2. Apply 4× augmentation per file (producing 5× total samples)
3. Convert all audio to 128×128 log-Mel spectrograms
4. Split data into 80% train / 20% test (stratified)
5. Run GWO hyperparameter search (4 wolves × 3 iterations × 3 epochs each)
6. Train the final model for 50 epochs using the best discovered parameters
7. Plot train vs. test accuracy curves

### 3. Expected output

```
Using device: cuda
Building 5x augmented dataset from audio (this is the heavy step)...
Total samples after 5x augmentation: 7200
Specs shape: (7200, 128, 128)  Labels shape: (7200,)
Train: torch.Size([5760, 1, 128, 128])  Test: torch.Size([1440, 1, 128, 128])
Num classes: 8

GWO Iter 1/3 | Best Acc so far: 63.82%
GWO Iter 2/3 | Best Acc so far: 79.44%
GWO Iter 3/3 | Best Acc so far: 79.44%

Best Params Found by GWO: {
  'lr': 0.00039727962728551547,
  'weight_decay': 1.4558506946611519e-05,
  'dropout_rate': 0.2,
  'lstm_hidden': 192,
  'cnn1_filters': 32,
  'cnn2_filters': 48,
  'cnn3_filters': 16,
  'optimizer': 'Adam',
  'batch_size': 16
}
GWO Validation Accuracy: 79.44%

Epoch  1/50 | Train Acc:  39.38% | Test Acc: 48.89%
Epoch  2/50 | Train Acc:  60.38% | Test Acc: 69.79%
Epoch  3/50 | Train Acc:  73.25% | Test Acc: 76.81%
Epoch  4/50 | Train Acc:  82.40% | Test Acc: 83.47%
Epoch  5/50 | Train Acc:  87.74% | Test Acc: 85.97%
Epoch  6/50 | Train Acc:  91.32% | Test Acc: 91.25%
Epoch  7/50 | Train Acc:  93.35% | Test Acc: 89.31%
Epoch  8/50 | Train Acc:  94.48% | Test Acc: 90.56%
Epoch  9/50 | Train Acc:  95.73% | Test Acc: 92.99%
Epoch 10/50 | Train Acc:  96.39% | Test Acc: 93.47%
Epoch 11/50 | Train Acc:  97.67% | Test Acc: 92.50%
Epoch 12/50 | Train Acc:  97.69% | Test Acc: 94.79%
Epoch 13/50 | Train Acc:  98.32% | Test Acc: 94.38%
...
Epoch 47/50 | Train Acc:  99.95% | Test Acc: 96.88%
Epoch 48/50 | Train Acc: 100.00% | Test Acc: 96.94%  ← Peak Test Accuracy
Epoch 49/50 | Train Acc: 100.00% | Test Acc: 96.81%
Epoch 50/50 | Train Acc: 100.00% | Test Acc: 96.81%
```

---

## Project Structure

```
├── ser_gwo_cnn_blstm.py      # Main script (single file)
├── README.md
└── audio_speech_actors_01-24/
    ├── Actor_01/
    │   └── *.wav
    ├── Actor_02/
    │   └── *.wav
    └── ...
```

---

## Results

### GWO Search

| Metric | Value |
|---|---|
| Wolves | 4 |
| Iterations | 3 |
| GWO Best Validation Accuracy | **79.44%** |
| Best Optimizer Found | Adam |
| Best Learning Rate | 0.000397 |
| Best LSTM Hidden Units | 192 |
| Best Dropout | 0.2 |
| Best Batch Size | 16 |

### Final Training (50 Epochs, GWO-Tuned Parameters)

| Metric | Value |
|---|---|
| Final Train Accuracy | **100.00%** |
| Peak Test Accuracy | **96.94%** (Epoch 48) |
| Test Accuracy at Epoch 50 | 96.81% |
| Test Accuracy at Epoch 10 | 93.47% |

**Convergence:** The model reaches ~91% test accuracy by epoch 6 and crosses 94% by epoch 12, reflecting rapid and stable convergence from GWO-tuned parameters. Training accuracy hits 100% from epoch 48 onward. The test accuracy remains tightly stable at ~96.8% in the final epochs, indicating no meaningful overfitting — a strong result for an 8-class emotion classifier on augmented RAVDESS data.

Training is plotted automatically at the end of the run:

- X-axis: Epoch (1–50)
- Y-axis: Accuracy (%)
- Two curves: Train Accuracy and Test Accuracy

---

## Configuration

Key constants at the top of the script:

```python
seed         = 42       # Global random seed for reproducibility
sample_rate  = 16000    # Audio resampling rate (Hz)
resize_dim   = 128      # Mel-spectrogram output size (128×128)
epochs       = 50       # Final training epochs
n_wolves     = 4        # GWO population size
iterations   = 3        # GWO search iterations
epochs_per_wolf = 3     # Quick evaluation epochs per wolf candidate
```

---

## How GWO Works Here (Brief)

```
Initialize N wolves with random hyperparameter sets
For each GWO iteration:
    Evaluate fitness (val accuracy) of each wolf
    Rank wolves: α (best) > β > δ > ω
    Update each wolf's position toward α using:
        A = 2·a·r₁ − a          (convergence factor)
        D = |C·Xα − Xᵢ|        (distance to alpha)
        Xᵢ ← Xα − A·D          (position update)
    Clamp updated values to valid ranges
    Keep top-N wolves across old and new populations
Return α wolf's parameters → use for final training
```

The parameter `a` decreases linearly from 2 → 0 across iterations, shifting wolves from exploration to exploitation.

---

## Limitations & Future Work

- GWO search is lightweight (3 wolves × 3 iters × 3 epochs) for speed — increasing these will yield better hyperparameters at the cost of more compute
- Augmentation is applied once at data-loading time (static augmentation); online augmentation per epoch could further improve robustness
- The model currently ignores speaker identity — speaker-independent cross-validation would give a more realistic performance estimate
- Possible extensions: add MFCC or delta features alongside Mel spectrograms, try Transformer encoder in place of BLSTM, or experiment with contrastive pre-training


---

## Acknowledgements

- [RAVDESS Dataset](https://zenodo.org/record/1188976) — S. Livingstone & F. Russo
- [librosa](https://librosa.org/) — audio feature extraction
- [PyTorch](https://pytorch.org/) — deep learning framework
- Grey Wolf Optimizer — Mirjalili et al. (2014), *Advances in Engineering Software*
