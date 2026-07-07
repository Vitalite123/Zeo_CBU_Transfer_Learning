# BiGRU Model Experiment Report: Zeolite Framework Type Prediction

## Overview

This report details the experiment of using Bidirectional GRU (Bidirectional Gated Recurrent Unit) model for zeolite framework type prediction. The model converts 104-dimensional tabular features into sequence data, utilizes BiGRU to capture dependencies between features, and combines attention mechanism for classification prediction.

---

## 1. Principles and Algorithms

### 1.1 GRU Unit Principle

GRU (Gated Recurrent Unit) is a simplified version of LSTM that controls information flow through gating mechanisms:

**Update Gate**:
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**Reset Gate**:
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**Candidate Hidden State**:
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$

**Final Hidden State**:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 1.2 Bidirectional GRU

Bidirectional GRU processes sequences from both forward and backward directions simultaneously:

$$\overrightarrow{h_t} = \text{GRU}_{\text{forward}}(x_t, \overrightarrow{h_{t-1}})$$
$$\overleftarrow{h_t} = \text{GRU}_{\text{backward}}(x_t, \overleftarrow{h_{t+1}})$$
$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

**Advantages**:
- Forward GRU captures historical information
- Backward GRU captures future information
- Combines context from both directions to improve feature representation capability

### 1.3 Attention Mechanism

Additive attention mechanism is used to weighted aggregate sequence outputs:

$$e_t = v^T \tanh(W h_t + b)$$
$$\alpha_t = \text{softmax}(e_t)$$
$$c = \sum_{t=1}^{T} \alpha_t h_t$$

---

## 2. Model Architecture

### 2.1 Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BiGRU for Zeolite Prediction                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: 104-dimensional feature vector                                   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                Reshape: (batch, 104) → (batch, 26, 4)           │   │
│  │                  Convert features to 26 time steps, 4 features per step              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Feature Embedding: Linear(4, 64)                   │   │
│  │                  Feature embedding for each time step                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Bidirectional GRU (2 layers)                  │   │
│  │                                                                 │   │
│  │    ┌──────────────────┐    ┌──────────────────┐                │   │
│  │    │   GRU Layer 1    │    │   GRU Layer 2    │                │   │
│  │    │  Hidden: 128     │───▶│  Hidden: 128     │                │   │
│  │    │  Bidirectional   │    │  Bidirectional   │                │   │
│  │    └──────────────────┘    └──────────────────┘                │   │
│  │                                                                 │   │
│  │    Output: (batch, 26, 256)  [128 forward + 128 backward]                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Attention Mechanism                          │   │
│  │                                                                 │   │
│  │    e = vᵀtanh(Wh + b)        Compute attention scores                      │   │
│  │    α = softmax(e)            Normalize attention weights                    │   │
│  │    c = Σαh                    Weighted sum                            │   │
│  │                                                                 │   │
│  │    Output: (batch, 256)        Context vector                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Fully Connected Layers                      │   │
│  │                                                                 │   │
│  │    FC1: Linear(256, 512) + BatchNorm + ReLU + Dropout(0.3)     │   │
│  │    FC2: Linear(512, 256) + BatchNorm + ReLU + Dropout(0.3)     │   │
│  │    FC3: Linear(256, 231)   Output layer                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  Output: 231-class probability distribution                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Model Parameter Configuration

| Parameter | Value | Description |
|------|-----|------|
| Sequence Length | 26 | 104/4 = 26 |
| Features per Step | 4 | Feature embedding input |
| Hidden Layer Size | 128 | GRU hidden units |
| GRU Layers | 2 | Two-layer BiGRU |
| Dropout | 0.3 | Regularization |
| Total Parameters | 802,600 | Trainable parameters |

### 2.3 Training Configuration

| Parameter | Value |
|------|-----|
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Loss Function | CrossEntropyLoss (label_smoothing=0.1) |
| Epochs | 30 |
| Early Stopping | patience=15 |
| LR Scheduler | ReduceLROnPlateau |

---

## 3. Dataset Description

### 3.1 Data Scale

| Dataset | Samples | Proportion |
|--------|--------|------|
| Training Set | 12,537 | 70% |
| Validation Set | 2,640 | 15% |
| Test Set | 2,641 | 15% |
| **Total** | **17,818** | 100% |

### 3.2 Feature Description

| Feature Category | Number of Features | Description |
|----------|--------|------|
| Element Composition | 45 | Si, Al, P, Na, etc. |
| OSDA Index | 3 | osda1/2/3_index |
| Synthesis Conditions | 4 | Temperature, time, seed, rotation |
| Aging Conditions | 2 | Temperature, time |
| pH Conditions | 2 | Acid, base |
| Gel Ratios | 5 | H2O_T, OH_T, Gel_Si_Al, etc. |
| OSDA Descriptors | 33 | 11 descriptors × 3 OSDAs |
| Aggregated Features | 10 | osda_avg/max/min, etc. |
| **Total** | **104** | |

### 3.3 Target Variable

- **Prediction Target**: Code1 (Zeolite framework type)
- **Number of Classes**: 231 different frameworks
- **Common Frameworks**: MFI, CHA, *BEA, AFI, MTW, etc.

---

## 4. Experimental Results

### 4.1 Overall Performance

| Metric | Value | Description |
|------|-----|------|
| **Test Accuracy** | **76.37%** | Test set accuracy |
| **Val Accuracy** | 77.42% | Best validation set accuracy |
| F1 Macro | 65.34% | Macro average F1 |
| F1 Weighted | 75.81% | Weighted average F1 |
| **Top-3 Accuracy** | **92.81%** | Top-3 prediction accuracy |
| **Top-5 Accuracy** | **95.42%** | Top-5 prediction accuracy |
| **Top-10 Accuracy** | **97.50%** | Top-10 prediction accuracy |
| Prediction Variance | 0.0026 | Prediction variance |
| Prediction Entropy | 1.287 | Prediction entropy |

### 4.2 Training Curves

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 10 | 2.0212 | 64.58% | 1.9411 | 67.27% |
| 20 | 1.6333 | 76.41% | 1.7045 | 75.15% |
| 30 | 1.4692 | 81.67% | 1.6530 | 77.42% |

### 4.3 Feature Importance (Top 20)

| Rank | Feature | Importance |
|------|------|--------|
| 1 | osda_total_volume | 0.1387 |
| 2 | osda1_index | 0.1318 |
| 3 | cryst_temp | 0.1094 |
| 4 | Si | 0.0986 |
| 5 | Al | 0.0938 |
| 6 | Na | 0.0723 |
| 7 | osda1_asphericity_mean_0 | 0.0605 |
| 8 | osda1_axes_mean_0 | 0.0586 |
| 9 | osda1_box_mean_0 | 0.0586 |
| 10 | rotation | 0.0527 |
| 11 | Ge | 0.0518 |
| 12 | P | 0.0488 |
| 13 | K | 0.0469 |
| 14 | osda1_getaway_mean_0 | 0.0449 |
| 15 | cryst_time | 0.0439 |
| 16 | osda1_bertz_ct_mean_0 | 0.0430 |
| 17 | osda1_box_mean_1 | 0.0420 |
| 18 | OH | 0.0381 |
| 19 | osda1_getaway_mean_1 | 0.0381 |
| 20 | F | 0.0371 |

### 4.4 Attention Weight Distribution

| Position | Feature Group | Average Attention Weight |
|------|--------|----------------|
| 1 | Elements (Si-Al) | See attention_weights.csv |
| 2 | Elements (P-Na) | See attention_weights.csv |
| ... | ... | ... |
| 24 | OSDA_Indices | Important position |
| 25 | Synthesis_Conditions | Important position |
| 26 | Aging_PH_Ratios | See attention_weights.csv |

### 4.5 Classification Performance (Top 10 Frameworks)

| Framework | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| MFI | - | - | - | High |
| CHA | - | - | - | High |
| *BEA | 0.7026 | 0.7697 | 0.7346 | 178 |
| AFI | 0.7813 | 0.9058 | 0.8389 | 138 |
| AEL | 0.8269 | 0.8269 | 0.8269 | 52 |
| FAU | - | - | - | Medium |
| LTA | - | - | - | Medium |
| MOR | - | - | - | Medium |

---

## 5. Visualization Chart Description

### 5.1 Generated Chart Files

| Filename | Description |
|--------|------|
| `training_curves.png` | Training/validation loss and accuracy curves |
| `confusion_matrix.png` | Normalized confusion matrix for Top 30 classes |
| `performance_by_class.png` | Precision/Recall/F1/Support for each framework |
| `overall_performance.png` | Overall performance metrics bar chart |
| `prediction_confidence.png` | Prediction confidence and entropy distribution |
| `feature_importance.png` | Top 20 feature importance |
| `attention_weights.png` | Attention weight distribution |

### 5.2 Generated Data Table Files

| Filename | Description |
|--------|------|
| `training_curves.csv` | Training metrics for each epoch |
| `confusion_matrix.csv` | Top 30 classes confusion matrix data |
| `performance_by_class.csv` | Classification performance for each framework |
| `overall_performance.csv` | Overall performance metrics |
| `prediction_confidence.csv` | Sample prediction confidence |
| `feature_importance.csv` | All 104 feature importance |
| `attention_weights.csv` | Attention weights for 26 positions |

---

## 6. Results Analysis and Discussion

### 6.1 Performance Analysis

1. **Good Overall Performance**: Test accuracy 76.37%, close to best validation performance 77.42%, indicating good model generalization ability

2. **Excellent Top-K Performance**: 
   - Top-5 accuracy 95.42% means the true framework has a 95% probability of appearing in the top 5 predictions
   - Top-10 accuracy 97.50% further improves

3. **Comparison with LeNet-5**:
   | Model | Accuracy | Top-5 Acc | Top-10 Acc |
   |------|----------|-----------|------------|
   | LeNet-5 | 73.80% | 91.22% | - |
   | **BiGRU** | **76.37%** | **95.42%** | **97.50%** |
   | Improvement | +2.57% | +4.20% | - |

4. **BiGRU Advantages Are Significant**: Compared to LeNet-5, BiGRU improves on all metrics

### 6.2 Feature Importance Analysis

**Most Important Feature Categories**:

1. **OSDA-related Features** (highest proportion):
   - `osda_total_volume`: Total OSDA volume, determines whether it can enter zeolite pores
   - `osda1_index`: Main OSDA type index
   - OSDA descriptors (asphericity, axes, box, etc.): Molecular shape information

2. **Synthesis Conditions**:
   - `cryst_temp`: Crystallization temperature, affects product type
   - `cryst_time`: Crystallization time
   - `rotation`: Rotation conditions

3. **Element Composition**:
   - `Si`, `Al`: Determine Si/Al ratio
   - `Na`, `P`, `K`, `Ge`: Influence of other elements

**Physical Significance**:
- OSDA is the key structure-directing agent in zeolite synthesis, its shape and size directly determine the product framework
- Crystallization temperature and time are key parameters controlling kinetic conditions
- Si/Al ratio determines the chemical composition range of the product

### 6.3 Attention Mechanism Analysis

The attention mechanism learned the importance of feature positions:
- OSDA indices and synthesis condition positions receive higher attention weights
- Element composition has differentiated attention at different positions
- This aligns with domain knowledge: OSDA and synthesis conditions are key to prediction

### 6.4 Model Advantages and Limitations

**Advantages**:
1. Sequence modeling capability: Converts tabular features to sequences, captures dependencies between features
2. Bidirectional information flow: Utilizes both forward and backward context simultaneously
3. Attention mechanism: Strong interpretability, visualizes feature importance
4. Superior performance: Accuracy and Top-K metrics are better than CNN

**Limitations**:
1. Feature grouping method: Simply grouping every 4 features may not be optimal
2. Computational efficiency: BiGRU has larger computational cost than CNN
3. Class imbalance: Still challenging for low-frequency framework prediction

### 6.5 Improvement Suggestions

1. **Feature Grouping Optimization**:
   - Group by feature semantics (elements, OSDA, conditions)
   - Use learnable feature embeddings

2. **Model Improvements**:
   - Add Transformer Encoder layers
   - Use multi-scale feature fusion
   - Introduce pre-training mechanism

3. **Class Imbalance Handling**:
   - Use Focal Loss
   - Data augmentation
   - Class weight adjustment

---

## 7. Conclusion

This experiment successfully implemented a BiGRU-based zeolite framework type prediction model:

1. **Test Accuracy**: 76.37%, better than LeNet-5's 73.80%
2. **Top-5 Accuracy**: 95.42%, has important value for practical applications
3. **Top-10 Accuracy**: 97.50%, suitable for use as a recommendation system
4. **Interpretability**: Attention mechanism and feature importance analysis provide physical significance

The BiGRU model effectively captures dependencies between features through sequence modeling and attention mechanism, demonstrating superior performance on the zeolite framework prediction task.

---

## Appendix

### A. File List

| File | Size | Description |
|------|------|------|
| `bigru_104.py` | ~30KB | Training script |
| `bigru_104.pkl` | ~3MB | Trained model |
| `bigru_104_results.json` | ~1KB | Results JSON |
| `training_curves.png/csv` | - | Training curves |
| `confusion_matrix.png/csv` | - | Confusion matrix |
| `performance_by_class.png/csv` | - | Classification performance |
| `overall_performance.png/csv` | - | Overall performance |
| `prediction_confidence.png/csv` | - | Prediction confidence |
| `feature_importance.png/csv` | - | Feature importance |
| `attention_weights.png/csv` | - | Attention weights |

### B. Running Commands

```bash
cd BiGRU
python bigru_104.py
```

### C. Dependency Environment

```
Python 3.14.2
PyTorch 2.10.0
scikit-learn 1.6.0
NumPy 2.2.0
Pandas 2.2.0
Matplotlib 3.10.0
Seaborn 0.13.0
```

---

*Report generation time: March 20, 2026*
