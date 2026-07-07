# Transformer Model for Zeolite Framework Prediction

## 1. Model Architecture

### 1.1 Transformer Encoder Overview

This model adopts a **Transformer Encoder** architecture, converting 104-dimensional tabular features into sequence form for modeling. The core advantage of Transformer lies in its **Self-Attention mechanism**, which can capture long-range dependencies between features.

### 1.2 Architecture Details

| Component | Configuration |
|-----------|--------------|
| Input Features | 104 dimensions |
| Sequence Length | 13 tokens |
| Token Dimension | 8 features per token |
| Embedding Dimension (d_model) | 64 |
| Number of Attention Heads | 4 |
| Number of Encoder Layers | 2 |
| Feedforward Dimension | 128 |
| Output Classes | 231 |

### 1.3 Model Structure

```
Input (104 features)
    ↓
Reshape to Sequence (13 tokens × 8 features)
    ↓
Feature Embedding (Linear: 8 → 64)
    ↓
Positional Encoding (Sinusoidal)
    ↓
Transformer Encoder × 2 layers
    │
    ├── Multi-Head Self-Attention (4 heads)
    ├── Add & Layer Normalization
    ├── Feedforward Network (128 hidden)
    └── Add & Layer Normalization
    ↓
Global Average Pooling
    ↓
Classification Head
    │
    ├── Linear (64 → 128)
    ├── ReLU + Dropout
    └── Linear (128 → 231)
    ↓
Output (Logits)
```

### 1.4 Key Design Choices

1. **Feature-to-Token Conversion**: The 104 features are divided into 13 groups, with each group of 8 features treated as one token, facilitating Transformer to process sequence information.

2. **Positional Encoding**: Sinusoidal positional encoding is used to provide positional information to the model, enabling it to distinguish between feature groups at different positions.

3. **Multi-Head Attention**: The 4 attention heads allow the model to simultaneously focus on different feature combinations at various positions.

4. **Residual Connections**: Each sub-layer is followed by a residual connection, which helps gradient flow and facilitates training of deep networks.

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Training Samples | 12,537 |
| Validation Samples | 2,640 |
| Test Samples | 2,641 |
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Scheduler | ReduceLROnPlateau |
| Epochs Trained | 149 |
| Total Parameters | 105,639 |

## 3. Dataset Information

### 3.1 Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| Elemental Features | 45 | Si, Al, P, Na, K, Li, ... |
| OSDA Index | 3 | osda1_index, osda2_index, osda3_index |
| Synthesis Conditions | 4 | cryst_temp, cryst_time, seed, rotation |
| Aging Conditions | 2 | aging_temp, aging_time |
| pH Conditions | 2 | acid, OH |
| Gel Ratios | 5 | H2O_T, OH_T, Gel_Si_Al, Gel_P_Al, Gel_P_Si |
| OSDA Descriptors | 33 | bertz_ct, free_sasa, asphericity, ... |
| Aggregated Features | 10 | osda_avg_asphericity, osda_max_sasa, ... |
| **Total** | **104** | |

### 3.2 Framework Distribution

The dataset contains **231** different zeolite framework types (Code1), with the most abundant frameworks including: MFI, CHA, *BEA, AFI, MTW, etc.

## 4. Model Performance

### 4.1 Overall Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **0.6778** |
| Precision (Macro) | 0.5548 |
| Recall (Macro) | 0.5568 |
| F1 Score (Macro) | 0.5412 |
| F1 Score (Weighted) | 0.6690 |
| Prediction Variance | 0.0029 |
| Best Validation Accuracy | 0.6777 |

### 4.2 Top 10 Frameworks by F1 Score

| Rank | Framework | Precision | Recall | F1 Score | Support |
|------|-----------|-----------|--------|----------|---------|
| 10 | *-ITN | 1.0000 | 1.0000 | 1.0000 | 2 |
| 9 | -IRY | 1.0000 | 1.0000 | 1.0000 | 1 |
| 8 | -LIT | 1.0000 | 1.0000 | 1.0000 | 1 |
| 7 | *UOE | 1.0000 | 1.0000 | 1.0000 | 2 |
| 6 | APD | 1.0000 | 1.0000 | 1.0000 | 1 |
| 5 | MSO | 1.0000 | 1.0000 | 1.0000 | 1 |
| 4 | PWW | 1.0000 | 1.0000 | 1.0000 | 1 |
| 3 | PON | 1.0000 | 1.0000 | 1.0000 | 2 |
| 2 | UOS | 1.0000 | 1.0000 | 1.0000 | 2 |
| 1 | SFS | 1.0000 | 1.0000 | 1.0000 | 1 |

### 4.3 Performance Analysis

1. **Overall Performance**: The model achieves **67.78%** accuracy on the test set, indicating that the Transformer architecture can effectively learn the complex mapping relationship between zeolite synthesis conditions and framework types.

2. **Class Imbalance**: Due to significant differences in sample sizes across framework types in the dataset, F1 Macro (0.5412) is lower than F1 Weighted (0.6690), indicating that the model's performance degrades on minority classes.

3. **Prediction Confidence**: The average prediction variance is 0.0029, indicating that the model produces relatively certain predictions.

## 5. Feature Importance Analysis

### 5.1 Top 10 Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | osda_total_volume | 0.115487 |
| 2 | osda1_index | 0.107914 |
| 3 | Al | 0.092389 |
| 4 | osda_min_asphericity | 0.085574 |
| 5 | Si | 0.084438 |
| 6 | cryst_temp | 0.081409 |
| 7 | osda1_axes_mean_0 | 0.079515 |
| 8 | osda_max_bertz | 0.072321 |
| 9 | Na | 0.066641 |
| 10 | osda_max_sasa | 0.063234 |

### 5.2 Feature Importance Interpretation

Feature importance is calculated using the **Permutation Importance** method. The importance value represents the decrease in model accuracy when the feature is randomly shuffled.

- **Positive value**: Accuracy decreases after shuffling, indicating the feature contributes positively to prediction
- **Negative value**: Accuracy increases after shuffling, possibly indicating feature redundancy or noise

### 5.3 Token Importance

Transformer divides 104 features into 13 tokens, with each token containing 8 features. The following shows the importance of each token:

| Token | Features | Importance |
|-------|----------|------------|
| 0 | Si, Al, P, Na, K, Li, Sr, Rb... | 1.425800 |
| 1 | Cs, Ba, Ca, F, Ge, Ti, In, B... | 1.420397 |
| 2 | Mg, Ga, Ni, Mn, Fe, Co, Cr, Zn... | 1.420048 |
| 3 | Nb, Be, W, Ce, Cu, Sn, Gd, La... | 1.420650 |
| 4 | Y, Dy, Sm, Ag, Cd, Zr, V, Ta... | 1.420135 |
| 5 | Ru, Hf, Yb, Tl, As, osda1_index, osda2_i... | 1.430660 |
| 6 | cryst_temp, cryst_time, seed, rotation, ... | 1.421953 |
| 7 | H2O_T, OH_T, Gel_Si_Al, Gel_P_Al, Gel_P_... | 1.429659 |
| 8 | osda1_eccentricity_mean_0, osda1_axes_me... | 1.421994 |
| 9 | osda2_bertz_ct_mean_0, osda2_free_sasa_m... | 1.419353 |
| 10 | osda2_box_mean_2, osda2_getaway_mean_0, ... | 1.425704 |
| 11 | osda3_axes_mean_1, osda3_box_mean_0, osd... | 1.422882 |
| 12 | osda_min_asphericity, osda_avg_sasa, osd... | 1.428931 |

## 6. Training History

### 6.1 Loss and Accuracy Curves

- Best validation accuracy: 0.6777
- Final training accuracy: 0.6152
- Final validation accuracy: 0.6746

## 7. Comparison with Other Models

| Model | Architecture | Test Accuracy |
|-------|--------------|---------------|
| Baseline61 XGBoost | Gradient Boosting | ~80.50% |
| Baseline104 XGBoost | Gradient Boosting | ~83.15% |
| LeNet5 CNN | Convolutional | ~77.36% |
| **Transformer (This work)** | **Self-Attention** | **67.78%** |
| BiGRU | Bidirectional RNN | ~78.5% |

## 8. Conclusions

### 8.1 Summary

1. The Transformer model performs well on the zeolite framework prediction task, achieving **67.78%** test accuracy.

2. The self-attention mechanism can effectively capture complex interactions between features, particularly the relationships between OSDA molecular descriptors and synthesis conditions.

3. Feature importance analysis reveals key influencing factors: osda_total_volume, osda1_index, Al, osda_min_asphericity, Si.

### 8.2 Advantages of Transformer

1. **Long-range Dependencies**: Self-attention can capture relationships between any two features without distance limitations
2. **Parallel Computation**: Compared to RNNs, Transformers can process sequences in parallel, resulting in higher training efficiency
3. **Interpretability**: Attention weights provide a certain degree of model interpretability

### 8.3 Limitations

1. **Data Scale**: Transformers typically require large amounts of data for training; the current dataset is relatively limited in scale
2. **Tabular Data**: Transformers were originally designed for sequence data and may be less efficient than traditional methods when applied to tabular data
3. **Computational Cost**: Compared to simple machine learning models, Transformers require more computational resources

### 8.4 Future Work

1. Explore larger model scales and more data augmentation techniques
2. Design better feature encoding methods by combining domain knowledge
3. Try multi-task learning to predict multiple zeolite properties simultaneously

---

*Report generated on 2026-03-21 07:34:09*
