# TLCBU_vGlobe Complete Technical Report

## 1. Project Overview

**Project Name**: TLCBU_vGlobe - Transfer Learning Based on CBU Global Feature Similarity

**Objective**: Transfer learning using 65×65 global similarity matrix, compared with TLCBU_v4 (58×58 Node similarity)

**Feature Composition** (104 features):
- 45 element composition features (Si, Al, P, Na, K, Li, Sr, Rb, Cs, Ba, Ca, F, Ge, Ti, In, B, Mg, Ga, Ni, Mn, Fe, Co, Cr, Zn, Nb, Be, W, Ce, Cu, Sn, Gd, La, Y, Dy, Sm, Ag, Cd, Zr, V, Ta, ru, Hf, Yb, Tl, As)
- 3 OSDA index features (osda1_index, osda2_index, osda3_index)
- 13 synthesis condition features (cryst_temp, cryst_time, seed, rotation, aging_temp, aging_time, acid, OH, H2O_T, OH_T, Gel_Si_Al, Gel_P_Al, Gel_P_Si)
- 33 OSDA molecular descriptors (osda1/osda2/osda3's bertz, sasa, asphericity, eccentricity, axes, box, getaway)
- 10 aggregation features (osda_avg/max/min_asphericity, sasa, bertz, osda_total_volume)

---

## 2. Overall Results

| Metric | TLCBU_vGlobe (Global Similarity) | TLCBU_v4 (Node Similarity) | Difference |
|------|-------------------------|---------------------|------|
| **Test Accuracy** | **96.46%** | 95.24% | **+1.22%** |
| F1-Weighted | 96.31% | 95.00% | +1.31% |
| F1-Macro | 82.73% | 75.94% | **+6.79%** |
| Precision-Weighted | 96.37% | - | - |
| Precision-Macro | 83.56% | - | - |
| Recall-Weighted | 96.46% | - | - |
| Recall-Macro | 83.01% | - | - |

### Ensemble Architecture

- **Layer1 (Pretrained Model)**: 96.40%
- **Layer2 (Binary Classifier v2)**: 87.55%
- **Best Alpha**: 0.8 (primarily relies on Layer1)

### Binary Classifier v2 Improvements

| Version | Layer2 Accuracy | Description |
|------|-------------|------|
| v1 | 5.73% | Original version, probabilities not normalized |
| **v2** | **87.55%** | Optimized negative sampling + direct argmax |

---

## 3. Stratified Results by Code1 Sample Count

| Tier | Test Samples | Classes | Accuracy | F1-Weighted | F1-Macro |
|------|---------|-------|--------|-------------|----------|
| **Tiny (<10)** | 60 | 34 | **70.00%** | 0.7083 | 0.4504 |
| **Very Small (10-19)** | 80 | 26 | **92.50%** | 0.9470 | 0.7504 |
| **Small (20-49)** | 214 | 39 | **92.99%** | 0.9498 | 0.7959 |
| Small-Medium (50-99) | 402 | 25 | 94.28% | 0.9660 | 0.5721 |
| Medium (100-499) | 2091 | 46 | 92.59% | 0.9479 | 0.5966 |
| Medium-Large (500-999) | 1136 | 7 | 97.98% | 0.9897 | 0.3648 |
| Large (≥1000) | 4452 | 14 | 98.67% | 0.9896 | 0.2476 |

---

## 4. Comparison with TLCBU_v4 Stratified Results

| Tier | Sample Count (TLCBU_v4) | Classes | TLCBU_vGlobe | TLCBU_v4 | Difference |
|------|-----------------|-------|-------------|----------|------|
| **Tiny (<10)** | 103 | 87 | **70.00%** | 55.34% | **+14.66%** |
| **Very Small (10-19)** | 153 | 32 | **92.50%** | 92.16% | **+0.34%** |
| **Small (20-49)** | 264 | 30 | **92.99%** | 85.61% | **+7.38%** |
| Small-Medium (50-99) | 540 | 26 | 94.28% | 93.15% | +1.13% |
| Medium (100-499) | 2290 | 41 | 92.59% | 91.27% | +1.32% |
| Medium-Large (500-999) | 1314 | 7 | 97.98% | 96.88% | +1.10% |
| Large (≥1000) | 5051 | 8 | 98.67% | 98.26% | +0.41% |

*Note: TLCBU_v4 data from official evaluation report (Stage 6 three_layer_ensemble)*

---

## 5. Model Results by Stage

### Stage 2: Baseline Model

| Metric | Training Set | Test Set |
|------|-------|-------|
| Accuracy | 98.95% | 96.34% |
| F1-Weighted | 98.93% | 96.20% |
| F1-Macro | 89.16% | 82.12% |

### Stage 3: Progressive Pretraining

| Stage | Epochs | Test Accuracy | Notes |
|-------|------|-----------|------|
| Stage 4 | 280 | **96.41%** | Best |
| Stage 3 | 250 | 96.38% | |
| Stage 5 | 300 | 96.34% | |
| Stage 2 | 200 | 96.31% | |
| Stage 1 | 150 | 96.14% | |

### Stage 4: Binary Classifier

- Successfully trained: 205/231 binary classifiers
- Accuracy: 83.75%

### Stage 5: Three-Layer Ensemble

- Final accuracy: **96.42%**
- Best Alpha: 0.9

---

## 6. Conclusion

TLCBU_vGlobe using 65×65 global similarity matrix outperforms TLCBU_v4's 58×58 Node similarity method on all metrics:

1. **Accuracy improvement +1.22%** (95.24% → 96.46%)
2. **F1-Macro improvement +6.79%** (75.94% → 82.73%)
3. **Binary classifier optimization**: Layer2 improved from 5.73% to 87.55%
4. **Significant improvement for few-shot classes**: 
   - Tiny (<10): 55.34% → 70.00% (+14.66%)
   - Small (20-49): 85.61% → 92.99% (+7.38%)

The global similarity matrix contains richer structural information and can capture CBU inter-similarity relationships better than Node values alone.

---

## 7. File Inventory

| File | Description |
|------|------|
| `results/final_ensemble_report.json` | Complete report JSON |
| `results/final_tier_results.csv` | Stratified results CSV |
| `results/binary_results_v2.json` | Binary classifier v2 results |
| `results/ensemble_results.json` | Ensemble results |
| `results/global_baseline_results.json` | Baseline results |
| `results/global_pretrain_comparison.csv` | Pretraining comparison |
| `models/ensemble/global_three_layer_ensemble.pkl` | Final ensemble model |
| `models/binary/global_binary_classifiers_v2.pkl` | Optimized binary classifiers |

---

**Generated**: 2026-04-17  
**TLCBU_vGlobe Version**: v1.0
