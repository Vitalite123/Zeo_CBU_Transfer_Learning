# CBUTransfer - Zeolite Framework Prediction via Transfer Learning

A research project for predicting zeolite framework structures (Code1) from synthesis conditions using machine learning and deep learning models, with a focus on transfer learning approaches based on Composite Building Units (CBU).

## Project Overview

This repository contains multiple model implementations for zeolite framework prediction:

| Model | Type | Features | Description |
|-------|------|----------|-------------|
| **Baseline61** | Tree-based (XGBoost) | 61 | Baseline with elemental composition + synthesis conditions |
| **Baseline104** | Tree-based (XGBoost/RF/LGBM) | 104 | Extended with OSDA molecular descriptors |
| **LeNet5** | 1D CNN | 104 | Convolutional neural network adapted for tabular data |
| **BiGRU** | Bidirectional RNN | 104 | GRU with attention mechanism |
| **Transformer** | Transformer Encoder | 104 | Self-attention for feature interactions |
| **CBU_Node_TL** | Transfer Learning | 104 | Node-based CBU grouping for progressive pretraining |
| **ExtCBU_TL** | Transfer Learning | 104 | Extended CBU transfer learning with global similarity |

## Dataset

Located in `Dataset/`:
- `Dataset_full.csv` - Complete dataset
- `Dataset_train.xlsx`, `Dataset_val.xlsx`, `Dataset_test.xlsx` - Pre-split data
- `Feature_List.md` - Feature documentation

### Feature Categories (104 features)

| Category | Count | Examples |
|----------|-------|----------|
| Elemental Composition | 45 | Si, Al, P, Na, K, Li, Sr, Rb... |
| OSDA Indices | 3 | osda1_index, osda2_index, osda3_index |
| Synthesis Conditions | 4 | cryst_temp, cryst_time, seed, rotation |
| Aging Conditions | 2 | aging_temp, aging_time |
| pH Conditions | 2 | acid, OH |
| Gel Ratios | 5 | H2O_T, OH_T, Gel_Si_Al, Gel_P_Al, Gel_P_Si |
| OSDA Molecular Descriptors | 33 | bertz_ct, free_sasa, asphericity, eccentricity... |
| Aggregated Features | 10 | osda_avg_asphericity, osda_max_sasa, osda_total_volume... |

## Project Structure

```
CBUTransfer/
├── Baseline61/              # 61-feature baseline model
│   ├── Script/
│   │   └── Baseline61.py
│   └── *.csv, *.pkl
├── Baseline104/             # 104-feature baseline models
│   ├── Script/
│   │   └── Baseline104.py   # XGBoost + RF + LightGBM ensemble
│   └── *.csv, *.pkl
├── LeNet5/                  # 1D CNN model
│   ├── Script/
│   │   └── lenet5_104.py
│   └── *.csv, *.pkl
├── BiGRU/                   # BiGRU with attention
│   ├── Script/
│   │   └── bigru_104.py
│   └── *.csv, *.pkl
├── Transformer/             # Transformer encoder
│   ├── Script/
│   │   └── transformer_104.py
│   └── *.csv, *.pkl
├── CBU_Node_TL/            # CBU Node-based Transfer Learning
│   ├── Script/
│   │   ├── stage1_cbu_grouping.py      # CBU grouping by Node value
│   │   ├── stage2_baseline_training.py  # Baseline model
│   │   ├── stage3_progressive_pretrain.py
│   │   ├── stage4_binary_finetune.py
│   │   ├── stage5_three_layer_ensemble.py
│   │   ├── stage6_comprehensive_evaluation.py
│   │   └── stage7_results_analysis.py
│   └── *.csv, *.pkl, *.md
├── ExtCBU_TL/              # Extended CBU Transfer Learning
│   ├── Script/
│   │   ├── stage1_cbu_grouping.py
│   │   ├── stage2_baseline_training.py
│   │   ├── stage3_pretrain_grouped.py
│   │   ├── stage4_binary_grouped.py
│   │   └── stage5_three_layer_ensemble.py
│   └── *.csv, *.pkl, *.md
└── Dataset/                 # Data files
    ├── Dataset_full.csv
    ├── Dataset_train.xlsx
    ├── Dataset_val.xlsx
    ├── Dataset_test.xlsx
    └── Feature_List.md
```

## Model Details

### Baseline Models

#### Baseline61
- **Model**: XGBoost Classifier
- **Features**: 61 (elemental composition + synthesis conditions)
- **Parameters**: n_estimators=300, max_depth=10, learning_rate=0.05

#### Baseline104
- **Models**: XGBoost, RandomForest, LightGBM ensemble
- **Features**: 104 (added OSDA molecular descriptors)
- **XGBoost**: n_estimators=300, max_depth=10, learning_rate=0.05
- **RandomForest**: n_estimators=400, max_depth=25
- **LightGBM**: n_estimators=800, max_depth=15, learning_rate=0.05

### Deep Learning Models

#### LeNet5 (1D)
- **Architecture**: Conv1D → Conv1D → Conv1D → FC → FC → FC
- **Layers**: 64→128→256 filters
- **Input**: 104 features (reshaped to 1D signal)

#### BiGRU
- **Architecture**: Feature Embedding → BiGRU (2 layers) → Attention → FC → FC → FC
- **Hidden Size**: 128
- **Features**: 26 timesteps × 4 features per step
- **Attention**: Weighted sequence aggregation

#### Transformer
- **Architecture**: Feature Tokenization → Embedding → Positional Encoding → Transformer Encoder → Pooling → FC → FC → FC
- **d_model**: 64
- **Attention Heads**: 4
- **Layers**: 3
- **Sequence**: 13 tokens × 8 features per token

### Transfer Learning (CBU_Node_TL / ExtCBU_TL)

Multi-stage pipeline for handling data imbalance via CBU-based transfer learning:

1. **Stage 1**: CBU grouping based on Node values or global similarity
2. **Stage 2**: Baseline XGBoost model training
3. **Stage 3**: Progressive pretraining on large CBU groups
4. **Stage 4**: Binary fine-tuning for small/unseen CBUs
5. **Stage 5**: Three-layer ensemble combining predictions
6. **Stage 6-7**: Comprehensive evaluation and analysis

## Installation

```bash
# Core dependencies
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install torch torchvision
pip install matplotlib seaborn
pip install openpyxl joblib
```

## Usage

### Baseline104 Model
```bash
cd Baseline104/Script
python Baseline104.py
```

### Deep Learning Models
```bash
cd LeNet5/Script
python lenet5_104.py

cd BiGRU/Script
python bigru_104.py

cd Transformer/Script
python transformer_104.py
```

### Transfer Learning Pipeline
```bash
cd CBU_Node_TL/Script
python stage1_cbu_grouping.py
python stage2_baseline_training.py
python stage3_progressive_pretrain.py
python stage4_binary_finetune.py
python stage5_three_layer_ensemble.py
```

## Output Files

Each model generates:
- `*.pkl` - Trained model and metadata
- `*.csv` - Performance metrics and predictions
- `*.json` - Results in JSON format
- `*.png` - Visualization plots
- `*.md` - Detailed analysis reports

## CBU Grouping Strategy

### Node-Based Grouping (CBU_Node_TL)
- Groups CBUs by their Node value (5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 30, 32, 36, 48)
- 18 natural groups based on structural similarity

### Global Similarity (ExtCBU_TL)
- Uses full CBU similarity matrix
- 4-tier grouping based on similarity percentiles:
  - G1_VeryHigh: Q75+
  - G2_High: Q50-Q75
  - G3_Medium: Q25-Q50
  - G4_Low: <Q25

## Model Comparison

| Model | Architecture | Typical Accuracy |
|-------|--------------|------------------|
| Baseline61 | XGBoost | ~80.5% |
| Baseline104 | XGBoost | ~83.2% |
| LeNet5 | 1D CNN | ~77.4% |
| BiGRU | BiGRU + Attention | ~78.5% |
| Transformer | Transformer Encoder | ~78-82% |

## Key Features

- **Multi-stage Transfer Learning**: Progressive pretraining strategy for handling class imbalance
- **CBU-based Grouping**: Leverages zeolite structural similarity for knowledge transfer
- **Comprehensive Evaluation**: Tier-based performance analysis by sample size
- **Feature Importance**: Permutation importance for model interpretability
- **Attention Analysis**: Visualizes feature interactions in deep learning models

## Research Focus

This project investigates:
1. How to improve prediction for rare zeolite frameworks with limited samples
2. Transfer learning strategies based on structural similarity (CBU)
3. Comparison of traditional ML vs deep learning for tabular data
4. Feature importance in zeolite synthesis prediction

## License

Research project for zeolite framework prediction.
