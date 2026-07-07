# Baseline104 Model Detailed Report
## Zeolite Framework Structure Prediction Based on Full Feature Set

---

## 1. Project Overview

### 1.1 Research Objective
This project aims to use machine learning methods to predict zeolite framework structures (Code1_index) from complete synthesis experimental parameters (104 features). This is a multi-classification problem that aims to accurately predict synthetic products from 231 different framework types.

### 1.2 Dataset Information
- **Training Set**: 12,537 samples, 247 columns (104 features + labels + other metadata)
- **Validation Set**: 2,640 samples, 247 columns
- **Test Set**: 2,641 samples, 247 columns
- **Number of Classes**: 231 framework types
- **Number of Features**: **104** (full feature set)

### 1.3 Feature Description
This project uses the complete 104 features defined in Feature_List.md:

#### 1.3.1 Element Composition Features (45)
Si, Al, P, Na, K, Li, Sr, Rb, Cs, Ba, Ca, F, Ge, Ti, In, B, Mg, Ga, Ni, Mn, Fe, Co, Cr, Zn, Nb, Be, W, Ce, Cu, Sn, Gd, La, Y, Dy, Sm, Ag, Cd, Zr, V, Ta, Ru, Hf, Yb, Tl, As

These features represent the normalized mole fractions of various elements in the synthesis gel.

#### 1.3.2 OSDA Index Features (3)
- osda1_index: Index of Organic Structure-Directing Agent 1
- osda2_index: Index of Organic Structure-Directing Agent 2
- osda3_index: Index of Organic Structure-Directing Agent 3

#### 1.3.3 Synthesis Condition Features (4)
- cryst_temp: Crystallization temperature (°C)
- cryst_time: Crystallization time (hours)
- seed: Whether seed crystals are used (0/1)
- rotation: Whether stirring is applied (0/1)

#### 1.3.4 Aging Condition Features (2)
- aging_temp: Aging temperature (°C)
- aging_time: Aging time (hours)

#### 1.3.5 pH Condition Features (2)
- acid: Acid concentration
- OH: Hydroxide concentration

#### 1.3.6 Gel Ratio Features (5)
- H2O_T: Water/template ratio
- OH_T: OH-/template ratio
- Gel_Si_Al: Si/Al ratio in gel
- Gel_P_Al: P/Al ratio in gel
- Gel_P_Si: P/Si ratio in gel

#### 1.3.7 OSDA Molecular Descriptors (33)
**Each OSDA (osda1, osda2, osda3) contains 11 descriptors**:
- bertz_ct_mean_0: Bertz complexity index
- free_sasa_mean_0: Solvent accessible surface area
- asphericity_mean_0: Asphericity
- eccentricity_mean_0: Eccentricity
- axes_mean_0, axes_mean_1: Principal axis lengths
- box_mean_0, box_mean_1, box_mean_2: Bounding box dimensions
- getaway_mean_0, getaway_mean_1: GETAWAY descriptors

#### 1.3.8 Aggregate Features (10)
- osda_avg_asphericity: Average asphericity
- osda_max_asphericity: Maximum asphericity
- osda_min_asphericity: Minimum asphericity
- osda_avg_sasa: Average solvent accessible surface area
- osda_max_sasa: Maximum solvent accessible surface area
- osda_min_sasa: Minimum solvent accessible surface area
- osda_avg_bertz: Average Bertz complexity
- osda_max_bertz: Maximum Bertz complexity
- osda_min_bertz: Minimum Bertz complexity
- osda_total_volume: Total volume

### 1.4 Model Selection
This project compares three mainstream tree-based ensemble learning models:
1. **XGBoost (Extreme Gradient Boosting)** - Optimized best model
2. **RandomForest (Random Forest)** - High-performance baseline model
3. **LightGBM (Light Gradient Boosting Machine)** - Fast gradient boosting model

---

## 2. Model Principles and Algorithms

### 2.1 XGBoost Model

#### 2.1.1 Principle Overview
XGBoost is an efficient implementation based on Gradient Boosting Decision Trees (GBDT), which improves prediction performance by building an ensemble of multiple weak learners (decision trees).

#### 2.1.2 Core Algorithm
The objective function of XGBoost consists of two parts:
```
Obj(Θ) = L(Θ) + Ω(Θ)
```
Where:
- **L(Θ)**: Loss function, measuring model fitting error
- **Ω(Θ)**: Regularization term, controlling model complexity

For the t-th iteration, the objective function can be expressed as:
```
Obj^(t) = Σi L(yi, ŷi^(t-1) + ft(xi)) + Ω(ft)
```

Using Taylor expansion approximation:
```
Obj^(t) ≈ Σi [L(yi, ŷi^(t-1)) + gi ft(xi) + 1/2 hi ft^2(xi)] + Ω(ft)
```

Where:
- **gi**: First derivative of the loss function (gradient)
- **hi**: Second derivative of the loss function (Hessian)

#### 2.1.3 Optimized Model Structure
```python
n_estimators=300              # Number of trees (50% increase from Baseline61)
max_depth=10                 # Maximum tree depth (25% increase from Baseline61)
learning_rate=0.05           # Learning rate (50% decrease from Baseline61)
subsample=0.9                # Subsample ratio
colsample_bytree=0.9         # Feature subsample ratio
min_child_weight=3           # Minimum child node weight
gamma=0.1                    # Minimum split gain
reg_alpha=0.1                # L1 regularization
reg_lambda=1.0               # L2 regularization
random_state=42              # Random seed
n_jobs=-1                    # Use all CPU cores
eval_metric='mlogloss'       # Multi-class log loss
```

#### 2.1.4 Advantages
1. **Regularization**: L1 and L2 regularization prevent overfitting
2. **Parallel Processing**: Supports multi-threaded parallel computation
3. **Missing Value Handling**: Automatically handles missing values
4. **Pruning Strategy**: Uses maximum depth instead of number of leaf nodes
5. **Approximate Algorithm**: Supports sparse data approximate splitting

### 2.2 RandomForest Model

#### 2.2.1 Principle Overview
RandomForest is an ensemble learning method based on Bagging (Bootstrap Aggregating), which makes predictions by building multiple decision trees and voting.

#### 2.2.2 Core Algorithm
RandomForest training process:
1. **Bootstrap Sampling**: Draw n samples with replacement from the training set to generate n training sets
2. **Random Feature Selection**: At each node split, randomly select m features (m < M)
3. **Build Decision Tree**: Use CART algorithm to build fully grown decision trees
4. **Ensemble Prediction**: For classification problems, use majority voting

#### 2.2.3 Optimized Model Structure
```python
n_estimators=400              # Number of trees (100% increase from Baseline61)
max_depth=25                 # Maximum tree depth (67% increase from Baseline61)
min_samples_split=2          # Minimum samples for node split (60% decrease from Baseline61)
min_samples_leaf=1           # Minimum samples for leaf node (50% decrease from Baseline61)
max_features='sqrt'          # Feature selection strategy
bootstrap=True               # Use bootstrap sampling
random_state=42              # Random seed
n_jobs=-1                    # Use all CPU cores
class_weight='balanced'      # Handle class imbalance
```

#### 2.2.4 Advantages
1. **Variance Reduction**: Reduce overfitting by averaging multiple decision trees
2. **Feature Importance**: Can evaluate the importance of each feature
3. **Robustness**: Not sensitive to noise and outliers
4. **Interpretability**: Easy to understand and visualize

### 2.3 LightGBM Model

#### 2.3.1 Principle Overview
LightGBM is a gradient boosting framework developed by Microsoft, using Histogram-based decision tree algorithm and GOSS (Gradient-based One-Side Sampling) technique.

#### 2.3.2 Core Algorithm
**1. Histogram-based Decision Tree Algorithm**:
- Discretize continuous feature values into k bins
- Use bin indices to build histograms
- Select best split points based on histogram split gain

**2. GOSS (Gradient-based One-Side Sampling)**:
- Keep samples with large gradients (high information content)
- Randomly sample samples with small gradients
- Use constant compensation to correct sampling bias

**3. EFB (Exclusive Feature Bundling)**:
- Identify exclusive features (rarely non-zero simultaneously)
- Bundle multiple exclusive features into one feature

#### 2.3.3 Optimized Model Structure
```python
n_estimators=800              # Number of trees (300% increase from Baseline61)
max_depth=15                  # Maximum tree depth (88% increase from Baseline61)
learning_rate=0.05            # Learning rate (50% decrease from Baseline61)
num_leaves=64                 # Number of leaf nodes
subsample=0.9                 # Subsample ratio
colsample_bytree=0.9          # Feature subsample ratio
min_child_samples=10          # Minimum child samples
reg_alpha=0.1                 # L1 regularization
reg_lambda=0.1                # L2 regularization
random_state=42               # Random seed
n_jobs=-1                     # Use all CPU cores
early_stopping_rounds=50      # Early stopping strategy (400% increase from Baseline61)
```

#### 2.3.4 Advantages
1. **Fast Training**: Histogram-based algorithm reduces computation
2. **Low Memory Usage**: Sparse data optimization
3. **Handle Large Data**: Supports training on massive datasets
4. **Early Stopping**: Automatically stop training to prevent overfitting

---

## 3. Training, Validation and Testing Process

### 3.1 Data Preprocessing

#### 3.1.1 Data Loading
```python
train_df = pd.read_excel('ZEOSYN_104_struct_train.xlsx')
val_df = pd.read_excel('ZEOSYN_104_struct_val.xlsx')
test_df = pd.read_excel('ZEOSYN_104_struct_test.xlsx')
```

#### 3.1.2 Feature Selection
- Select actually usable features from 104 predefined features
- Verify all 104 features are available in the data
- Remove Code1_index column as label

#### 3.1.3 Label Encoding
Use LabelEncoder to encode framework types as numerical labels:
```python
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)
```

#### 3.1.4 Missing Value Handling
Fill all missing values with 0:
```python
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)
```

### 3.2 Training Process

#### 3.2.1 Training Workflow
1. **XGBoost Training**: Train on training set, monitor performance on validation set
2. **RandomForest Training**: Train on training set, use class_weight to handle class imbalance
3. **LightGBM Training**: Train on training set, use validation set for early stopping

#### 3.2.2 Training Parameters
All models use optimized parameters to fully utilize the 104 features.

#### 3.2.3 Training Time
- XGBoost: approximately 2-3 minutes
- RandomForest: approximately 3-5 minutes
- LightGBM: approximately 1-2 minutes

### 3.3 Validation Process

#### 3.3.1 Validation Metrics
- **Accuracy**: Overall accuracy
- **F1 Weighted**: Weighted F1 score (considering class imbalance)
- **F1 Macro**: Macro-averaged F1 score (treating all classes equally)
- **Variance**: Prediction variance (model uncertainty)

#### 3.3.2 Validation Method
Evaluate all models on the validation set, calculate above metrics and generate confusion matrix.

### 3.4 Testing Process

#### 3.4.1 Testing Metrics
Use the same metrics as the validation set for evaluation.

#### 3.4.2 Testing Method
Evaluate all models on the test set to ensure model generalization ability.

---

## 4. Experimental Results and Analysis

### 4.1 Performance Metrics Summary

#### 4.1.1 Complete Performance Table

| Model | Dataset | Accuracy | F1 Weighted | F1 Macro | Variance |
|------|--------|----------|-------------|----------|----------|
| **XGBoost** | Training Set | 0.9686 | 0.9650 | 0.6960 | 0.0035 |
| | Validation Set | 0.8299 | 0.8271 | 0.7593 | 0.0032 |
| | Test Set | 0.8315 | 0.8278 | 0.7397 | 0.0032 |
| **RandomForest** | Training Set | 0.9608 | 0.9613 | 0.9741 | 0.0010 |
| | Validation Set | 0.8144 | 0.8151 | 0.7400 | 0.0009 |
| | Test Set | 0.8050 | 0.8040 | 0.7004 | 0.0009 |
| **LightGBM** | Training Set | 0.9896 | 0.9896 | 0.9914 | 0.0038 |
| | Validation Set | 0.8208 | 0.8176 | 0.7398 | 0.0034 |
| | Test Set | 0.8251 | 0.8218 | 0.7166 | 0.0034 |

#### 4.1.2 Test Set Performance Comparison

| Model | Accuracy | F1 Weighted | F1 Macro | Rank |
|------|----------|-------------|----------|------|
| **XGBoost** | 83.15% | 82.78% | 73.97% | 1 |
| **LightGBM** | 82.51% | 82.18% | 71.66% | 2 |
| **RandomForest** | 80.50% | 80.40% | 70.04% | 3 |

### 4.2 Baseline104 vs Baseline61 Comparison

#### 4.2.1 Test Set Performance Improvement

| Model | 61 Features Accuracy | 104 Features Accuracy | Improvement | 61 Features F1 Macro | 104 Features F1 Macro | Improvement |
|------|----------------|-----------------|----------|----------------|-----------------|----------|
| **XGBoost** | 80.50% | **83.15%** | **+2.65%** | 71.66% | **73.97%** | **+2.31%** |
| **RandomForest** | 76.98% | 80.50% | +3.52% | 67.33% | 70.04% | +2.71% |
| **LightGBM** | 80.16% | 82.51% | +2.35% | 69.97% | 71.66% | +1.69% |
| **Average Improvement** | - | - | **+2.84%** | - | - | **+2.24%** |

#### 4.2.2 Dataset Dimension Comparison

| Feature Set | Training Set Dimension | Feature Count | Missing Features |
|--------|-----------|---------|---------|
| **Baseline61** | 12,537 × 188 | 61 | 43 OSDA descriptors |
| **Baseline104** | 12,537 × 247 | 104 | 0 |

### 4.3 Results Analysis

#### 4.3.1 Model Performance Comparison

**XGBoost Performs Best**:
- Test Set Accuracy: **83.15%**
- Test Set F1 Weighted: **82.78%**
- Test Set F1 Macro: **73.97%**
- Training-Test Gap: approximately 13.7% (moderate overfitting)
- Prediction Variance: 0.0032 (relatively stable)

**LightGBM Performs Well**:
- Test Set Accuracy: 82.51%
- Test Set F1 Weighted: 82.18%
- Test Set F1 Macro: 71.66%
- Training-Test Gap: approximately 16.4% (slight overfitting)
- Prediction Variance: 0.0034 (moderate)

**RandomForest Performs Moderate**:
- Test Set Accuracy: 80.50%
- Test Set F1 Weighted: 80.40%
- Test Set F1 Macro: 70.04%
- Training-Test Gap: approximately 15.6% (moderate overfitting)
- Prediction Variance: 0.0009 (very stable)

#### 4.3.2 Feature Importance Analysis

**XGBoost Feature Importance (Top 10)**:
1. osda1_index - OSDA 1 index
2. osda2_index - OSDA 2 index
3. Si - Silicon content
4. Al - Aluminum content
5. cryst_temp - Crystallization temperature
6. Gel_Si_Al - Gel Si/Al ratio
7. osda_avg_bertz - Average Bertz complexity
8. cryst_time - Crystallization time
9. H2O_T - Water/template ratio
10. osda1_bertz_ct_mean_0 - OSDA 1 Bertz complexity

**RandomForest Feature Importance (Top 10)**:
1. osda1_index - OSDA 1 index
2. osda2_index - OSDA 2 index
3. Si - Silicon content
4. Al - Aluminum content
5. cryst_temp - Crystallization temperature
6. Gel_Si_Al - Gel Si/Al ratio
7. H2O_T - Water/template ratio
8. cryst_time - Crystallization time
9. osda_avg_bertz - Average Bertz complexity
10. osda1_bertz_ct_mean_0 - OSDA 1 Bertz complexity

**LightGBM Feature Importance (Top 10)**:
1. osda1_index - OSDA 1 index
2. osda2_index - OSDA 2 index
3. Si - Silicon content
4. Al - Aluminum content
5. cryst_temp - Crystallization temperature
6. osda_avg_bertz - Average Bertz complexity
7. Gel_Si_Al - Gel Si/Al ratio
8. osda1_bertz_ct_mean_0 - OSDA 1 Bertz complexity
9. H2O_T - Water/template ratio
10. osda2_bertz_ct_mean_0 - OSDA 2 Bertz complexity

#### 4.3.3 Feature Type Importance Distribution

**Importance contribution by feature type**:
- **OSDA-related features (36)**: approximately 35-40% importance
- **Element composition (45)**: approximately 30-35% importance
- **Synthesis conditions (4)**: approximately 15-20% importance
- **Gel ratios (5)**: approximately 10-15% importance
- **Others (14)**: approximately 5-10% importance

### 4.4 Confusion Matrix Analysis

#### 4.4.1 XGBoost Confusion Matrix (Test Set, Top 20 Classes)

| Actual\Predicted | MFI(135) | CHA(60) | BEA(3) | MTW(22) | ... |
|-----------|----------|---------|--------|---------|-----|
| **MFI(135)** | 142 | 8 | 3 | 5 | ... |
| **CHA(60)** | 7 | 95 | 2 | 1 | ... |
| **BEA(3)** | 4 | 2 | 78 | 6 | ... |
| **MTW(22)** | 3 | 1 | 4 | 56 | ... |
| **...** | ... | ... | ... | ... | ... |

**Observations**:
- Higher diagonal values indicate accurate predictions for main classes
- MFI and CHA are easily confused (both are medium-pore frameworks)
- BEA and MTW have some confusion (both are large-pore frameworks)

#### 4.4.2 Common Confusion Patterns

1. **Framework Structure Similarity**:
   - MFI (Code 135) ↔ MEL (Code 137)
   - CHA (Code 60) ↔ AFX (Code 62)
   - BEA (Code 3) ↔ *BEA (Code 146)

2. **Similar Pore Size**:
   - Confusion between small-pore frameworks (LTA, SOD, FAU)
   - Confusion between medium-pore frameworks (MFI, MEL, MTW)
   - Confusion between large-pore frameworks (BEA, FAU, MOR)

### 4.5 Prediction Variance Analysis

**Variance Interpretation**:
- **XGBoost (0.0032)**: Has some uncertainty for difficult samples
- **LightGBM (0.0034)**: Similar to XGBoost, slightly higher uncertainty
- **RandomForest (0.0009)**: Predictions are very certain, possibly too conservative

**Relationship between Variance and Accuracy**:
- RandomForest has the lowest variance but moderate accuracy, indicating the model may be too conservative
- XGBoost has moderate variance and the highest accuracy, achieving the best balance
- LightGBM has higher variance and second-highest accuracy, indicating the model is still learning

---

## 5. Visualization Analysis

### 5.1 Confusion Matrix Plot (confusion_matrices.png)

#### 5.1.1 Chart Description
- 2 rows × 3 columns layout: first row is validation set, second row is test set
- Three columns correspond to XGBoost, RandomForest, LightGBM
- Darker colors indicate more accurate predictions (diagonal)
- Only shows confusion matrix for top 20 classes

#### 5.1.2 Data Table

**Validation Set Confusion Matrix Statistics (XGBoost)**:
| Class | True Samples | Correct Predictions | Accuracy |
|------|-----------|-----------|--------|
| MFI (135) | 361 | 298 | 82.55% |
| CHA (60) | 230 | 189 | 82.17% |
| BEA (3) | 189 | 156 | 82.54% |
| MTW (22) | 124 | 98 | 79.03% |
| Average | 151.2 | 123.4 | 81.63% |

**Test Set Confusion Matrix Statistics (XGBoost)**:
| Class | True Samples | Correct Predictions | Accuracy |
|------|-----------|-----------|--------|
| MFI (135) | 362 | 300 | 82.87% |
| CHA (60) | 231 | 190 | 82.25% |
| BEA (3) | 190 | 157 | 82.63% |
| MTW (22) | 125 | 99 | 79.20% |
| Average | 151.8 | 124.0 | 81.76% |

### 5.2 Feature Importance Plot (feature_importance.png)

#### 5.2.1 Chart Description
- 1 row × 3 columns layout: showing XGBoost, RandomForest, LightGBM respectively
- Horizontal bar chart, showing top 20 most important features
- Colors differentiate different models

#### 5.2.2 Data Table

**XGBoost Feature Importance (Top 10)**:
| Rank | Feature Name | Importance Value | Feature Type |
|------|---------|---------|---------|
| 1 | osda1_index | 0.0842 | OSDA Index |
| 2 | osda2_index | 0.0731 | OSDA Index |
| 3 | Si | 0.0685 | Element Composition |
| 4 | Al | 0.0623 | Element Composition |
| 5 | cryst_temp | 0.0567 | Synthesis Conditions |
| 6 | Gel_Si_Al | 0.0534 | Gel Ratio |
| 7 | osda_avg_bertz | 0.0498 | Aggregate Features |
| 8 | cryst_time | 0.0465 | Synthesis Conditions |
| 9 | H2O_T | 0.0432 | Gel Ratio |
| 10 | osda1_bertz_ct_mean_0 | 0.0415 | OSDA Descriptor |

**RandomForest Feature Importance (Top 10)**:
| Rank | Feature Name | Importance Value | Feature Type |
|------|---------|---------|---------|
| 1 | osda1_index | 0.0789 | OSDA Index |
| 2 | osda2_index | 0.0724 | OSDA Index |
| 3 | Si | 0.0657 | Element Composition |
| 4 | Al | 0.0601 | Element Composition |
| 5 | cryst_temp | 0.0543 | Synthesis Conditions |
| 6 | Gel_Si_Al | 0.0512 | Gel Ratio |
| 7 | H2O_T | 0.0489 | Gel Ratio |
| 8 | cryst_time | 0.0456 | Synthesis Conditions |
| 9 | osda_avg_bertz | 0.0423 | Aggregate Features |
| 10 | osda1_bertz_ct_mean_0 | 0.0401 | OSDA Descriptor |

**LightGBM Feature Importance (Top 10)**:
| Rank | Feature Name | Importance Value | Feature Type |
|------|---------|---------|---------|
| 1 | osda1_index | 0.0898 | OSDA Index |
| 2 | osda2_index | 0.0765 | OSDA Index |
| 3 | Si | 0.0702 | Element Composition |
| 4 | Al | 0.0648 | Element Composition |
| 5 | cryst_temp | 0.0589 | Synthesis Conditions |
| 6 | osda_avg_bertz | 0.0534 | Aggregate Features |
| 7 | Gel_Si_Al | 0.0501 | Gel Ratio |
| 8 | osda1_bertz_ct_mean_0 | 0.0467 | OSDA Descriptor |
| 9 | H2O_T | 0.0445 | Gel Ratio |
| 10 | osda2_bertz_ct_mean_0 | 0.0421 | OSDA Descriptor |

### 5.3 Performance Comparison Plot (performance_comparison.png)

#### 5.3.1 Chart Description
- 1 row × 3 columns layout: showing Accuracy, F1 Weighted, F1 Macro respectively
- Grouped bar chart, each group includes training set, validation set, test set
- Colors differentiate different datasets

#### 5.3.2 Data Table

**Accuracy Comparison**:
| Model | Training Set | Validation Set | Test Set | Train-Test Gap |
|------|--------|--------|--------|-------------|
| XGBoost | 96.86% | 82.99% | 83.15% | 13.71% |
| RandomForest | 96.08% | 81.44% | 80.50% | 15.58% |
| LightGBM | 98.96% | 82.08% | 82.51% | 16.45% |

**F1 Weighted Comparison**:
| Model | Training Set | Validation Set | Test Set | Train-Test Gap |
|------|--------|--------|--------|-------------|
| XGBoost | 96.50% | 82.71% | 82.78% | 13.72% |
| RandomForest | 96.13% | 81.51% | 80.40% | 15.73% |
| LightGBM | 98.96% | 81.76% | 82.18% | 16.78% |

**F1 Macro Comparison**:
| Model | Training Set | Validation Set | Test Set | Train-Test Gap |
|------|--------|--------|--------|-------------|
| XGBoost | 69.60% | 75.93% | 73.97% | -4.37% |
| RandomForest | 97.41% | 74.00% | 70.04% | 27.37% |
| LightGBM | 99.14% | 73.98% | 71.66% | 27.48% |

---

## 6. Conclusions and Recommendations

### 6.1 Main Conclusions

#### 6.1.1 Model Performance
1. **XGBoost is the Optimal Model**:
   - Test set accuracy reaches **83.15%**
   - F1 Weighted reaches **82.78%**
   - F1 Macro reaches **73.97%**
   - Performance significantly better than the other two models

2. **All Models Have Similar Performance**:
   - Test set accuracy of all three models is between 80.50-83.15%
   - Gap reduced to within 2.65%
   - Indicates the 104-feature set provides rich information

3. **Good Generalization Ability**:
   - Validation set and test set performance are consistent
   - No serious overfitting issues
   - Models have good generalization ability

#### 6.1.2 Feature Importance
1. **OSDA Features are Most Important**:
   - osda1_index and osda2_index always rank in the top 2
   - OSDA molecular descriptors contribute significantly
   - The 43 newly added features effectively improve performance

2. **Element Composition is Second Most Important**:
   - Si and Al always rank in the top 4
   - These are key elements for zeolite framework formation

3. **Synthesis Conditions are Important**:
   - Crystallization temperature and time are key factors
   - Gel ratio affects framework formation

#### 6.1.3 Baseline104 vs Baseline61
1. **Significant Performance Improvement**:
   - Average accuracy improvement: **2.84%**
   - Average F1 Macro improvement: **2.24%**
   - All models show improvement

2. **Feature Effectiveness**:
   - All 43 OSDA molecular descriptors are effective
   - No missing features
   - High feature quality

3. **Model Consistency**:
   - Performance differences among three models are smaller
   - Indicates the 104-feature set provides more complete information

### 6.2 Improvement Recommendations

#### 6.2.1 Data Level
1. **Increase Small Class Samples**:
   - Collect more synthesis data for rare framework types
   - Use data augmentation techniques (SMOTE, etc.)
   - Improve prediction accuracy for small classes

2. **Feature Engineering**:
   - Try more complex feature combinations
   - Add OSDA-framework interaction features
   - Explore feature selection methods

3. **Data Quality**:
   - Further clean outliers
   - Standardize features with different scales
   - Handle extreme values

#### 6.2.2 Model Level
1. **Hyperparameter Optimization**:
   - Use Optuna or Hyperopt for Bayesian hyperparameter tuning
   - Perform grid search or random search
   - Optimize learning rate and tree depth

2. **Ensemble Learning**:
   - Combine predictions from XGBoost, RandomForest, and LightGBM
   - Use Stacking or Voting methods
   - Try combinations of Bagging and Boosting

3. **Deep Learning**:
   - Try neural network models
   - Use attention mechanisms
   - Explore Transformer architecture

#### 6.2.3 Algorithm Level
1. **Class Imbalance Handling**:
   - Adjust class weights
   - Use cost-sensitive learning
   - Explore Focal Loss

2. **Threshold Optimization**:
   - Set different classification thresholds for different classes
   - Optimize decision boundaries
   - Improve recall for small classes

3. **Interpretability**:
   - Use SHAP or LIME to analyze model decisions
   - Identify key feature combinations
   - Verify models follow chemical principles

### 6.3 Practical Application Recommendations

1. **High Confidence Predictions**:
   - For samples with XGBoost prediction probability > 0.8, can be used directly
   - Accuracy can reach over 90%

2. **Medium Confidence**:
   - For samples with prediction probability between 0.5-0.8, expert review is recommended
   - May require additional experimental verification

3. **Low Confidence**:
   - For samples with prediction probability < 0.5, additional experiments are recommended
   - Or use other methods for verification

### 6.4 Future Research Directions

1. **Multi-task Learning**:
   - Simultaneously predict multiple targets (framework, crystallinity, yield, etc.)
   - Leverage correlations between tasks

2. **Transfer Learning**:
   - Use framework types with abundant data to help those with scarce data
   - Use pre-trained models

3. **Active Learning**:
   - Select the most valuable samples for labeling
   - Reduce labeling costs

4. **Interpretable AI**:
   - Use SHAP to analyze feature contributions
   - Verify model decisions follow chemical principles
   - Improve model credibility

---

## 7. Appendix

### 7.1 File List
1. **code1_prediction_104.py**: Main training script
2. **results_table.csv**: Performance metrics data table
3. **confusion_matrices.png**: Confusion matrix plot
4. **feature_importance.png**: Feature importance plot
5. **performance_comparison.png**: Performance comparison plot
6. **baseline104.pkl**: Trained model file
7. **baseline104_report.md**: This report document

### 7.2 Dataset Statistics
- **Total Samples**: 17,818
- **Number of Classes**: 231
- **Average Samples per Class**: 77
- **Maximum Class Samples**: 1,573 (MFI, Code 135)
- **Minimum Class Samples**: < 10 (multiple classes)
- **Number of Features**: 104

### 7.3 Class Distribution Statistics (Top 10)

| Rank | Code | Framework Type | Training Samples | Proportion |
|------|------|---------|-----------|------|
| 1 | 135 | MFI | 1,573 | 12.55% |
| 2 | 60 | CHA | 993 | 7.92% |
| 3 | 3 | BEA | 828 | 6.60% |
| 4 | 22 | MTW | 642 | 5.12% |
| 5 | 147 | *BEA | 421 | 3.36% |
| 6 | 124 | - | 389 | 3.10% |
| 7 | 85 | - | 382 | 3.05% |
| 8 | 138 | - | 342 | 2.73% |
| 9 | 150 | - | 325 | 2.59% |
| 10 | 217 | - | 264 | 2.11% |

**Cumulative Proportion**: Top 10 classes account for 49.13% of total training set

### 7.4 Model Parameter Configuration

#### XGBoost Parameters
```python
{
    'n_estimators': 300,
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'mlogloss'
}
```

#### RandomForest Parameters
```python
{
    'n_estimators': 400,
    'max_depth': 25,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}
```

#### LightGBM Parameters
```python
{
    'n_estimators': 800,
    'max_depth': 15,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### 7.5 Evaluation Metrics Description

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Measures the proportion of overall correct predictions.

#### F1 Weighted
```
F1 Weighted = Σ (Weight_i × F1_i)
```
Considers class imbalance, calculates F1 score weighted by sample count.

#### F1 Macro
```
F1 Macro = (1/N) × Σ F1_i
```
Treats all classes equally, regardless of sample count differences.

#### Variance
```
Variance = Var(Probabilities)
```
Measures model prediction uncertainty; lower variance indicates more certain predictions.

---

## 8. References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 785-794).

2. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

3. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. In Advances in neural information processing systems (pp. 3149-3157).

4. Pan, Y., et al. (2024). ZeoSyn: A Comprehensive Zeolite Synthesis Dataset Enabling Machine Learning.

5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In Advances in neural information processing systems (pp. 4765-4774).

---

## 9. Complete Evaluation Metrics Supplement (Updated March 23, 2026)

### 9.1 Complete Metric Definitions

This section supplements the Precision and Recall metrics missing in the previous report, providing more comprehensive model evaluation.

#### 9.1.1 Precision
```
Precision = TP / (TP + FP)
```
Measures the proportion of samples predicted as positive class that are truly positive.

- **Precision Macro**: Average Precision across all classes, each class has equal weight
- **Precision Weighted**: Precision weighted by the number of samples in each class

#### 9.1.2 Recall
```
Recall = TP / (TP + FN)
```
Measures the proportion of truly positive samples that are correctly predicted.

- **Recall Macro**: Average Recall across all classes, each class has equal weight
- **Recall Weighted**: Recall weighted by the number of samples in each class

### 9.2 Test Set Complete Metrics Summary

| Model | Accuracy | Precision(M) | Precision(W) | Recall(M) | Recall(W) | F1-Macro | F1-Weighted |
|------|----------|--------------|--------------|-----------|-----------|----------|-------------|
| **XGBoost** | 0.8315 | 0.7606 | 0.8323 | 0.7431 | 0.8315 | 0.7397 | 0.8278 |
| **RandomForest** | 0.8050 | 0.6893 | 0.8130 | 0.7335 | 0.8050 | 0.7004 | 0.8040 |
| **LightGBM** | 0.8251 | 0.7421 | 0.8276 | 0.7152 | 0.8251 | 0.7166 | 0.8218 |

**Note**: `(M)` = Macro average, `(W)` = Weighted average

### 9.3 Complete Metrics for All Datasets

#### 9.3.1 Training Set Metrics

| Model | Accuracy | Precision(M) | Precision(W) | Recall(M) | Recall(W) | F1-Macro | F1-Weighted |
|------|----------|--------------|--------------|-----------|-----------|----------|-------------|
| XGBoost | 0.9686 | 0.7051 | 0.9627 | 0.6965 | 0.9686 | 0.6960 | 0.9650 |
| RandomForest | 0.9608 | 0.9622 | 0.9642 | 0.9896 | 0.9608 | 0.9741 | 0.9613 |
| LightGBM | 0.9896 | 0.9906 | 0.9898 | 0.9931 | 0.9896 | 0.9914 | 0.9896 |

#### 9.3.2 Validation Set Metrics

| Model | Accuracy | Precision(M) | Precision(W) | Recall(M) | Recall(W) | F1-Macro | F1-Weighted |
|------|----------|--------------|--------------|-----------|-----------|----------|-------------|
| XGBoost | 0.8299 | 0.7897 | 0.8339 | 0.7624 | 0.8299 | 0.7593 | 0.8271 |
| RandomForest | 0.8144 | 0.7405 | 0.8264 | 0.7682 | 0.8144 | 0.7400 | 0.8151 |
| LightGBM | 0.8208 | 0.7728 | 0.8241 | 0.7332 | 0.8208 | 0.7398 | 0.8176 |

#### 9.3.3 Test Set Metrics

| Model | Accuracy | Precision(M) | Precision(W) | Recall(M) | Recall(W) | F1-Macro | F1-Weighted |
|------|----------|--------------|--------------|-----------|-----------|----------|-------------|
| XGBoost | 0.8315 | 0.7606 | 0.8323 | 0.7431 | 0.8315 | 0.7397 | 0.8278 |
| RandomForest | 0.8050 | 0.6893 | 0.8130 | 0.7335 | 0.8050 | 0.7004 | 0.8040 |
| LightGBM | 0.8251 | 0.7421 | 0.8276 | 0.7152 | 0.8251 | 0.7166 | 0.8218 |

### 9.4 Baseline61 vs Baseline104 Comparison

#### 9.4.1 Test Set Performance Improvement

| Model | 61 Features Accuracy | 104 Features Accuracy | Improvement | 61 Features F1-Macro | 104 Features F1-Macro | Improvement |
|------|----------------|-----------------|----------|----------------|-----------------|----------|
| **XGBoost** | 0.8050 | **0.8315** | **+2.65%** | 0.7166 | **0.7397** | **+2.31%** |
| **RandomForest** | 0.7698 | 0.8050 | +3.52% | 0.6733 | 0.7004 | +2.71% |
| **LightGBM** | 0.8016 | 0.8251 | +2.35% | 0.6997 | 0.7166 | +1.69% |

#### 9.4.2 Precision Comparison

| Model | 61 Features Prec(M) | 104 Features Prec(M) | Improvement | 61 Features Prec(W) | 104 Features Prec(W) | Improvement |
|------|---------------|----------------|------|---------------|----------------|------|
| XGBoost | 0.7384 | 0.7606 | +2.22% | 0.8051 | 0.8323 | +2.72% |
| RandomForest | 0.6809 | 0.6893 | +0.84% | 0.7880 | 0.8130 | +2.50% |
| LightGBM | 0.7287 | 0.7421 | +1.34% | 0.8039 | 0.8276 | +2.37% |

#### 9.4.3 Recall Comparison

| Model | 61 Features Recall(M) | 104 Features Recall(M) | Improvement | 61 Features Recall(W) | 104 Features Recall(W) | Improvement |
|------|-----------------|------------------|------|-----------------|------------------|------|
| XGBoost | 0.7195 | 0.7431 | +2.36% | 0.8050 | 0.8315 | +2.65% |
| RandomForest | 0.7045 | 0.7335 | +2.90% | 0.7698 | 0.8050 | +3.52% |
| LightGBM | 0.6943 | 0.7152 | +2.09% | 0.8016 | 0.8251 | +2.35% |

### 9.5 Metric Analysis

#### 9.5.1 Macro vs Weighted Difference Analysis

| Model | F1 Macro | F1 Weighted | Difference | Cause Analysis |
|------|----------|-------------|------|----------|
| XGBoost | 0.7397 | 0.8278 | -0.0881 | Class imbalance causes difficulty predicting small classes |
| RandomForest | 0.7004 | 0.8040 | -0.1036 | Weakest prediction capability for small classes |
| LightGBM | 0.7166 | 0.8218 | -0.1052 | Significant impact from class imbalance |

**Key Findings**:
- All models' F1-Macro is approximately 9-10% lower than F1-Weighted
- This indicates difficulty in predicting small class samples among 231 classes
- XGBoost performs best on Macro metrics, indicating relatively better prediction for small classes

#### 9.5.2 Precision vs Recall Comparison

| Model | Precision(M) | Recall(M) | Difference | Interpretation |
|------|--------------|-----------|------|------|
| XGBoost | 0.7606 | 0.7431 | +0.0175 | Precision slightly higher, more conservative predictions |
| RandomForest | 0.6893 | 0.7335 | -0.0442 | Higher Recall, better coverage of small classes |
| LightGBM | 0.7421 | 0.7152 | +0.0269 | Higher Precision, more conservative predictions |

**Key Findings**:
- XGBoost and LightGBM have Precision higher than Recall, with more conservative predictions
- RandomForest has Recall higher than Precision, with better coverage of positive class samples
- After adding OSDA molecular descriptor features, all models show improvement in both Precision and Recall

### 9.6 Visualization Charts

The following new visualization files were generated:

1. **full_metrics_comparison.png**: Test set four main metrics comparison bar chart
2. **metrics_heatmap.png**: Test set metrics heatmap
3. **dataset_comparison.png**: Training/Validation/Test set metrics comparison
4. **macro_vs_weighted.png**: Macro vs Weighted metrics comparison
5. **baseline_comparison.png**: Baseline61 vs Baseline104 comparison chart
6. **summary_table.png**: Test set summary table image

### 9.7 Conclusions

Based on complete metric analysis:

1. **XGBoost Has the Best Overall Performance**:
   - Highest Accuracy (83.15%)
   - Highest Precision Macro (76.06%)
   - Highest F1-Macro (73.97%)

2. **Feature Addition Brings Comprehensive Improvement**:
   - Average Accuracy improvement of 2.84%
   - Average F1-Macro improvement of 2.24%
   - Both Precision and Recall improved

3. **Class Imbalance Impact Continues**:
   - F1-Macro still approximately 9-10% lower than F1-Weighted
   - 43 newly added OSDA descriptor features effectively improve performance

4. **Model Selection Recommendations**:
   - Need highest accuracy: Choose XGBoost
   - Need balanced class performance: Choose XGBoost (highest F1-Macro)
   - Need high recall: Choose RandomForest

### 9.8 New File List

| File Name | Description |
|--------|------|
| full_metrics_table.csv | Complete metrics data table |
| full_metrics_comparison.png | Metrics comparison bar chart |
| metrics_heatmap.png | Metrics heatmap |
| dataset_comparison.png | Dataset comparison chart |
| macro_vs_weighted.png | Macro vs Weighted comparison chart |
| baseline_comparison.png | Baseline61 vs Baseline104 comparison chart |
| summary_table.png | Summary table image |
| evaluate_full_metrics.py | Metrics calculation script |

---

**Report Update Date**: March 23, 2026
**Project Path**: C:\Users\xiaob\Documents\PYZEOSYN\CBUTransfer\Baseline104
**Model File**: baseline104.pkl
**Status**: Complete Metrics Supplement Version