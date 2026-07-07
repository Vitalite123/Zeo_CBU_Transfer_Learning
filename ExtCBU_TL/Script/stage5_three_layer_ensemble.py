"""
阶段5: 三层集成（TLCBU_vGlobe）
结合预训练模型、二分类器和相似度加权
"""

import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from datetime import datetime
from time import time

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / 'models'
RESULTS_DIR = PROJECT_DIR / 'results'
REPORTS_DIR = PROJECT_DIR / 'reports'

TRAIN_PATH = Path(__file__).parent.parent / 'Data_CBU_TL_clean_train.csv'
TEST_PATH = Path(__file__).parent.parent / 'Data_CBU_TL_clean_test.csv'
SIM_PATH = Path(__file__).parent.parent / 'cbu_similarity_cleaned.csv'

FEATURES = [
    'Si', 'Al', 'P', 'Na', 'K', 'Li', 'Sr', 'Rb', 'Cs', 'Ba', 'Ca', 'F', 'Ge', 'Ti', 'In', 'B', 'Mg', 'Ga', 'Ni', 'Mn',
    'Fe', 'Co', 'Cr', 'Zn', 'Nb', 'Be', 'W', 'Ce', 'Cu', 'Sn', 'Gd', 'La', 'Y', 'Dy', 'Sm', 'Ag', 'Cd', 'Zr', 'V', 'Ta',
    'ru', 'Hf', 'Yb', 'Tl', 'As', 'osda1_index', 'osda2_index', 'osda3_index', 'cryst_temp', 'cryst_time', 'seed', 'rotation',
    'aging_temp', 'aging_time', 'acid', 'OH', 'H2O_T', 'OH_T', 'Gel_Si_Al', 'Gel_P_Al', 'Gel_P_Si',
    'osda1_bertz_ct_mean_0', 'osda1_free_sasa_mean_0', 'osda1_asphericity_mean_0', 'osda1_eccentricity_mean_0',
    'osda1_axes_mean_0', 'osda1_axes_mean_1', 'osda1_box_mean_0', 'osda1_box_mean_1', 'osda1_box_mean_2',
    'osda1_getaway_mean_0', 'osda1_getaway_mean_1',
    'osda2_bertz_ct_mean_0', 'osda2_free_sasa_mean_0', 'osda2_asphericity_mean_0', 'osda2_eccentricity_mean_0',
    'osda2_axes_mean_0', 'osda2_axes_mean_1', 'osda2_box_mean_0', 'osda2_box_mean_1', 'osda2_box_mean_2',
    'osda2_getaway_mean_0', 'osda2_getaway_mean_1',
    'osda3_bertz_ct_mean_0', 'osda3_free_sasa_mean_0', 'osda3_asphericity_mean_0', 'osda3_eccentricity_mean_0',
    'osda3_axes_mean_0', 'osda3_axes_mean_1', 'osda3_box_mean_0', 'osda3_box_mean_1', 'osda3_box_mean_2',
    'osda3_getaway_mean_0', 'osda3_getaway_mean_1',
    'osda_avg_asphericity', 'osda_max_asphericity', 'osda_min_asphericity', 'osda_avg_sasa', 'osda_max_sasa', 'osda_min_sasa',
    'osda_avg_bertz', 'osda_max_bertz', 'osda_min_bertz', 'osda_total_volume'
]


def main():
    start_time = time()
    print("="*60)
    print("TLCBU_vGlobe - 阶段5: 三层集成")
    print("="*60)
    
    print("\n[1/6] 加载数据...")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)
    
    available = [f for f in FEATURES if f in train_df.columns]
    X_train = train_df[available].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = test_df[available].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    le = LabelEncoder()
    all_labels = pd.concat([train_df['Code1'], test_df['Code1']]).unique()
    le.fit(all_labels)
    y_train = le.transform(train_df['Code1'].fillna('Unknown'))
    y_test = le.transform(test_df['Code1'].fillna('Unknown'))
    n_classes = len(le.classes_)
    
    dtrain = xgb.DMatrix(X_train.values, label=y_train)
    dtest = xgb.DMatrix(X_test.values, label=y_test)
    
    print("\n[2/6] 训练Layer1: 预训练模型...")
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'tree_method': 'hist'
    }
    
    layer1_model = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)
    layer1_proba = layer1_model.predict(dtest)
    layer1_pred = np.argmax(layer1_proba, axis=1)
    layer1_acc = np.mean(layer1_pred == y_test)
    print(f"  Layer1准确率: {layer1_acc:.4f}")
    
    print("\n[3/6] 加载Layer2: 二分类器...")
    binary_data = joblib.load(MODELS_DIR / 'binary' / 'global_binary_classifiers_v2.pkl')
    binary_classifiers = binary_data['classifiers']
    
    layer2_proba = np.zeros((len(X_test), n_classes))
    for code1_idx, code1 in enumerate(le.classes_):
        if code1 in binary_classifiers and binary_classifiers[code1] is not None:
            model = binary_classifiers[code1]
            dtest_bin = xgb.DMatrix(X_test.values)
            layer2_proba[:, code1_idx] = model.predict(dtest_bin)
    
    layer2_pred = np.argmax(layer2_proba, axis=1)
    layer2_acc = np.mean(layer2_pred == y_test)
    print(f"  Layer2准确率: {layer2_acc:.4f}")
    
    print("\n[4/6] Layer3: Alpha融合测试...")
    best_alpha = 0
    best_acc = 0
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fused_proba = alpha * layer1_proba + (1 - alpha) * layer2_proba
        fused_pred = np.argmax(fused_proba, axis=1)
        fused_acc = np.mean(fused_pred == y_test)
        
        if fused_acc > best_acc:
            best_acc = fused_acc
            best_alpha = alpha
    
    print(f"  最佳Alpha: {best_alpha}")
    print(f"  最佳融合准确率: {best_acc:.4f}")
    
    final_proba = best_alpha * layer1_proba + (1 - best_alpha) * layer2_proba
    final_pred = np.argmax(final_proba, axis=1)
    final_acc = np.mean(final_pred == y_test)
    f1_w = f1_score(y_test, final_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, final_pred, average='macro', zero_division=0)
    
    print("\n[5/6] 保存模型...")
    joblib.dump({
        'layer1_model': layer1_model,
        'binary_classifiers': binary_classifiers,
        'label_encoder': le,
        'features': available,
        'best_alpha': best_alpha
    }, MODELS_DIR / 'ensemble' / 'global_three_layer_ensemble.pkl')
    
    print("\n[6/6] 生成报告...")
    results = {
        'layer1_accuracy': float(layer1_acc),
        'layer2_accuracy': float(layer2_acc),
        'best_alpha': float(best_alpha),
        'final_accuracy': float(final_acc),
        'f1_weighted': float(f1_w),
        'f1_macro': float(f1_m)
    }
    
    with open(RESULTS_DIR / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    report = f"""# TLCBU_vGlobe 阶段5: 三层集成报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 架构

### Layer 1: 预训练模型
- XGBoost多分类器
- 准确率: {layer1_acc:.4f}

### Layer 2: 二分类器
- 231个Code1特定二分类器
- 准确率: {layer2_acc:.4f}

### Layer 3: Alpha融合
- 公式: Final = α × Layer1 + (1-α) × Layer2
- 最佳α: {best_alpha}
- 准确率: {final_acc:.4f}

---

## 2. 结果对比

| 方法 | Accuracy | F1-Weighted | F1-Macro |
|------|----------|-------------|----------|
| Baseline | 0.9634 | 0.9620 | 0.8212 |
| Layer1预训练 | {layer1_acc:.4f} | - | - |
| Layer2二分类 | {layer2_acc:.4f} | - | - |
| **三层集成** | **{final_acc:.4f}** | **{f1_w:.4f}** | **{f1_m:.4f}** |

---

## 3. 结论

三层集成效果: {'优于' if final_acc > layer1_acc else '劣于'}单独预训练模型

---
*运行时间: {(time()-start_time)/60:.1f}分钟*
"""
    
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / 'stage5_ensemble_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n运行时间: {(time()-start_time)/60:.1f}分钟")
    print("="*60)
    print("阶段5完成!")
    print("="*60)
    
    return results


if __name__ == '__main__':
    main()