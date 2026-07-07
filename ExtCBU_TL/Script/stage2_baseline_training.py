"""
阶段2: Baseline XGBoost模型训练（TLCBU_vGlobe）
基于全局特征相似度的迁移学习项目

任务：
1. 加载数据和特征
2. 训练XGBoost Baseline模型
3. 评估验证集和测试集性能

输出：
- models/baseline/global_baseline_xgboost.pkl
- results/baseline/global_baseline_results.json
"""

import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ========== 路径配置 ==========
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / 'data'
MODELS_DIR = PROJECT_DIR / 'models' / 'baseline'
RESULTS_DIR = PROJECT_DIR / 'results'

# 数据文件
TRAIN_DATA_PATH = Path(__file__).parent.parent / 'Data_CBU_TL_clean_train.csv'
TEST_DATA_PATH = Path(__file__).parent.parent / 'Data_CBU_TL_clean_test.csv'

# 特征列表（104特征）
FEATURE_104 = [
    'Si', 'Al', 'P', 'Na', 'K', 'Li', 'Sr', 'Rb', 'Cs', 'Ba',
    'Ca', 'F', 'Ge', 'Ti', 'In', 'B', 'Mg', 'Ga', 'Ni', 'Mn',
    'Fe', 'Co', 'Cr', 'Zn', 'Nb', 'Be', 'W', 'Ce', 'Cu', 'Sn',
    'Gd', 'La', 'Y', 'Dy', 'Sm', 'Ag', 'Cd', 'Zr', 'V', 'Ta',
    'Ru', 'Hf', 'Yb', 'Tl', 'As',
    'osda1_index', 'osda2_index', 'osda3_index',
    'cryst_temp', 'cryst_time', 'seed', 'rotation',
    'aging_temp', 'aging_time',
    'acid', 'OH',
    'H2O_T', 'OH_T', 'Gel_Si_Al', 'Gel_P_Al', 'Gel_P_Si',
    'osda1_bertz_ct_mean_0', 'osda1_free_sasa_mean_0', 'osda1_asphericity_mean_0',
    'osda1_eccentricity_mean_0', 'osda1_axes_mean_0', 'osda1_axes_mean_1',
    'osda1_box_mean_0', 'osda1_box_mean_1', 'osda1_box_mean_2',
    'osda1_getaway_mean_0', 'osda1_getaway_mean_1',
    'osda2_bertz_ct_mean_0', 'osda2_free_sasa_mean_0', 'osda2_asphericity_mean_0',
    'osda2_eccentricity_mean_0', 'osda2_axes_mean_0', 'osda2_axes_mean_1',
    'osda2_box_mean_0', 'osda2_box_mean_1', 'osda2_box_mean_2',
    'osda2_getaway_mean_0', 'osda2_getaway_mean_1',
    'osda3_bertz_ct_mean_0', 'osda3_free_sasa_mean_0', 'osda3_asphericity_mean_0',
    'osda3_eccentricity_mean_0', 'osda3_axes_mean_0', 'osda3_axes_mean_1',
    'osda3_box_mean_0', 'osda3_box_mean_1', 'osda3_box_mean_2',
    'osda3_getaway_mean_0', 'osda3_getaway_mean_1',
    'osda_avg_asphericity', 'osda_max_asphericity', 'osda_min_asphericity',
    'osda_avg_sasa', 'osda_max_sasa', 'osda_min_sasa',
    'osda_avg_bertz', 'osda_max_bertz', 'osda_min_bertz',
    'osda_total_volume'
]

TARGET_COLUMN = 'Code1'


def load_data():
    """加载数据"""
    print("\n[1/4] 加载数据...")
    
    train_df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_DATA_PATH, low_memory=False)
    
    print(f"  训练集: {train_df.shape[0]}样本")
    print(f"  测试集: {test_df.shape[0]}样本")
    
    return train_df, test_df


def extract_features(df):
    """提取104特征"""
    print("\n[2/4] 提取104特征...")
    
    available_features = [f for f in FEATURE_104 if f in df.columns]
    print(f"  可用特征: {len(available_features)}/104")
    
    X = df[available_features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())
    
    return X, available_features


def encode_labels(train_df, test_df):
    """编码标签"""
    print("\n[3/4] 编码标签...")
    
    train_labels = train_df[TARGET_COLUMN].fillna('Unknown')
    test_labels = test_df[TARGET_COLUMN].fillna('Unknown')
    
    all_labels = pd.concat([train_labels, test_labels]).unique()
    
    le = LabelEncoder()
    le.fit(all_labels)
    
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)
    
    num_classes = len(le.classes_)
    print(f"  类别数: {num_classes}")
    
    return y_train, y_test, le, num_classes


def train_xgboost_model(X_train, y_train, X_test, y_test, num_classes):
    """训练XGBoost Baseline模型"""
    print("\n[4/4] 训练Baseline模型...")
    
    all_classes = np.arange(num_classes)
    missing_classes = np.setdiff1d(all_classes, np.unique(y_train))
    if len(missing_classes) > 0:
        print(f"  警告: {len(missing_classes)}个类别在训练集中缺失")
    
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**params)
    
    try:
        model.fit(X_train, y_train, verbose=50)
    except Exception as e:
        print(f"  训练出错: {e}")
        print("  尝试简化参数...")
        params['max_depth'] = 5
        params['n_estimators'] = 100
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=50)
    
    return model


def evaluate_model(model, X, y_true, label_encoder, set_name='Test'):
    """评估模型"""
    y_pred_proba = model.predict_proba(X)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = np.mean(y_pred == y_true)
    
    try:
        from sklearn.metrics import f1_score
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    except:
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n{set_name}集评估结果:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Weighted: {f1_weighted:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro)
    }


def main():
    """主函数"""
    print("\n" + "="*60)
    print("TLCBU_vGlobe - 阶段2: Baseline XGBoost训练")
    print("="*60 + "\n")
    
    train_df, test_df = load_data()
    
    X_train, features = extract_features(train_df)
    X_test, _ = extract_features(test_df)
    
    y_train, y_test, label_encoder, num_classes = encode_labels(train_df, test_df)
    
    model = train_xgboost_model(X_train.values, y_train, X_test.values, y_test, num_classes)
    
    train_metrics = evaluate_model(model, X_train.values, y_train, num_classes, 'Train')
    test_metrics = evaluate_model(model, X_test.values, y_test, num_classes, 'Test')
    
    results = {
        'train': train_metrics,
        'test': test_metrics,
        'num_classes': num_classes,
        'num_features': len(features)
    }
    
    results_path = RESULTS_DIR / 'global_baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存: {results_path}")
    
    model_path = MODELS_DIR / 'global_baseline_xgboost.pkl'
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder,
        'features': features
    }, model_path)
    print(f"模型保存: {model_path}")
    
    print("\n" + "="*60)
    print("阶段2完成!")
    print("="*60)
    
    return results


if __name__ == '__main__':
    main()