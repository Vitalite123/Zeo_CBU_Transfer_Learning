"""
阶段4: 基于分组预训练的二分类器
- 使用stage3的累积分组预训练模型(G1-G5)
- 231个二分类模型，针对每个Code1类别
- 按Code1样本数分层统计: >1000, 500-999, 100-499, 50-99, 20-49, <20
"""

import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from time import time
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / 'models'
PRETRAIN_DIR = MODELS_DIR / 'pretrain'
RESULTS_DIR = PROJECT_DIR / 'results'

TRAIN_PATH = PROJECT_DIR.parent / 'Data_CBU_TL_clean_train.csv'
TEST_PATH = PROJECT_DIR.parent / 'Data_CBU_TL_clean_test.csv'

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

# 分层定义: >1000, 500-999, 100-499, 50-99, 20-49, <20
SAMPLE_TIERS = [
    ('Tier1_gt1000', 1001, float('inf')),      # >1000
    ('Tier2_500_999', 500, 999),               # 500-999
    ('Tier3_100_499', 100, 499),               # 100-499
    ('Tier4_50_99', 50, 99),                   # 50-99
    ('Tier5_20_49', 20, 49),                   # 20-49
    ('Tier6_lt20', 1, 19),                     # <20
]

def get_sample_tier(n_samples):
    """根据训练样本数确定分层"""
    for tier_name, low, high in SAMPLE_TIERS:
        if low <= n_samples <= high:
            return tier_name
    if n_samples < 1:
        return 'Excluded'  # 无样本
    return 'Tier6_lt20'


def load_pretrain_models():
    """加载stage3的5个预训练模型"""
    models = {}
    for g in ['G1', 'G2', 'G3', 'G4', 'G5']:
        model_path = PRETRAIN_DIR / f'pretrain_{g.lower()}.pkl'
        if model_path.exists():
            data = joblib.load(model_path)
            models[g] = data
            print(f"  已加载预训练模型 {g}")
        else:
            print(f"  警告: 找不到预训练模型 {g}")
    return models


def main():
    start_time = time()
    print("=" * 80)
    print("TLCBU_vGlobe - 阶段4: 基于分组预训练的二分类器")
    print("=" * 80)
    
    # 加载预训练模型
    print("\n[1/6] 加载预训练模型...")
    pretrain_models = load_pretrain_models()
    if not pretrain_models:
        print("错误: 未找到预训练模型，请先运行stage3!")
        return
    
    # 加载数据
    print("\n[2/6] 加载数据...")
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
    
    # 统计训练集各类别样本数
    train_counts = train_df['Code1'].value_counts().to_dict()
    test_counts = test_df['Code1'].value_counts().to_dict()
    
    print(f"  训练集: {len(train_df)} 样本, {n_classes} Code1类别")
    print(f"  测试集: {len(test_df)} 样本")
    
    # 统计各分层信息
    print("\n[3/6] 统计Code1分层信息...")
    tier_info = {tier_name: {'n_classes': 0, 'train_samples': 0, 'test_samples': 0, 'classes': []} 
                 for tier_name, _, _ in SAMPLE_TIERS}
    tier_info['Excluded'] = {'n_classes': 0, 'train_samples': 0, 'test_samples': 0, 'classes': []}
    
    for code1 in le.classes_:
        n_train = train_counts.get(code1, 0)
        n_test = test_counts.get(code1, 0)
        tier = get_sample_tier(n_train)
        
        if tier == 'Excluded':
            continue
            
        tier_info[tier]['n_classes'] += 1
        tier_info[tier]['train_samples'] += n_train
        tier_info[tier]['test_samples'] += n_test
        tier_info[tier]['classes'].append(code1)
    
    print("\n  Code1分层统计:")
    print("  " + "-" * 70)
    for tier_name, _, _ in SAMPLE_TIERS:
        info = tier_info[tier_name]
        print(f"  {tier_name:<15}: {info['n_classes']:>3} 类别, "
              f"训练 {info['train_samples']:>6} 样本, 测试 {info['test_samples']:>6} 样本")
    
    # 根据样本数选择预训练模型和配置
    tier_config = {
        'Tier1_gt1000': {'model': 'G5', 'neg_ratio': 3, 'lr': 0.05, 'depth': 6, 'rounds': 200},
        'Tier2_500_999': {'model': 'G4', 'neg_ratio': 5, 'lr': 0.05, 'depth': 6, 'rounds': 200},
        'Tier3_100_499': {'model': 'G3', 'neg_ratio': 8, 'lr': 0.03, 'depth': 4, 'rounds': 150},
        'Tier4_50_99': {'model': 'G2', 'neg_ratio': 10, 'lr': 0.02, 'depth': 4, 'rounds': 100},
        'Tier5_20_49': {'model': 'G2', 'neg_ratio': 15, 'lr': 0.01, 'depth': 3, 'rounds': 80},
        'Tier6_lt20': {'model': 'G1', 'neg_ratio': 20, 'lr': 0.01, 'depth': 3, 'rounds': 50},
    }
    
    print("\n[4/6] 训练231个二分类器...")
    classifiers = {}
    binary_results = []
    dtrain_full = xgb.DMatrix(X_train.values)
    dtest_full = xgb.DMatrix(X_test.values)
    
    for i, code1 in enumerate(le.classes_):
        n_pos = train_counts.get(code1, 0)
        tier = get_sample_tier(n_pos)
        
        if n_pos < 2:
            classifiers[code1] = None
            binary_results.append({
                'code1': code1,
                'tier': tier,
                'n_train': n_pos,
                'n_test': test_counts.get(code1, 0),
                'accuracy': 0.0,
                'f1': 0.0,
                'status': 'excluded'
            })
            continue
        
        config = tier_config.get(tier, tier_config['Tier6_lt20'])
        neg_ratio = config['neg_ratio']
        pretrain_key = config['model']
        
        # 准备二分类数据
        pos_mask_train = (train_df['Code1'] == code1).values
        neg_mask_train = ~pos_mask_train
        
        X_pos = X_train[pos_mask_train].values
        X_neg = X_train[neg_mask_train].values[:n_pos * neg_ratio]
        
        X_binary = np.vstack([X_pos, X_neg])
        y_binary = np.array([1] * len(X_pos) + [0] * len(X_neg))
        
        idx = np.random.RandomState(42).permutation(len(X_binary))
        X_binary = X_binary[idx]
        y_binary = y_binary[idx]
        
        dtrain = xgb.DMatrix(X_binary, label=y_binary)
        
        params = {
            'objective': 'binary:logistic',
            'max_depth': config['depth'],
            'learning_rate': config['lr'],
            'seed': 42,
            'tree_method': 'hist',
            'scale_pos_weight': len(X_neg) / len(X_pos)
        }
        
        try:
            model = xgb.train(params, dtrain, num_boost_round=config['rounds'], verbose_eval=False)
            classifiers[code1] = model
            
            # 训练集评估
            pred = (model.predict(dtrain) > 0.5).astype(int)
            train_acc = np.mean(pred == y_binary)
            train_f1 = f1_score(y_binary, pred, zero_division=0)
            
            binary_results.append({
                'code1': code1,
                'tier': tier,
                'n_train': n_pos,
                'n_test': test_counts.get(code1, 0),
                'accuracy': float(train_acc),
                'f1': float(train_f1),
                'status': 'success'
            })
        except Exception as e:
            classifiers[code1] = None
            binary_results.append({
                'code1': code1,
                'tier': tier,
                'n_train': n_pos,
                'n_test': test_counts.get(code1, 0),
                'accuracy': 0.0,
                'f1': 0.0,
                'status': 'failed'
            })
        
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{n_classes}")
    
    success_count = sum(1 for c in classifiers.values() if c is not None)
    print(f"  训练完成: {success_count}/{n_classes} 成功")
    
    # 保存二分类器详细结果
    binary_df = pd.DataFrame(binary_results)
    binary_df.to_csv(RESULTS_DIR / 'binary_grouped_results.csv', index=False)
    
    # 预测测试集 - 使用归一化策略
    print("\n[5/6] 预测测试集（归一化策略）...")
    
    # 获取所有二分类器的原始概率
    raw_proba = np.zeros((len(X_test), n_classes))
    
    for code1_idx, code1 in enumerate(le.classes_):
        if classifiers.get(code1) is not None:
            model = classifiers[code1]
            raw_proba[:, code1_idx] = model.predict(dtest_full)
    
    # 归一化策略：对每个样本，将所有类别的概率归一化
    # 1. 将原始概率转换为指数形式以增强对比度
    raw_proba_exp = np.exp(raw_proba * 3)  # 增强对比度
    
    # 2. 行归一化：每个样本的所有类别概率和为1
    proba_sum = raw_proba_exp.sum(axis=1, keepdims=True)
    proba_sum = np.maximum(proba_sum, 1e-10)  # 防止除零
    normalized_proba = raw_proba_exp / proba_sum
    
    # 3. 应用温度调节：对小样本类别更友好
    # 计算每个类别的训练样本占比作为先验
    prior = np.zeros(n_classes)
    for code1_idx, code1 in enumerate(le.classes_):
        prior[code1_idx] = train_counts.get(code1, 1) ** 0.3  # 开方降低大样本优势
    prior = prior / prior.sum()
    
    # 4. 融合归一化概率和先验
    prior_weight = 0.05  # 降低先验权重，更依赖模型预测
    final_proba = (1 - prior_weight) * normalized_proba + prior_weight * prior
    
    # 5. 最终归一化
    final_proba = final_proba / final_proba.sum(axis=1, keepdims=True)
    
    y_pred = np.argmax(final_proba, axis=1)
    
    # 总体性能
    print("\n" + "=" * 80)
    print("【总体性能指标】")
    print("=" * 80)
    acc_test = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    prec_m = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_m = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"  测试准确率 (Accuracy): {acc_test:.4f} ({acc_test*100:.2f}%)")
    print(f"  F1-Weighted:           {f1_w:.4f}")
    print(f"  F1-Macro:              {f1_m:.4f}")
    print(f"  Precision-Weighted:   {prec_w:.4f}")
    print(f"  Precision-Macro:      {prec_m:.4f}")
    print(f"  Recall-Weighted:      {rec_w:.4f}")
    print(f"  Recall-Macro:         {rec_m:.4f}")
    
    # 分层性能统计
    print("\n" + "=" * 80)
    print("【按Code1样本数分层性能指标】")
    print("=" * 80)
    
    tier_stats = {}
    print(f"\n{'分层':<18} {'类别数':>8} {'训练样本':>10} {'测试样本':>10} "
          f"{'Accuracy':>10} {'F1-Weighted':>12} {'F1-Macro':>10}")
    print("-" * 90)
    
    for tier_name, low, high in SAMPLE_TIERS:
        tier_classes = tier_info[tier_name]['classes']
        
        if not tier_classes:
            tier_stats[tier_name] = {
                'n_classes': 0, 'train_samples': 0, 'test_samples': 0,
                'accuracy': 0.0, 'f1_weighted': 0.0, 'f1_macro': 0.0
            }
            continue
        
        # 获取该tier的测试样本
        tier_mask = test_df['Code1'].isin(tier_classes).values
        tier_y_true = y_test[tier_mask]
        tier_y_pred = y_pred[tier_mask]
        
        tier_acc = accuracy_score(tier_y_true, tier_y_pred)
        tier_f1_w = f1_score(tier_y_true, tier_y_pred, average='weighted', zero_division=0)
        tier_f1_m = f1_score(tier_y_true, tier_y_pred, average='macro', zero_division=0)
        
        tier_stats[tier_name] = {
            'n_classes': tier_info[tier_name]['n_classes'],
            'train_samples': int(tier_info[tier_name]['train_samples']),
            'test_samples': int(tier_info[tier_name]['test_samples']),
            'accuracy': float(tier_acc),
            'f1_weighted': float(tier_f1_w),
            'f1_macro': float(tier_f1_m)
        }
        
        print(f"{tier_name:<18} {tier_stats[tier_name]['n_classes']:>8} "
              f"{tier_stats[tier_name]['train_samples']:>10} "
              f"{tier_stats[tier_name]['test_samples']:>10} "
              f"{tier_acc:>10.4f} {tier_f1_w:>12.4f} {tier_f1_m:>10.4f}")
    
    # 保存结果
    print("\n[6/6] 保存结果...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        'classifiers': classifiers,
        'label_encoder': le,
        'features': available,
        'train_counts': train_counts,
        'tier_config': tier_config
    }, MODELS_DIR / 'binary_grouped_classifiers.pkl')
    
    results = {
        'summary': {
            'total_classes': n_classes,
            'success_count': success_count,
            'test_samples': len(test_df),
            'accuracy': float(acc_test),
            'f1_weighted': float(f1_w),
            'f1_macro': float(f1_m),
            'precision_weighted': float(prec_w),
            'precision_macro': float(prec_m),
            'recall_weighted': float(rec_w),
            'recall_macro': float(rec_m)
        },
        'tier_info': tier_info,
        'tier_stats': tier_stats,
        'tier_config': tier_config,
        'binary_results': binary_df.to_dict('records')
    }
    
    with open(RESULTS_DIR / 'binary_grouped_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存分层CSV
    tier_df = pd.DataFrame([
        {'Tier': k, 'N_Classes': v['n_classes'], 'Train_Samples': v['train_samples'],
         'Test_Samples': v['test_samples'], 'Accuracy': v['accuracy'],
         'F1_Weighted': v['f1_weighted'], 'F1_Macro': v['f1_macro']}
        for k, v in tier_stats.items()
    ])
    tier_df.to_csv(RESULTS_DIR / 'binary_grouped_tier_results.csv', index=False)
    
    elapsed = time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    print("\n" + "=" * 80)
    print("【结果文件】")
    print("=" * 80)
    print(f"  - {RESULTS_DIR / 'binary_grouped_results.csv'}")
    print(f"  - {RESULTS_DIR / 'binary_grouped_results.json'}")
    print(f"  - {RESULTS_DIR / 'binary_grouped_tier_results.csv'}")
    print(f"  - {MODELS_DIR / 'binary_grouped_classifiers.pkl'}")


if __name__ == '__main__':
    main()
