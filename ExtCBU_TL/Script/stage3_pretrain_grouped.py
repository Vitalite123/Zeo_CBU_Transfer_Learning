"""
阶段3: 分组预训练模型（TLCBU_vGlobe）
根据CBU相似度分组，训练5个预训练模型
- 分组索引: cbu_category (小写三个字母)
- 预测目标: Code1 (骨架类型)
"""

import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / 'models' / 'pretrain'
RESULTS_DIR = PROJECT_DIR / 'results'
REPORTS_DIR = PROJECT_DIR / 'reports'

# 数据文件在上级目录
DATA_DIR = PROJECT_DIR.parent
TRAIN_PATH = DATA_DIR / 'Data_CBU_TL_clean_train.csv'
TEST_PATH = DATA_DIR / 'Data_CBU_TL_clean_test.csv'

# CBU分组定义（使用小写，与cbu_category一致）
CBU_GROUPS = {
    'G1_VeryHigh': ['mel', 'd4r', 'd6r', 'cas', 'mfi', 'sod', 'cha', 'mor', 'lau', 'lta', 'phi', 'lev', 'ats', 'd8r', 'dsc', 'afs'],
    'G2_High': ['stf', 'bre', 'jbw', 'imf', 'dzc', 'non', 'mtw', 'doh', 'gme', 'ton', 'ddr', 'mtt', 'pcr', 'fer', 'bik', 'bea'],
    'G3_Medium': ['d3r', 'vsv', 'rte', 'lov', 'pau', 'sti', 'bog', 'mtn', 'mwf', 'afi', 'gis', 'rth', 'rut', 'dcc', 'ltl', 'ast'],
    'G4_Low': ['aft', 'mei', 'can', 'nsc', 'abw', 'nat', 'clo', 'atn', 'dnc', 'mso', 'bph', 'afy', 'los', 'aww', 'eab', 'ave']
}

# 全量特征
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


def get_group_for_cbu_category(cbu_cat):
    """根据cbu_category获取分组（小写匹配）"""
    if pd.isna(cbu_cat):
        return None
    cbu_lower = str(cbu_cat).lower()
    for group, cbus in CBU_GROUPS.items():
        if cbu_lower in cbus:
            return group
    return None  # 不在任何组中


def main():
    print("=" * 80)
    print("TLCBU_vGlobe - 阶段3: 分组预训练模型")
    print("=" * 80)
    
    print("\n[1/8] 加载数据...")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)
    
    available = [f for f in FEATURES if f in train_df.columns]
    X_train_full = train_df[available].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_full = test_df[available].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 预测目标: Code1
    le_full = LabelEncoder()
    all_labels = pd.concat([train_df['Code1'], test_df['Code1']]).unique()
    le_full.fit(all_labels)
    y_train_full = le_full.transform(train_df['Code1'].fillna('Unknown'))
    y_test_full = le_full.transform(test_df['Code1'].fillna('Unknown'))
    n_classes_full = len(le_full.classes_)
    
    train_counts = train_df['Code1'].value_counts().to_dict()
    
    print(f"  训练集: {len(train_df)} 样本, {n_classes_full} Code1类别")
    print(f"  测试集: {len(test_df)} 样本")
    
    # 使用cbu_category进行分组
    train_df['Group'] = train_df['cbu_category'].apply(get_group_for_cbu_category)
    test_df['Group'] = test_df['cbu_category'].apply(get_group_for_cbu_category)
    
    # 累积分组定义（在使用前定义）
    cumulative_groups = {
        'G1': CBU_GROUPS['G1_VeryHigh'],
        'G2': CBU_GROUPS['G1_VeryHigh'] + CBU_GROUPS['G2_High'],
        'G3': CBU_GROUPS['G1_VeryHigh'] + CBU_GROUPS['G2_High'] + CBU_GROUPS['G3_Medium'],
        'G4': CBU_GROUPS['G1_VeryHigh'] + CBU_GROUPS['G2_High'] + CBU_GROUPS['G3_Medium'] + CBU_GROUPS['G4_Low'],
        'G5': 'ALL'  # 全部数据
    }
    
    # 统计分组
    print("\n  训练集cbu_category分组统计:")
    group_stats = {}
    for group in ['G1', 'G2', 'G3', 'G4']:
        group_cbus = cumulative_groups[group]
        group_mask = train_df['cbu_category'].isin(group_cbus)
        group_df = train_df[group_mask]
        n_cbus = group_df['cbu_category'].nunique()
        n_samples = len(group_df)
        n_code1_classes = group_df['Code1'].nunique()
        group_stats[group] = {'n_cbus': n_cbus, 'n_samples': n_samples, 'n_code1': n_code1_classes, 'cbus': group_cbus}
        print(f"    {group}: {n_cbus} cbu, {n_samples} 样本, {n_code1_classes} Code1类别")
    
    # 未分组CBU
    grouped_cbus = set()
    for g in ['G1', 'G2', 'G3', 'G4']:
        grouped_cbus.update(cumulative_groups[g])
    ungrouped_mask = ~train_df['cbu_category'].isin(grouped_cbus)
    ungrouped_df = train_df[ungrouped_mask]
    print(f"    未分组: {ungrouped_df['cbu_category'].nunique()} cbu, {len(ungrouped_df)} 样本")
    
    # 模型配置
    model_configs = {
        'G1': {'max_depth': 8, 'lr': 0.1, 'rounds': 250, 'subsample': 0.9, 'colsample': 0.9},
        'G2': {'max_depth': 8, 'lr': 0.1, 'rounds': 250, 'subsample': 0.9, 'colsample': 0.9},
        'G3': {'max_depth': 8, 'lr': 0.08, 'rounds': 200, 'subsample': 0.85, 'colsample': 0.85},
        'G4': {'max_depth': 6, 'lr': 0.08, 'rounds': 200, 'subsample': 0.85, 'colsample': 0.85},
        'G5': {'max_depth': 10, 'lr': 0.08, 'rounds': 350, 'subsample': 0.9, 'colsample': 0.9}
    }
    
    print("\n[2/8] 训练分组模型...")
    
    all_results = {}
    all_models = {}
    all_feature_importance = {}
    
    # 获取所有CBU列表
    all_cbus = CBU_GROUPS['G1_VeryHigh'] + CBU_GROUPS['G2_High'] + \
               CBU_GROUPS['G3_Medium'] + CBU_GROUPS['G4_Low']
    
    for group_name in ['G1', 'G2', 'G3', 'G4', 'G5']:
        print(f"\n  --- 训练 {group_name} ---")
        
        config = model_configs[group_name]
        
        # 确定该组的CBU列表（累积分组）
        if cumulative_groups[group_name] == 'ALL':
            train_mask = pd.Series([True] * len(train_df))
            test_mask = pd.Series([True] * len(test_df))
        else:
            target_cbus = cumulative_groups[group_name]
            train_mask = train_df['cbu_category'].isin(target_cbus)
            test_mask = test_df['cbu_category'].isin(target_cbus)
        
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"    跳过: 无数据")
            continue
        
        X_train_group = X_train_full.loc[train_mask.values].values
        X_test_group = X_test_full.loc[test_mask.values].values
        y_train_group = y_train_full[train_mask.values]
        y_test_group = y_test_full[test_mask.values]
        
        n_classes = len(np.unique(y_train_group))
        
        # 加权采样
        sample_weights = np.ones(len(y_train_group))
        for idx, label in enumerate(y_train_group):
            class_name = le_full.inverse_transform([label])[0]
            if class_name in train_counts:
                n = train_counts[class_name]
                weight = np.sqrt(500 / max(n, 1))
                sample_weights[idx] = min(max(weight, 0.5), 10.0)
        
        dtrain = xgb.DMatrix(X_train_group, label=y_train_group, weight=sample_weights)
        dtest = xgb.DMatrix(X_test_group, label=y_test_group)
        
        params = {
            'objective': 'multi:softprob',
            'num_class': max(n_classes, int(y_train_group.max()) + 1),
            'max_depth': config['max_depth'],
            'learning_rate': config['lr'],
            'seed': 42,
            'tree_method': 'hist',
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample'],
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        model = xgb.train(params, dtrain, num_boost_round=config['rounds'], verbose_eval=False)
        
        # 预测
        y_pred = np.argmax(model.predict(dtest), axis=1)
        
        # 计算指标
        acc = accuracy_score(y_test_group, y_pred)
        f1_w = f1_score(y_test_group, y_pred, average='weighted', zero_division=0)
        f1_m = f1_score(y_test_group, y_pred, average='macro', zero_division=0)
        prec_w = precision_score(y_test_group, y_pred, average='weighted', zero_division=0)
        rec_w = recall_score(y_test_group, y_pred, average='weighted', zero_division=0)
        
        print(f"    样本数: {len(X_train_group)}, Code1类别数: {n_classes}")
        print(f"    Acc: {acc:.4f}, F1-W: {f1_w:.4f}, F1-M: {f1_m:.4f}")
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        all_feature_importance[group_name] = importance
        
        # 保存模型
        all_models[group_name] = {
            'model': model,
            'label_encoder': le_full,
            'features': available,
            'config': config,
            'n_classes': n_classes,
            'target_cbus': target_cbus
        }
        
        all_results[group_name] = {
            'n_train_samples': int(len(X_train_group)),
            'n_test_samples': int(len(X_test_group)),
            'n_code1_classes': n_classes,
            'accuracy': float(acc),
            'f1_weighted': float(f1_w),
            'f1_macro': float(f1_m),
            'precision_weighted': float(prec_w),
            'recall_weighted': float(rec_w)
        }
    
    print("\n[3/8] 输出总体结果...")
    
    print("\n" + "=" * 80)
    print("【分组预训练模型结果汇总】")
    print("=" * 80)
    print(f"{'模型':<15} {'训练样本':<10} {'测试样本':<10} {'Code1类':<8} {'Accuracy':<12} {'F1-W':<10} {'F1-M':<10}")
    print("-" * 80)
    
    for group_name in ['G1', 'G2', 'G3', 'G4', 'G5']:
        if group_name in all_results:
            r = all_results[group_name]
            print(f"{group_name:<15} {r['n_train_samples']:<10} {r['n_test_samples']:<10} {r['n_code1_classes']:<8} {r['accuracy']:<12.4f} {r['f1_weighted']:<10.4f} {r['f1_macro']:<10.4f}")
    
    print("\n[4/8] 输出Feature Importance...")
    
    # 合并feature importance
    all_importance = {}
    for group_name, importance in all_feature_importance.items():
        for feat, imp in importance.items():
            if feat not in all_importance:
                all_importance[feat] = {}
            all_importance[feat][group_name] = imp
    
    importance_df = pd.DataFrame(all_importance).T.fillna(0)
    importance_df['Total'] = importance_df.sum(axis=1)
    importance_df = importance_df.sort_values('Total', ascending=False)
    
    print("\n  Top 20 重要特征:")
    print(f"  {'特征':<40} {'G1':<10} {'G2':<10} {'G3':<10} {'G4':<10} {'G5':<10} {'总计':<10}")
    print("  " + "-" * 100)
    for idx, row in importance_df.head(20).iterrows():
        g1 = row.get('G1', 0)
        g2 = row.get('G2', 0)
        g3 = row.get('G3', 0)
        g4 = row.get('G4', 0)
        g5 = row.get('G5', 0)
        print(f"  {idx:<40} {g1:<10.2f} {g2:<10.2f} {g3:<10.2f} {g4:<10.2f} {g5:<10.2f} {row['Total']:<10.2f}")
    
    print("\n[5/8] 保存模型...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for group_name, model_data in all_models.items():
        model_file = MODELS_DIR / f'pretrain_{group_name.lower()}.pkl'
        joblib.dump(model_data, model_file)
        print(f"  保存: {model_file}")
    
    # 保存累积分组定义
    cumulative_def = {k: (v if v != 'ALL' else 'ALL_DATA') for k, v in cumulative_groups.items()}
    cumulative_def['G5'] = 'ALL_DATA'
    
    print("\n[6/8] 保存结果...")
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    results_json = {
        'summary': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_models': 5,
            'cumulative_groups': {
                'G1': 'G1_VeryHigh',
                'G2': 'G1_VeryHigh + G2_High',
                'G3': 'G1_VeryHigh + G2_High + G3_Medium',
                'G4': 'G1_VeryHigh + G2_High + G3_Medium + G4_Low',
                'G5': 'ALL_DATA'
            },
            'group_definitions': CBU_GROUPS,
            'model_configs': model_configs,
            'feature_count': len(available),
            'prediction_target': 'Code1',
            'grouping_column': 'cbu_category'
        },
        'results': all_results,
        'feature_importance': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_feature_importance.items()}
    }
    
    with open(RESULTS_DIR / 'stage3_grouped_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    results_df = pd.DataFrame([
        {
            'Group': k,
            'N_Train_Samples': v['n_train_samples'],
            'N_Test_Samples': v['n_test_samples'],
            'N_Code1_Classes': v['n_code1_classes'],
            'Accuracy': v['accuracy'],
            'Precision_Weighted': v['precision_weighted'],
            'Recall_Weighted': v['recall_weighted'],
            'F1_Weighted': v['f1_weighted'],
            'F1_Macro': v['f1_macro']
        }
        for k, v in all_results.items()
    ])
    results_df.to_csv(RESULTS_DIR / 'stage3_grouped_results.csv', index=False)
    importance_df.to_csv(RESULTS_DIR / 'stage3_feature_importance.csv')
    
    print("\n[7/8] 生成报告...")
    
    REPORTS_DIR.mkdir(exist_ok=True)
    
    report = f"""# TLCBU_vGlobe 阶段3: 累积分组预训练模型报告

**项目**: 基于CBU相似度分组的累积迁移学习预训练  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**方法**: 累积分组预训练（5个模型）

---

## 1. 数据结构说明

- **分组索引列**: `cbu_category` (小写三个字母，如 'mel', 'afi', 'd4r')
- **预测目标列**: `Code1` (骨架类型，如 'MFI', 'AFI', 'UTL')

---

## 2. CBU分组定义与累积策略

根据CBU相似度分析，将CBU分为4个基础组：

### 2.1 基础分组
| 分组 | CBU列表 | 数量 |
|------|---------|------|
| G1_VeryHigh | {', '.join(CBU_GROUPS['G1_VeryHigh'])} | {len(CBU_GROUPS['G1_VeryHigh'])}个 |
| G2_High | {', '.join(CBU_GROUPS['G2_High'])} | {len(CBU_GROUPS['G2_High'])}个 |
| G3_Medium | {', '.join(CBU_GROUPS['G3_Medium'])} | {len(CBU_GROUPS['G3_Medium'])}个 |
| G4_Low | {', '.join(CBU_GROUPS['G4_Low'])} | {len(CBU_GROUPS['G4_Low'])}个 |

### 2.2 累积分组策略
| 模型 | 数据集 | 包含CBU数量 |
|------|--------|-----------|
| G1 | G1_VeryHigh | {len(CBU_GROUPS['G1_VeryHigh'])} |
| G2 | G1 + G2 | {len(CBU_GROUPS['G1_VeryHigh'] + CBU_GROUPS['G2_High'])} |
| G3 | G1 + G2 + G3 | {len(CBU_GROUPS['G1_VeryHigh'] + CBU_GROUPS['G2_High'] + CBU_GROUPS['G3_Medium'])} |
| G4 | G1 + G2 + G3 + G4 | {len(CBU_GROUPS['G1_VeryHigh'] + CBU_GROUPS['G2_High'] + CBU_GROUPS['G3_Medium'] + CBU_GROUPS['G4_Low'])} |
| G5 | 全部数据 | 所有cbu_category |

---

## 3. 模型性能结果

### 3.1 总体性能汇总

| 模型 | 数据集 | 训练样本 | 测试样本 | Code1类别 | Accuracy | Precision | Recall | F1-Weighted | F1-Macro |
|------|--------|---------|---------|----------|----------|-----------|--------|-------------|----------|
"""
    
    for group_name in ['G1', 'G2', 'G3', 'G4', 'G5']:
        if group_name in all_results:
            r = all_results[group_name]
            dataset_desc = cumulative_groups[group_name] if cumulative_groups[group_name] != 'ALL' else 'ALL_DATA'
            report += f"| {group_name} | {dataset_desc} | {r['n_train_samples']} | {r['n_test_samples']} | {r['n_code1_classes']} | {r['accuracy']:.4f} | {r['precision_weighted']:.4f} | {r['recall_weighted']:.4f} | {r['f1_weighted']:.4f} | {r['f1_macro']:.4f} |\n"
    
    report += f"""
### 3.2 最佳模型分析

- **最高Accuracy**: {max(all_results.values(), key=lambda x: x['accuracy'])['accuracy']:.4f}
- **最高F1-Weighted**: {max(all_results.values(), key=lambda x: x['f1_weighted'])['f1_weighted']:.4f}
- **最高F1-Macro**: {max(all_results.values(), key=lambda x: x['f1_macro'])['f1_macro']:.4f}

---

## 4. Feature Importance 分析

### 4.1 Top 20 重要特征

| 排名 | 特征 | G1 | G2 | G3 | G4 | G5 | 总计 |
|------|------|----|----|----|----|----|------|
"""
    
    for rank, (idx, row) in enumerate(importance_df.head(20).iterrows(), 1):
        g1 = row.get('G1', 0)
        g2 = row.get('G2', 0)
        g3 = row.get('G3', 0)
        g4 = row.get('G4', 0)
        g5 = row.get('G5', 0)
        report += f"| {rank} | {idx} | {g1:.2f} | {g2:.2f} | {g3:.2f} | {g4:.2f} | {g5:.2f} | {row['Total']:.2f} |\n"
    
    report += f"""
---

## 5. 输出文件

| 文件类型 | 路径 |
|---------|------|
| 模型文件 | `models/pretrain/pretrain_g1.pkl` ~ `pretrain_g5.pkl` |
| 结果JSON | `results/stage3_grouped_results.json` |
| 结果CSV | `results/stage3_grouped_results.csv` |
| Feature Importance | `results/stage3_feature_importance.csv` |
| 本报告 | `reports/stage3_grouped_report.md` |

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(REPORTS_DIR / 'stage3_grouped_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n[8/8] 完成!")
    
    print("\n" + "=" * 80)
    print("阶段3完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
