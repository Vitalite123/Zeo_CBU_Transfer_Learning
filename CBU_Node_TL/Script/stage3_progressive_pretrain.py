"""
阶段3 v2: 基于CBU类别的真实渐进式预训练
============================================

实际使用cbu_category筛选训练数据，实现真正的预训练策略
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义路径
DATA_DIR = Path(__file__).parent.parent
DATA_PATH = DATA_DIR / 'Data_CBU_TL_clean.csv'
CBU_GROUPING_PATH = Path(__file__).parent / 'results' / 'similarity_groups' / 'cbu_similarity_groups_v2.csv'
OUTPUT_DIR = Path(__file__).parent
MODELS_DIR = OUTPUT_DIR / 'models' / 'pretrain_v2'
RESULTS_DIR = OUTPUT_DIR / 'results' / 'pretrain_v2'
REPORTS_DIR = OUTPUT_DIR / 'reports'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("阶段3 v2: 基于CBU类别的真实渐进式预训练")
print("="*80)

# ============================================================================
# 步骤1: 定义104个特征和CBU分组
# ============================================================================
print("\n[步骤1] 加载特征和CBU分组...")

# 加载CBU分组
cbu_grouping = pd.read_csv(CBU_GROUPING_PATH)
cbu_to_group = dict(zip(cbu_grouping['CBU'], cbu_grouping['Group']))

# 定义104个特征
elemental_composition_features = [
    'Si', 'Al', 'P', 'Na', 'K', 'Li', 'Sr', 'Rb', 'Cs', 'Ba', 'Ca', 'F', 'Ge', 'Ti', 'In',
    'B', 'Mg', 'Ga', 'Ni', 'Mn', 'Fe', 'Co', 'Cr', 'Zn', 'Nb', 'Be', 'W', 'Ce', 'Cu',
    'Sn', 'Gd', 'La', 'Y', 'Dy', 'Sm', 'Ag', 'Cd', 'Zr', 'V', 'Ta', 'Ru', 'Hf', 'Yb', 'Tl', 'As'
]
osda_index_features = ['osda1_index', 'osda2_index', 'osda3_index']
synthesis_condition_features = ['cryst_temp', 'cryst_time', 'seed', 'rotation']
aging_condition_features = ['aging_temp', 'aging_time']
ph_condition_features = ['acid', 'OH']
gel_ratio_features = ['H2O_T', 'OH_T', 'Gel_Si_Al', 'Gel_P_Al', 'Gel_P_Si']
osda_descriptors = []
descriptor_bases = [
    'bertz_ct_mean_0', 'free_sasa_mean_0', 'asphericity_mean_0',
    'eccentricity_mean_0', 'axes_mean_0', 'axes_mean_1',
    'box_mean_0', 'box_mean_1', 'box_mean_2',
    'getaway_mean_0', 'getaway_mean_1'
]
for osda_num in [1, 2, 3]:
    for desc in descriptor_bases:
        osda_descriptors.append(f'osda{osda_num}_{desc}')
aggregated_features = [
    'osda_avg_asphericity', 'osda_max_asphericity', 'osda_min_asphericity',
    'osda_avg_sasa', 'osda_max_sasa', 'osda_min_sasa',
    'osda_avg_bertz', 'osda_max_bertz', 'osda_min_bertz',
    'osda_total_volume'
]
features_104 = (
    elemental_composition_features + osda_index_features + synthesis_condition_features +
    aging_condition_features + ph_condition_features + gel_ratio_features +
    osda_descriptors + aggregated_features
)

print(f"✓ 104个特征定义完成")
print(f"✓ CBU分组加载完成: {cbu_grouping.shape[0]}个CBU")

# ============================================================================
# 步骤2: 加载数据集
# ============================================================================
print("\n[步骤2] 加载数据集...")

df = pd.read_csv(DATA_PATH, low_memory=False)
X = df[features_104].copy().fillna(0)
y = df['Code1'].copy()
cbu_categories = df['cbu_category'].copy()

print(f"✓ 数据集加载成功: {df.shape[0]}行 × {df.shape[1]}列")
print(f"✓ cbu_category列存在，可用于筛选")

# ============================================================================
# 步骤3: 定义预训练阶段（基于CBU分组）
# ============================================================================
print("\n[步骤3] 定义预训练阶段...")

pretrain_stages = {
    'stage1': {
        'name': 'Stage 1: 小Node值CBU预训练',
        'groups': ['G1', 'G2', 'G3'],
        'cbus': ['lov', 'd3r', 'nat', 'vsv', 'd4r', 'mor', 'sti'],
        'description': 'Node值5-12，6个CBU'
    },
    'stage2': {
        'name': 'Stage 2: 中Node值CBU预训练',
        'groups': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
        'cbus': ['lov', 'd3r', 'nat', 'vsv', 'd4r', 'mor', 'sti', 
                'bea', 'bre', 'jbw', 'mtt',
                'afi', 'afs', 'ats', 'bog', 'cas', 'd6r', 'lau', 'rth', 'stf'],
        'description': 'Node值5-16，24个CBU'
    },
    'stage3': {
        'name': 'Stage 3: 大部分CBU预训练',
        'groups': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'],
        'cbus': ['lov', 'd3r', 'nat', 'vsv', 'd4r', 'mor', 'sti',
                'bea', 'bre', 'jbw', 'mtt',
                'afi', 'afs', 'ats', 'bog', 'cas', 'd6r', 'lau', 'rth', 'stf',
                'bik', 'fer',
                'abw', 'bph', 'mel', 'mfi', 'mtw',
                'non', 'ton',
                'aww', 'd8r', 'ddr', 'imf', 'rte'],
        'description': 'Node值5-20，31个CBU'
    },
    'stage4': {
        'name': 'Stage 4: 完整CBU预训练',
        'groups': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 
                   'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18'],
        'cbus': cbu_grouping['CBU'].tolist(),
        'description': 'Node值5-48，58个CBU'
    }
}

for stage_name, stage_info in pretrain_stages.items():
    print(f"  {stage_name}: {len(stage_info['cbus'])}个CBU - {stage_info['description']}")

# ============================================================================
# 步骤4: 准备数据
# ============================================================================
print("\n[步骤4] 准备数据...")

# 数据集划分（与阶段2相同）
unique_code1s = y.value_counts().index.tolist()
train_indices = []
test_indices = []

for code1 in unique_code1s:
    code1_indices = np.where(y == code1)[0]
    n_samples = len(code1_indices)
    if n_samples == 1:
        train_indices.extend(code1_indices)
    elif n_samples == 2:
        train_indices.extend(code1_indices[:1])
        test_indices.extend(code1_indices[1:])
    else:
        n_train = int(n_samples * 0.8)
        train_indices.extend(code1_indices[:n_train])
        test_indices.extend(code1_indices[n_train:])

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

X_train_full = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)
y_train_full = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)
cbu_train_full = cbu_categories.iloc[train_indices].reset_index(drop=True)

# 划分验证集
val_indices = []
train_train_indices = []
for code1 in y_train_full.unique():
    code1_indices = np.where(y_train_full == code1)[0]
    n_samples = len(code1_indices)
    if n_samples == 1:
        train_train_indices.extend(code1_indices)
    elif n_samples == 2:
        train_train_indices.extend(code1_indices[:1])
        val_indices.extend(code1_indices[1:])
    else:
        n_train = int(n_samples * 0.9)
        train_train_indices.extend(code1_indices[:n_train])
        val_indices.extend(code1_indices[n_train:])

train_train_indices = np.array(train_train_indices)
val_indices = np.array(val_indices)

X_train_full = X_train_full.iloc[train_train_indices].reset_index(drop=True)
X_val = X_train_full.iloc[val_indices].reset_index(drop=True)
y_train_full = y_train_full.iloc[train_train_indices].reset_index(drop=True)
y_val = y_train_full.iloc[val_indices].reset_index(drop=True)
cbu_train_full = cbu_train_full.iloc[train_train_indices].reset_index(drop=True)
cbu_val = cbu_train_full.iloc[val_indices].reset_index(drop=True)

print(f"✓ 训练集: {X_train_full.shape[0]}样本, {y_train_full.nunique()}个类别")
print(f"✓ 验证集: {X_val.shape[0]}样本, {y_val.nunique()}个类别")
print(f"✓ 测试集: {X_test.shape[0]}样本, {y_test.nunique()}个类别")

# 标签编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_full)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print(f"✓ 标签编码完成: {len(label_encoder.classes_)}个类别")

# ============================================================================
# 步骤5: 渐进式预训练（使用CBU筛选）
# ============================================================================
print("\n[步骤5] 渐进式预训练（使用CBU筛选）...")

xgb_params = {
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

stage_results = {}

for stage_name, stage_info in pretrain_stages.items():
    print(f"\n{'='*60}")
    print(f"{stage_info['name']}")
    print(f"{'='*60}")
    
    stage_cbus = stage_info['cbus']
    print(f"CBU列表: {', '.join(stage_cbus[:5])}{'...' if len(stage_cbus) > 5 else ''}")
    print(f"CBU数量: {len(stage_cbus)}")
    
    # 筛选属于当前阶段CBU的训练数据
    mask_train = cbu_train_full.isin(stage_cbus)
    mask_val = cbu_val.isin(stage_cbus)
    
    X_train_stage = X_train_full[mask_train].reset_index(drop=True)
    y_train_stage = y_train_full[mask_train].reset_index(drop=True)
    y_train_stage_encoded = label_encoder.transform(y_train_stage)
    
    X_val_stage = X_val[mask_val].reset_index(drop=True)
    y_val_stage = y_val[mask_val].reset_index(drop=True)
    y_val_stage_encoded = label_encoder.transform(y_val_stage)
    
    print(f"筛选后训练集: {X_train_stage.shape[0]}样本")
    print(f"筛选后验证集: {X_val_stage.shape[0]}样本")
    
    if X_train_stage.shape[0] == 0:
        print("⚠️  警告: 筛选后训练集为空，跳过此阶段")
        continue
    
    # 训练模型
    print("训练模型...")
    model = xgb.XGBClassifier(**xgb_params)
    
    model.fit(
        X_train_stage, y_train_stage_encoded,
        eval_set=[(X_val_stage, y_val_stage_encoded)],
        verbose=False
    )
    
    print("✓ 模型训练完成")
    
    # 评估模型（在完整测试集上）
    y_test_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_test_pred)
    precision_macro = precision_score(y_test_encoded, y_test_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test_encoded, y_test_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test_encoded, y_test_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\n测试集性能:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    print(f"  F1-Weighted: {f1_weighted:.4f}")
    
    # 保存模型
    model_path = MODELS_DIR / f'{stage_name}_model.pkl'
    joblib.dump({
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': features_104,
        'params': xgb_params,
        'stage_info': stage_info
    }, model_path)
    print(f"✓ 模型已保存: {model_path}")
    
    # 保存结果
    stage_results[stage_name] = {
        'stage_name': stage_info['name'],
        'cbus': stage_cbus,
        'n_cbus': len(stage_cbus),
        'groups': stage_info['groups'],
        'description': stage_info['description'],
        'train_samples_filtered': X_train_stage.shape[0],
        'val_samples_filtered': X_val_stage.shape[0],
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'model_path': str(model_path)
    }

# ============================================================================
# 步骤6: 保存结果
# ============================================================================
print("\n[步骤6] 保存结果...")

import json
results_json_path = RESULTS_DIR / 'stage_results.json'
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(stage_results, f, indent=2, ensure_ascii=False)
print(f"✓ 结果JSON已保存: {results_json_path}")

summary_df = pd.DataFrame(stage_results).T
summary_csv_path = RESULTS_DIR / 'pretrain_summary.csv'
summary_df.to_csv(summary_csv_path)
print(f"✓ 结果CSV已保存: {summary_csv_path}")

# ============================================================================
# 步骤7: 生成详细报告
# ============================================================================
print("\n[步骤7] 生成详细报告...")

report_path = REPORTS_DIR / 'pretrain_v2_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 阶段3 v2: 基于CBU类别的渐进式预训练报告\n\n")
    f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## 1. 概述\n\n")
    f.write("本报告详细描述了基于CBU类别的渐进式预训练过程和性能评估结果。\n\n")
    f.write("**方法**: 基于CBU类别筛选的渐进式预训练\n")
    f.write(f"**特征数**: {len(features_104)}\n")
    f.write(f"**类别数**: {len(label_encoder.classes_)}\n")
    f.write(f"**CBU组数**: 18\n\n")

    f.write("## 2. 预训练阶段设计\n\n")
    f.write("### 2.1 阶段划分\n\n")
    
    for stage_name, stage_info in pretrain_stages.items():
        if stage_name in stage_results:
            result = stage_results[stage_name]
            f.write(f"### {result['stage_name']}\n\n")
            f.write(f"- **CBU组**: {', '.join(result['groups'])}\n")
            f.write(f"- **CBU数量**: {result['n_cbus']}\n")
            f.write(f"- **Node范围**: {result['description']}\n")
            f.write(f"- **筛选后训练集**: {result['train_samples_filtered']}样本\n")
            f.write(f"- **筛选后验证集**: {result['val_samples_filtered']}样本\n\n")

    f.write("### 2.2 预训练策略\n\n")
    f.write("**核心思想**:\n")
    f.write("- 从小Node值的CBU开始，只使用这些CBU的样本进行预训练\n")
    f.write("- 逐步扩展到更多CBU，使用更多CBU的样本\n")
    f.write("- 每个阶段在完整的测试集上评估性能\n\n")

    f.write("## 3. 预训练结果\n\n")
    f.write("### 3.1 各阶段性能对比\n\n")
    f.write("| 阶段 | CBU数量 | 筛选训练集 | 筛选验证集 | Accuracy | F1-Macro | F1-Weighted |\n")
    f.write("|------|---------|-----------|-----------|----------|----------|-------------|\n")
    
    for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
        if stage_name in stage_results:
            result = stage_results[stage_name]
            f.write(f"| {result['stage_name']} | {result['n_cbus']} | {result['train_samples_filtered']} | {result['val_samples_filtered']} | {result['accuracy']:.4f} | {result['f1_macro']:.4f} | {result['f1_weighted']:.4f} |\n")
    f.write("\n")

    f.write("### 3.2 性能分析\n\n")
    f.write("**关键发现**:\n")
    
    if len(stage_results) > 0:
        stage_names = list(stage_results.keys())
        if len(stage_names) >= 2:
            first_stage = stage_names[0]
            last_stage = stage_names[-1]
            acc_improvement = (stage_results[last_stage]['accuracy'] - stage_results[first_stage]['accuracy']) * 100
            f1_improvement = (stage_results[last_stage]['f1_macro'] - stage_results[first_stage]['f1_macro']) * 100
            
            f.write(f"- Accuracy提升: {acc_improvement:+.2f}% (Stage 1 → Stage 4)\n")
            f.write(f"- F1-Macro提升: {f1_improvement:+.2f}% (Stage 1 → Stage 4)\n\n")

    f.write("## 4. 结论\n\n")
    f.write("### 4.1 主要发现\n\n")
    
    if len(stage_results) > 0:
        last_stage = stage_names[-1]
        result = stage_results[last_stage]
        f.write(f"1. **最终性能**: Stage 4达到{result['accuracy']*100:.2f}%准确率\n")
        f.write(f"2. **F1-Macro**: {result['f1_macro']*100:.2f}%\n")
        f.write(f"3. **F1-Weighted**: {result['f1_weighted']*100:.2f}%\n")
        f.write(f"4. **CBU覆盖**: 从6个CBU扩展到58个CBU\n\n")

    f.write("### 4.2 改进方向\n\n")
    f.write("- 优化CBU分组策略\n")
    f.write("- 尝试不同的预训练阶段划分\n")
    f.write("- 探索基于相似度的知识迁移\n")
    f.write("- 考虑使用样本数而非Node值进行分组\n\n")

print(f"✓ 详细报告已保存: {report_path}")

# ============================================================================
# 步骤8: 打印总结
# ============================================================================
print("\n" + "="*80)
print("渐进式预训练总结")
print("="*80)
print(f"特征数: {len(features_104)}")
print(f"类别数: {len(label_encoder.classes_)}")
print(f"CBU组数: 18")
print()
print("预训练阶段:")
for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
    if stage_name in stage_results:
        result = stage_results[stage_name]
        print(f"  {result['stage_name']}: {result['n_cbus']}个CBU, 筛选训练集={result['train_samples_filtered']}, Acc={result['accuracy']:.2%}, F1-M={result['f1_macro']:.2%}")
print()
print("输出文件:")
print(f"  - {MODELS_DIR}/")
print(f"  - {results_json_path}")
print(f"  - {summary_csv_path}")
print(f"  - {report_path}")
print("="*80)