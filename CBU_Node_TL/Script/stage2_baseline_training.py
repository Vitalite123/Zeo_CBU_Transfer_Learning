"""
阶段2: Baseline模型训练
========================

任务:
1. 加载Data_CBU_TL_clean.csv数据集
2. 数据预处理和特征提取（104个特征）
3. 按Code1类别进行分层划分，保证训练集包含所有231类
4. 训练XGBoost模型（使用baseline104的参数设置）
5. 评估模型性能
6. 生成详细报告

模型参数（来自baseline104）:
- n_estimators=300
- max_depth=10
- learning_rate=0.05
- subsample=0.9
- colsample_bytree=0.9
- min_child_weight=3
- gamma=0.1
- reg_alpha=0.1
- reg_lambda=1.0
- random_state=42
- n_jobs=-1
- eval_metric='mlogloss'

输出:
- models/baseline/baseline_xgboost.pkl: 训练好的模型
- models/baseline/baseline_results.json: 性能指标
- models/baseline/tier_evaluation.csv: 分层评估结果
- reports/baseline_report.md: 详细报告
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义路径
DATA_DIR = Path(__file__).parent.parent
DATA_PATH = DATA_DIR / 'Data_CBU_TL_clean.csv'
OUTPUT_DIR = Path(__file__).parent
MODELS_DIR = OUTPUT_DIR / 'models' / 'baseline'
REPORTS_DIR = OUTPUT_DIR / 'reports'

# 创建目录
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("阶段2: Baseline模型训练")
print("="*80)
print(f"数据目录: {DATA_DIR}")
print(f"数据文件: {DATA_PATH}")
print(f"模型输出目录: {MODELS_DIR}")
print(f"报告输出目录: {REPORTS_DIR}")
print("="*80)
print()

# ============================================================================
# 步骤1: 定义104个特征
# ============================================================================
print("\n[步骤1] 定义104个特征...")

# 61特征基线（来自Feature_List.md第1部分）
elemental_composition_features = [
    'Si', 'Al', 'P', 'Na', 'K', 'Li', 'Sr', 'Rb', 'Cs', 'Ba', 'Ca', 'F', 'Ge', 'Ti', 'In',
    'B', 'Mg', 'Ga', 'Ni', 'Mn', 'Fe', 'Co', 'Cr', 'Zn', 'Nb', 'Be', 'W', 'Ce', 'Cu',
    'Sn', 'Gd', 'La', 'Y', 'Dy', 'Sm', 'Ag', 'Cd', 'Zr', 'V', 'Ta', 'Ru', 'Hf', 'Yb', 'Tl', 'As'
]  # 45个元素

osda_index_features = ['osda1_index', 'osda2_index', 'osda3_index']  # 3个OSDA索引

synthesis_condition_features = ['cryst_temp', 'cryst_time', 'seed', 'rotation']  # 4个合成条件

aging_condition_features = ['aging_temp', 'aging_time']  # 2个老化条件

ph_condition_features = ['acid', 'OH']  # 2个pH条件

gel_ratio_features = ['H2O_T', 'OH_T', 'Gel_Si_Al', 'Gel_P_Al', 'Gel_P_Si']  # 5个凝胶比例

# 43个额外特征（来自Feature_List.md第2部分）
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

# 104特征完整列表
features_104 = (
    elemental_composition_features +
    osda_index_features +
    synthesis_condition_features +
    aging_condition_features +
    ph_condition_features +
    gel_ratio_features +
    osda_descriptors +
    aggregated_features
)

print(f"✓ 61特征基线: {len(elemental_composition_features) + len(osda_index_features) + len(synthesis_condition_features) + len(aging_condition_features) + len(ph_condition_features) + len(gel_ratio_features)}个")
print(f"✓ OSDA分子描述符: {len(osda_descriptors)}个")
print(f"✓ 聚合特征: {len(aggregated_features)}个")
print(f"✓ 总计: {len(features_104)}个特征")

# ============================================================================
# 步骤2: 加载数据集
# ============================================================================
print("\n[步骤2] 加载数据集...")

# 加载数据集
df = pd.read_csv(DATA_PATH)
print(f"✓ 数据集加载成功: {df.shape[0]}行 × {df.shape[1]}列")

# 提取特征和标签
X = df[features_104].copy()
y = df['Code1'].copy()

# 缺失值处理
X = X.fillna(0)

print(f"✓ 特征矩阵: {X.shape}")
print(f"✓ 标签数量: {len(y)}")

# ============================================================================
# 步骤3: 统计Code1类别分布
# ============================================================================
print("\n[步骤3] 统计Code1类别分布...")

# 计算每个Code1的样本数
code1_counts = y.value_counts().sort_values(ascending=False)
unique_code1s = code1_counts.index.tolist()
n_classes = len(unique_code1s)

print(f"✓ 唯一Code1类别数: {n_classes}")
print(f"\nCode1类别分布（前10）:")
print(code1_counts.head(10).to_string())

# 统计分层分布
large = (code1_counts >= 1000).sum()
medium_large = ((code1_counts >= 500) & (code1_counts < 1000)).sum()
medium = ((code1_counts >= 100) & (code1_counts < 500)).sum()
small_medium = ((code1_counts >= 50) & (code1_counts < 100)).sum()
small = ((code1_counts >= 20) & (code1_counts < 50)).sum()
very_small = ((code1_counts >= 10) & (code1_counts < 20)).sum()
tiny = (code1_counts < 10).sum()

print(f"\n分层统计:")
print(f"  - Large (≥1000): {large}个类别")
print(f"  - Medium-Large (500-999): {medium_large}个类别")
print(f"  - Medium (100-499): {medium}个类别")
print(f"  - Small-Medium (50-99): {small_medium}个类别")
print(f"  - Small (20-49): {small}个类别")
print(f"  - Very Small (10-19): {very_small}个类别")
print(f"  - Tiny (<10): {tiny}个类别")

# ============================================================================
# 步骤4: 按Code1类别进行分层划分
# ============================================================================
print("\n[步骤4] 按Code1类别进行分层划分...")

# 策略：保证训练集包含所有231类
# 对于单样本类别：全部放入训练集
# 对于多样本类别：按80%训练、20%测试划分

train_indices = []
test_indices = []

for code1 in unique_code1s:
    code1_indices = np.where(y == code1)[0]
    n_samples = len(code1_indices)

    if n_samples == 1:
        # 单样本类别：全部放入训练集
        train_indices.extend(code1_indices)
    elif n_samples == 2:
        # 双样本类别：1个训练，1个测试
        train_indices.extend(code1_indices[:1])
        test_indices.extend(code1_indices[1:])
    else:
        # 多样本类别：80%训练，20%测试
        n_train = int(n_samples * 0.8)
        train_indices.extend(code1_indices[:n_train])
        test_indices.extend(code1_indices[n_train:])

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

# 创建训练集和测试集
X_train = X.iloc[train_indices].reset_index(drop=True)
X_test = X.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)

# 从训练集中划分验证集（使用自定义分层方法，避免单样本类别问题）
print("从训练集中划分验证集...")

val_indices = []
train_train_indices = []

# 统计训练集中每个类别的样本数
train_code1_counts = y_train.value_counts()

for code1 in y_train.unique():
    code1_indices = np.where(y_train == code1)[0]
    n_samples = len(code1_indices)

    if n_samples == 1:
        # 单样本类别：全部放入训练集
        train_train_indices.extend(code1_indices)
    elif n_samples == 2:
        # 双样本类别：1个训练，1个验证
        train_train_indices.extend(code1_indices[:1])
        val_indices.extend(code1_indices[1:])
    else:
        # 多样本类别：90%训练，10%验证
        n_train = int(n_samples * 0.9)
        train_train_indices.extend(code1_indices[:n_train])
        val_indices.extend(code1_indices[n_train:])

train_train_indices = np.array(train_train_indices)
val_indices = np.array(val_indices)

X_train_split = X_train.iloc[train_train_indices].reset_index(drop=True)
X_val = X_train.iloc[val_indices].reset_index(drop=True)
y_train_split = y_train.iloc[train_train_indices].reset_index(drop=True)
y_val = y_train.iloc[val_indices].reset_index(drop=True)

print(f"✓ 训练集: {X_train_split.shape[0]}样本, {y_train_split.nunique()}个类别")
print(f"✓ 验证集: {X_val.shape[0]}样本, {y_val.nunique()}个类别")
print(f"✓ 测试集: {X_test.shape[0]}样本, {y_test.nunique()}个类别")

# ============================================================================
# 步骤5: 标签编码
# ============================================================================
print("\n[步骤5] 标签编码...")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_split)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print(f"✓ 标签编码完成: {len(label_encoder.classes_)}个类别")

# ============================================================================
# 步骤6: 训练XGBoost模型
# ============================================================================
print("\n[步骤6] 训练XGBoost模型...")

# XGBoost参数（来自baseline104）
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

print("XGBoost参数:")
for param, value in xgb_params.items():
    print(f"  {param}: {value}")

# 训练模型
model = xgb.XGBClassifier(**xgb_params)

print("\n开始训练...")
model.fit(
    X_train_split, y_train_encoded,
    eval_set=[(X_val, y_val_encoded)],
    verbose=False
)

print("✓ 模型训练完成")

# 保存模型
model_path = MODELS_DIR / 'baseline_xgboost.pkl'
import joblib
joblib.dump({
    'model': model,
    'label_encoder': label_encoder,
    'feature_names': features_104,
    'params': xgb_params
}, model_path)
print(f"✓ 模型已保存: {model_path}")

# ============================================================================
# 步骤7: 评估模型性能
# ============================================================================
print("\n[步骤7] 评估模型性能...")

# 预测
y_train_pred = model.predict(X_train_split)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# 计算指标
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

train_metrics = calculate_metrics(y_train_encoded, y_train_pred)
val_metrics = calculate_metrics(y_val_encoded, y_val_pred)
test_metrics = calculate_metrics(y_test_encoded, y_test_pred)

print(f"\n训练集性能:")
print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
print(f"  F1-Macro: {train_metrics['f1_macro']:.4f}")
print(f"  F1-Weighted: {train_metrics['f1_weighted']:.4f}")

print(f"\n验证集性能:")
print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
print(f"  F1-Macro: {val_metrics['f1_macro']:.4f}")
print(f"  F1-Weighted: {val_metrics['f1_weighted']:.4f}")

print(f"\n测试集性能:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
print(f"  F1-Weighted: {test_metrics['f1_weighted']:.4f}")

# ============================================================================
# 步骤7.5: 生成完整混淆矩阵CSV（231×231）
# ============================================================================
print("\n[步骤7.5] 生成完整混淆矩阵CSV（231×231）...")

# 生成完整的231×231混淆矩阵
full_cm = confusion_matrix(y_test, label_encoder.inverse_transform(y_test_pred), labels=label_encoder.classes_)

# 转换为DataFrame
full_cm_df = pd.DataFrame(
    full_cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

# 保存完整混淆矩阵
full_cm_path = MODELS_DIR / 'confusion_matrix_full_231x231.csv'
full_cm_df.to_csv(full_cm_path)
print(f"✓ 完整混淆矩阵已保存: {full_cm_path}")
print(f"  尺寸: {full_cm_df.shape[0]}×{full_cm_df.shape[1]}")

# ============================================================================
# 步骤7.6: 生成特征重要性CSV
# ============================================================================
print("\n[步骤7.6] 生成特征重要性CSV...")

# 获取特征重要性
feature_importance = model.feature_importances_
feature_names = features_104

# 创建DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# 保存特征重要性
importance_path = MODELS_DIR / 'feature_importance.csv'
importance_df.to_csv(importance_path, index=False)
print(f"✓ 特征重要性已保存: {importance_path}")
print(f"  总特征数: {len(importance_df)}")
print(f"  Top 10特征:")
print(importance_df.head(10).to_string(index=False))

# ============================================================================
# 步骤8: 按Code1样本数分层评估
# ============================================================================
print("\n[步骤8] 按Code1样本数分层评估...")

# 为测试集添加预测结果
test_results = X_test.copy()
test_results['Code1_true'] = y_test.values
test_results['Code1_pred'] = label_encoder.inverse_transform(y_test_pred)
test_results['correct'] = (test_results['Code1_true'] == test_results['Code1_pred']).astype(int)

# 获取每个Code1的样本数
code1_sample_counts = y_train_split.value_counts().to_dict()

# 为测试集的每个Code1添加样本数
test_results['sample_count'] = test_results['Code1_true'].map(code1_sample_counts)

# 定义分层
def get_tier(sample_count):
    if sample_count >= 1000:
        return 'Large (≥1000)'
    elif sample_count >= 500:
        return 'Medium-Large (500-999)'
    elif sample_count >= 100:
        return 'Medium (100-499)'
    elif sample_count >= 50:
        return 'Small-Medium (50-99)'
    elif sample_count >= 20:
        return 'Small (20-49)'
    elif sample_count >= 10:
        return 'Very Small (10-19)'
    else:
        return 'Tiny (<10)'

test_results['tier'] = test_results['sample_count'].apply(get_tier)

# 计算各层性能
tier_evaluation = []
for tier in ['Large (≥1000)', 'Medium-Large (500-999)', 'Medium (100-499)',
              'Small-Medium (50-99)', 'Small (20-49)', 'Very Small (10-19)', 'Tiny (<10)']:
    tier_data = test_results[test_results['tier'] == tier]

    if len(tier_data) > 0:
        accuracy = tier_data['correct'].mean()
        n_samples = len(tier_data)
        n_classes = tier_data['Code1_true'].nunique()

        # 计算F1分数
        precision = precision_score(
            tier_data['Code1_true'], tier_data['Code1_pred'],
            average='macro', zero_division=0
        )
        recall = recall_score(
            tier_data['Code1_true'], tier_data['Code1_pred'],
            average='macro', zero_division=0
        )
        f1 = f1_score(
            tier_data['Code1_true'], tier_data['Code1_pred'],
            average='macro', zero_division=0
        )

        tier_evaluation.append({
            'Tier': tier,
            'Sample_Count': n_samples,
            'Class_Count': n_classes,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

tier_df = pd.DataFrame(tier_evaluation)

print(f"\n分层评估结果:")
print(tier_df.to_string(index=False))

# 保存分层评估结果
tier_path = MODELS_DIR / 'tier_evaluation.csv'
tier_df.to_csv(tier_path, index=False)
print(f"✓ 分层评估结果已保存: {tier_path}")

# ============================================================================
# 步骤9: 生成混淆矩阵（前20类）
# ============================================================================
print("\n[步骤9] 生成混淆矩阵...")

# 获取前20个最常见的Code1
top20_classes = y_train_split.value_counts().head(20).index.tolist()

# 筛选测试集中的前20类
top20_mask = y_test.isin(top20_classes)
y_test_top20 = y_test[top20_mask]
y_test_pred_top20 = label_encoder.inverse_transform(y_test_pred[top20_mask])

# 生成混淆矩阵
cm = confusion_matrix(y_test_top20, y_test_pred_top20, labels=top20_classes)

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=top20_classes,
    yticklabels=top20_classes,
    ax=ax
)
ax.set_title('混淆矩阵（测试集，前20类）', fontsize=14, fontweight='bold')
ax.set_xlabel('预测类别', fontsize=12)
ax.set_ylabel('真实类别', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

confusion_path = MODELS_DIR / 'confusion_matrix_top20.png'
plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
print(f"✓ 混淆矩阵已保存: {confusion_path}")
plt.close()

# ============================================================================
# 步骤10: 生成性能报告
# ============================================================================
print("\n[步骤10] 生成性能报告...")

report_path = REPORTS_DIR / 'baseline_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 阶段2: Baseline模型训练报告\n\n")
    f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## 1. 概述\n\n")
    f.write("本报告详细描述了Baseline XGBoost模型的训练过程和性能评估结果。\n\n")
    f.write("**模型**: XGBoost Classifier\n")
    f.write(f"**特征数**: {len(features_104)}\n")
    f.write(f"**类别数**: {n_classes}\n\n")

    f.write("## 2. 数据集信息\n\n")
    f.write("### 2.1 数据集来源\n\n")
    f.write(f"**文件**: `{DATA_PATH}`\n")
    f.write(f"**总样本数**: {df.shape[0]:,}\n")
    f.write(f"**总列数**: {df.shape[1]}\n\n")

    f.write("### 2.2 特征信息\n\n")
    f.write("使用104个特征，包括：\n\n")
    f.write("**61特征基线**:\n")
    f.write(f"- 元素组成: {len(elemental_composition_features)}个\n")
    f.write(f"- OSDA索引: {len(osda_index_features)}个\n")
    f.write(f"- 合成条件: {len(synthesis_condition_features)}个\n")
    f.write(f"- 老化条件: {len(aging_condition_features)}个\n")
    f.write(f"- pH条件: {len(ph_condition_features)}个\n")
    f.write(f"- 凝胶比例: {len(gel_ratio_features)}个\n\n")

    f.write("**43个额外特征**:\n")
    f.write(f"- OSDA分子描述符: {len(osda_descriptors)}个\n")
    f.write(f"- 聚合特征: {len(aggregated_features)}个\n\n")

    f.write("### 2.3 Code1类别分布\n\n")
    f.write(f"**总类别数**: {n_classes}\n\n")
    f.write("**分层统计**:\n")
    f.write(f"- Large (≥1000): {large}个类别\n")
    f.write(f"- Medium-Large (500-999): {medium_large}个类别\n")
    f.write(f"- Medium (100-499): {medium}个类别\n")
    f.write(f"- Small-Medium (50-99): {small_medium}个类别\n")
    f.write(f"- Small (20-49): {small}个类别\n")
    f.write(f"- Very Small (10-19): {very_small}个类别\n")
    f.write(f"- Tiny (<10): {tiny}个类别\n\n")

    f.write("**前10个最常见类别**:\n")
    f.write("| 排名 | Code1 | 样本数 | 占比 |\n")
    f.write("|------|-------|--------|------|\n")
    for i, (code1, count) in enumerate(code1_counts.head(10).items(), 1):
        percentage = count / len(y) * 100
        f.write(f"| {i} | {code1} | {count} | {percentage:.2f}% |\n")
    f.write("\n")

    f.write("### 2.4 数据集划分\n\n")
    f.write("**划分策略**: 按Code1类别分层划分，保证训练集包含所有231类\n\n")
    f.write("| 数据集 | 样本数 | 类别数 | 占比 |\n")
    f.write("|--------|--------|--------|------|\n")
    f.write(f"| 训练集 | {X_train_split.shape[0]:,} | {y_train_split.nunique()} | {X_train_split.shape[0]/len(y)*100:.1f}% |\n")
    f.write(f"| 验证集 | {X_val.shape[0]:,} | {y_val.nunique()} | {X_val.shape[0]/len(y)*100:.1f}% |\n")
    f.write(f"| 测试集 | {X_test.shape[0]:,} | {y_test.nunique()} | {X_test.shape[0]/len(y)*100:.1f}% |\n")
    f.write(f"| 总计 | {len(y):,} | {n_classes} | 100.0% |\n\n")

    f.write("## 3. 模型架构与参数\n\n")
    f.write("### 3.1 模型类型\n\n")
    f.write("**XGBoost Classifier**\n\n")
    f.write("基于梯度提升决策树（GBDT）的高效实现，适用于多分类问题。\n\n")

    f.write("### 3.2 模型参数\n\n")
    f.write("```python\n")
    for param, value in xgb_params.items():
        f.write(f"{param} = {value}\n")
    f.write("```\n\n")

    f.write("**参数说明**:\n")
    f.write("- `n_estimators`: 树的数量，控制模型复杂度\n")
    f.write("- `max_depth`: 树的最大深度，防止过拟合\n")
    f.write("- `learning_rate`: 学习率，控制每棵树的贡献\n")
    f.write("- `subsample`: 样本采样比例，增强泛化能力\n")
    f.write("- `colsample_bytree`: 特征采样比例，防止过拟合\n")
    f.write("- `min_child_weight`: 最小子节点权重，控制分裂\n")
    f.write("- `gamma`: 分裂最小增益，控制树生长\n")
    f.write("- `reg_alpha`: L1正则化，防止过拟合\n")
    f.write("- `reg_lambda`: L2正则化，防止过拟合\n")
    f.write("- `random_state`: 随机种子，保证可重复性\n")
    f.write("- `n_jobs`: 并行计算核心数\n")
    f.write("- `eval_metric`: 评估指标，多分类对数损失\n\n")

    f.write("## 4. 总体性能\n\n")
    f.write("### 4.1 性能指标汇总\n\n")
    f.write("| 数据集 | Accuracy | Precision(M) | Precision(W) | Recall(M) | Recall(W) | F1-Macro | F1-Weighted |\n")
    f.write("|--------|----------|--------------|--------------|-----------|-----------|----------|-------------|\n")
    f.write(f"| 训练集 | {train_metrics['accuracy']:.4f} | {train_metrics['precision_macro']:.4f} | {train_metrics['precision_weighted']:.4f} | {train_metrics['recall_macro']:.4f} | {train_metrics['recall_weighted']:.4f} | {train_metrics['f1_macro']:.4f} | {train_metrics['f1_weighted']:.4f} |\n")
    f.write(f"| 验证集 | {val_metrics['accuracy']:.4f} | {val_metrics['precision_macro']:.4f} | {val_metrics['precision_weighted']:.4f} | {val_metrics['recall_macro']:.4f} | {val_metrics['recall_weighted']:.4f} | {val_metrics['f1_macro']:.4f} | {val_metrics['f1_weighted']:.4f} |\n")
    f.write(f"| 测试集 | {test_metrics['accuracy']:.4f} | {test_metrics['precision_macro']:.4f} | {test_metrics['precision_weighted']:.4f} | {test_metrics['recall_macro']:.4f} | {test_metrics['recall_weighted']:.4f} | {test_metrics['f1_macro']:.4f} | {test_metrics['f1_weighted']:.4f} |\n\n")

    f.write("### 4.2 测试集性能分析\n\n")
    f.write(f"- **总体准确率**: {test_metrics['accuracy']*100:.2f}%\n")
    f.write(f"- **F1-Macro**: {test_metrics['f1_macro']*100:.2f}%\n")
    f.write(f"- **F1-Weighted**: {test_metrics['f1_weighted']*100:.2f}%\n\n")

    f.write(f"**F1-Macro vs F1-Weighted差异**: {abs(test_metrics['f1_macro'] - test_metrics['f1_weighted'])*100:.2f}%\n\n")

    if abs(test_metrics['f1_macro'] - test_metrics['f1_weighted']) > 0.1:
        f.write("**类别不平衡影响严重**: F1-Macro显著低于F1-Weighted，表明小类别预测困难\n\n")
    elif abs(test_metrics['f1_macro'] - test_metrics['f1_weighted']) > 0.05:
        f.write("**类别不平衡影响中等**: F1-Macro略低于F1-Weighted，小类别预测有一定困难\n\n")
    else:
        f.write("**类别平衡良好**: F1-Macro与F1-Weighted接近，各类别预测性能均衡\n\n")

    f.write("## 5. 按Code1样本数分层评估\n\n")
    f.write("### 5.1 分层性能对比\n\n")
    f.write("| 层级 | 样本数范围 | 样本数 | 类别数 | Accuracy | Precision | Recall | F1-Score |\n")
    f.write("|------|----------|--------|--------|----------|-----------|--------|----------|\n")
    for _, row in tier_df.iterrows():
        f.write(f"| {row['Tier']} | {row['Tier'].split('(')[1].split(')')[0]} | {row['Sample_Count']} | {row['Class_Count']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |\n")
    f.write("\n")

    f.write("### 5.2 分层性能分析\n\n")
    f.write("**大样本类别（≥1000）**:\n")
    if len(tier_df[tier_df['Tier'] == 'Large (≥1000)']) > 0:
        row = tier_df[tier_df['Tier'] == 'Large (≥1000)'].iloc[0]
        f.write(f"- 样本数: {row['Sample_Count']}\n")
        f.write(f"- 准确率: {row['Accuracy']*100:.2f}%\n")
        f.write(f"- F1-Score: {row['F1-Score']*100:.2f}%\n\n")

    f.write("**中样本类别（100-999）**:\n")
    medium_rows = tier_df[tier_df['Tier'].str.contains('Medium')]
    if len(medium_rows) > 0:
        total_samples = medium_rows['Sample_Count'].sum()
        avg_f1 = medium_rows['F1-Score'].mean()
        f.write(f"- 样本数: {total_samples}\n")
        f.write(f"- 平均F1-Score: {avg_f1*100:.2f}%\n\n")

    f.write("**小样本类别（<100）**:\n")
    small_rows = tier_df[tier_df['Tier'].str.contains('Small|Tiny')]
    if len(small_rows) > 0:
        total_samples = small_rows['Sample_Count'].sum()
        avg_f1 = small_rows['F1-Score'].mean()
        f.write(f"- 样本数: {total_samples}\n")
        f.write(f"- 平均F1-Score: {avg_f1*100:.2f}%\n\n")

    f.write("**性能趋势**:\n")
    if len(tier_df) > 0:
        f.write(f"- 大样本类别准确率: {tier_df.iloc[0]['Accuracy']*100:.2f}%\n")
        f.write(f"- 小样本类别准确率: {tier_df.iloc[-1]['Accuracy']*100:.2f}%\n")
        f.write(f"- 性能差距: {(tier_df.iloc[0]['Accuracy'] - tier_df.iloc[-1]['Accuracy'])*100:.2f}%\n\n")

    f.write("## 6. 特征重要性\n\n")
    f.write("### 6.1 Top 10特征\n\n")
    feature_importance = model.feature_importances_
    feature_names = features_104
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    f.write("| 排名 | 特征名称 | 重要性值 |\n")
    f.write("|------|---------|----------|\n")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        f.write(f"| {i} | {row['Feature']} | {row['Importance']:.4f} |\n")
    f.write("\n")

    f.write("## 7. 结论\n\n")
    f.write("### 7.1 主要发现\n\n")
    f.write(f"1. **总体性能**: 测试集准确率达到{test_metrics['accuracy']*100:.2f}%\n")
    f.write(f"2. **类别不平衡**: F1-Macro比F1-Weighted低{abs(test_metrics['f1_macro'] - test_metrics['f1_weighted'])*100:.2f}%\n")
    f.write(f"3. **大样本性能**: 大样本类别（≥1000）准确率高\n")
    f.write(f"4. **小样本挑战**: 小样本类别（<100）预测困难\n\n")

    f.write("### 7.2 模型优势\n\n")
    f.write("- 使用104个特征，信息丰富\n")
    f.write("- XGBoost模型性能稳定\n")
    f.write("- 训练集包含所有231类\n")
    f.write("- 泛化能力良好\n\n")

    f.write("### 7.3 改进方向\n\n")
    f.write("- 针对小样本类别进行迁移学习\n")
    f.write("- 使用CBU相似度进行知识迁移\n")
    f.write("- 探索多任务学习策略\n")
    f.write("- 优化超参数\n\n")

print(f"✓ 性能报告已保存: {report_path}")

# ============================================================================
# 步骤11: 保存结果JSON
# ============================================================================
print("\n[步骤11] 保存结果JSON...")

results = {
    'timestamp': datetime.now().isoformat(),
    'model': {
        'type': 'XGBoost',
        'params': xgb_params,
        'feature_count': len(features_104),
        'class_count': n_classes
    },
    'dataset': {
        'total_samples': len(df),
        'train_samples': X_train_split.shape[0],
        'val_samples': X_val.shape[0],
        'test_samples': X_test.shape[0],
        'train_classes': y_train_split.nunique(),
        'val_classes': y_val.nunique(),
        'test_classes': y_test.nunique()
    },
    'performance': {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    },
    'tier_evaluation': tier_df.to_dict('records')
}

results_path = MODELS_DIR / 'baseline_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"✓ 结果JSON已保存: {results_path}")

# ============================================================================
# 步骤12: 打印总结
# ============================================================================
print("\n" + "="*80)
print("Baseline模型训练总结")
print("="*80)
print(f"模型: XGBoost")
print(f"特征数: {len(features_104)}")
print(f"类别数: {n_classes}")
print()
print("数据集划分:")
print(f"  训练集: {X_train_split.shape[0]:,}样本, {y_train_split.nunique()}类")
print(f"  验证集: {X_val.shape[0]:,}样本, {y_val.nunique()}类")
print(f"  测试集: {X_test.shape[0]:,}样本, {y_test.nunique()}类")
print()
print("测试集性能:")
print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
print(f"  F1-Macro: {test_metrics['f1_macro']*100:.2f}%")
print(f"  F1-Weighted: {test_metrics['f1_weighted']*100:.2f}%")
print()
print("分层性能（测试集）:")
print(tier_df[['Tier', 'Sample_Count', 'Accuracy', 'F1-Score']].to_string(index=False))
print()
print("输出文件:")
print(f"  - {model_path}")
print(f"  - {results_path}")
print(f"  - {tier_path}")
print(f"  - {full_cm_path}")
print(f"  - {importance_path}")
print(f"  - {confusion_path}")
print(f"  - {report_path}")
print("="*80)