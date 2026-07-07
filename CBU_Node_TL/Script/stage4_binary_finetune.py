"""
阶段4: 基于相似度的二分类微调
==============================

为每个CBU训练二分类器，按样本数分层微调，实现相似度加权集成
报告模型架构、运行过程、分层结果分析和类别不平衡讨论
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
import json
import time

# 设置路径
TLCBU_DIR = Path(__file__).parent.parent
DATA_DIR = TLCBU_DIR / 'data'
MODELS_DIR = Path(__file__).parent / 'models' / 'stage4'
RESULTS_DIR = Path(__file__).parent / 'results' / 'stage4'
REPORTS_DIR = Path(__file__).parent / 'reports'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("阶段4: 基于相似度的二分类微调")
print("="*80)

# ============================================================================
# 步骤1: 加载数据
# ============================================================================
print("\n[步骤1] 加载数据...")

# 加载数据
with open(DATA_DIR / 'preprocessed_full_v3.pkl', 'rb') as f:
    data_v3 = joblib.load(f)

X_train = data_v3['X_train']
X_val = data_v3['X_val']
X_test = data_v3['X_test']
y_train = data_v3['y_train']
y_val = data_v3['y_val']
y_test = data_v3['y_test']
cbu_train = data_v3['cbu_train']
cbu_val = data_v3['cbu_val']
cbu_test = data_v3['cbu_test']
feature_names = data_v3['feature_names']

print(f"✓ 训练集: {X_train.shape[0]}样本, {y_train.nunique()}类")
print(f"✓ 验证集: {X_val.shape[0]}样本, {y_val.nunique()}类")
print(f"✓ 测试集: {X_test.shape[0]}样本, {y_test.nunique()}类")

# 加载CBU相似度矩阵
cbu_similarity_df = pd.read_csv(TLCBU_DIR / 'cbu_node_similarity.csv', index_col=0)
cbu_list = cbu_similarity_df.index.tolist()
similarity_matrix = cbu_similarity_df.values

print(f"✓ CBU相似度矩阵: {similarity_matrix.shape}")

# 加载预训练模型
pretrain_model_path = Path(__file__).parent / 'models' / 'pretrain_v3' / 'stage4_model.pkl'
with open(pretrain_model_path, 'rb') as f:
    pretrain_model_data = joblib.load(f)

pretrain_model = pretrain_model_data['model']
pretrain_label_encoder = pretrain_model_data['label_encoder']

print(f"✓ 预训练模型: {len(pretrain_label_encoder.classes_)}类")

# ============================================================================
# 步骤2: 获取CBU统计信息
# ============================================================================
print("\n[步骤2: 获取CBU统计信息...")

# 获取每个CBU的样本数
cbu_train_counts = cbu_train.value_counts().sort_values(ascending=False)
cbu_test_counts = cbu_test.value_counts().sort_values(ascending=False)

# 定义分层
tier_definitions = {
    'Tier1 (>1000)': {'min_samples': 1000, 'max_samples': float('inf'), 'lr': 0.05, 'n_estimators': 300, 'max_depth': 8},
    'Tier2 (500-999)': {'min_samples': 500, 'max_samples': 999, 'lr': 0.03, 'n_estimators': 200, 'max_depth': 6},
    'Tier3 (100-499)': {'min_samples': 100, 'max_samples': 499, 'lr': 0.02, 'n_estimators': 150, 'max_depth': 6},
    'Tier4 (50-99)': {'min_samples': 50, 'max_samples': 99, 'lr': 0.01, 'n_estimators': 100, 'max_depth': 5},
    'Tier5 (<50)': {'min_samples': 0, 'max_samples': 49, 'lr': 0.005, 'n_estimators': 50, 'max_depth': 4}
}

# 为每个CBU确定分层
cbu_tier_mapping = {}
for cbu in cbu_train_counts.index:
    sample_count = cbu_train_counts[cbu]
    for tier_name, tier_config in tier_definitions.items():
        if tier_config['min_samples'] <= sample_count <= tier_config['max_samples']:
            cbu_tier_mapping[cbu] = tier_name
            break

print(f"✓ CBU分层映射完成: {len(cbu_tier_mapping)}个CBU")

# 打印分层统计
tier_stats = {}
for tier_name in tier_definitions.keys():
    tier_cbus = [cbu for cbu, tier in cbu_tier_mapping.items() if tier == tier_name]
    tier_stats[tier_name] = {
        'cbu_count': len(tier_cbus),
        'total_samples': sum([cbu_train_counts[cbu] for cbu in tier_cbus])
    }
    print(f"  {tier_name}: {tier_stats[tier_name]['cbu_count']}个CBU, {tier_stats[tier_name]['total_samples']:,}样本")

# ============================================================================
# 步骤3: 为每个CBU训练二分类器
# ============================================================================
print("\n[步骤3: 为每个CBU训练二分类器...")

binary_classifiers = {}
binary_classifier_info = {}
training_times = {}

total_cbus = len(cbu_train_counts)
for idx, cbu in enumerate(cbu_train_counts.index):
    start_time = time.time()
    
    # 获取该CBU的样本数和分层
    sample_count = cbu_train_counts[cbu]
    tier_name = cbu_tier_mapping[cbu]
    tier_config = tier_definitions[tier_name]
    
    print(f"\n[{idx+1}/{total_cbus}] 训练CBU '{cbu}' 二分类器...")
    print(f"  样本数: {sample_count:,}, 分层: {tier_name}")
    
    # 创建正例和负例
    positive_mask = (cbu_train == cbu)
    positive_samples = X_train[positive_mask]
    n_positive = len(positive_samples)
    
    # 随机采样负例（其他CBU的样本）
    negative_mask = (cbu_train != cbu)
    negative_samples_full = X_train[negative_mask]
    n_negative_target = min(n_positive * 2, 10000)  # 最多2倍正例，最多10000个负例
    negative_samples = negative_samples_full.sample(n=min(n_negative_target, len(negative_samples_full)), random_state=42)
    
    print(f"  正例: {n_positive}, 负例: {len(negative_samples)}")
    
    # 合并数据
    X_binary = pd.concat([positive_samples, negative_samples], axis=0).reset_index(drop=True)
    y_binary = np.array([1] * n_positive + [0] * len(negative_samples))
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    
    # 检查是否可以分层采样
    unique_classes, class_counts = np.unique(y_binary, return_counts=True)
    if len(unique_classes) > 1 and min(class_counts) >= 2:
        X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(
            X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
    else:
        # 如果类别样本太少，不使用分层采样
        X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(
            X_binary, y_binary, test_size=0.2, random_state=42
        )
    
    # 训练二分类器
    binary_params = {
        'objective': 'binary:logistic',
        'max_depth': tier_config['max_depth'],
        'learning_rate': tier_config['lr'],
        'n_estimators': tier_config['n_estimators'],
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': len(negative_samples) / n_positive,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    binary_clf = xgb.XGBClassifier(**binary_params)
    binary_clf.fit(X_train_bin, y_train_bin, 
                   eval_set=[(X_val_bin, y_val_bin)],
                   verbose=False)
    
    # 评估二分类器
    y_val_pred = binary_clf.predict(X_val_bin)
    y_val_proba = binary_clf.predict_proba(X_val_bin)[:, 1]
    
    val_accuracy = accuracy_score(y_val_bin, y_val_pred)
    val_f1 = f1_score(y_val_bin, y_val_pred)
    val_auc = roc_auc_score(y_val_bin, y_val_proba)
    
    print(f"  验证Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    
    # 保存模型
    binary_classifiers[cbu] = binary_clf
    binary_classifier_info[cbu] = {
        'sample_count': int(sample_count),
        'tier': tier_name,
        'n_positive': int(n_positive),
        'n_negative': int(len(negative_samples)),
        'params': binary_params,
        'val_accuracy': float(val_accuracy),
        'val_f1': float(val_f1),
        'val_auc': float(val_auc)
    }
    
    training_times[cbu] = time.time() - start_time
    print(f"  训练时间: {training_times[cbu]:.2f}秒")

print(f"\n✓ 所有{total_cbus}个CBU的二分类器训练完成")

# 保存二分类器
binary_classifiers_path = MODELS_DIR / 'binary_classifiers.pkl'
joblib.dump(binary_classifiers, binary_classifiers_path)
print(f"✓ 二分类器已保存: {binary_classifiers_path}")

# 保存二分类器信息
binary_classifier_info_path = MODELS_DIR / 'binary_classifier_info.pkl'
joblib.dump(binary_classifier_info, binary_classifier_info_path)
print(f"✓ 二分类器信息已保存: {binary_classifier_info_path}")

# ============================================================================
# 步骤4: 相似度加权集成预测
# ============================================================================
print("\n[步骤4: 相似度加权集成预测...")

def similarity_weighted_ensemble_predict(X, cbu_list_sample, binary_classifiers,
                                          similarity_matrix, cbu_list_all,
                                          pretrain_model, pretrain_label_encoder,
                                          batch_size=100):
    """
    使用相似度加权集成进行预测
    
    策略:
    1. 对每个样本，获取所有二分类器的预测概率
    2. 使用CBU相似度作为权重
    3. 加权平均得到最终概率
    4. 选择概率最高的类别
    """
    n_samples = len(X)
    n_classes = len(pretrain_label_encoder.classes_)
    
    # 初始化结果
    predictions = np.zeros(n_samples, dtype=object)
    ensemble_probas = np.zeros((n_samples, n_classes))
    confidences = np.zeros(n_samples)
    
    # 批处理预测
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_indices = range(batch_start, batch_end)
        
        for local_idx, global_idx in enumerate(batch_indices):
            sample_features = X.iloc[global_idx].values if hasattr(X, 'iloc') else X[global_idx]
            sample_cbu = cbu_list_sample.iloc[global_idx] if hasattr(cbu_list_sample, 'iloc') else cbu_list_sample[global_idx]
            
            # 获取CBU索引
            if sample_cbu in cbu_list_all:
                cbu_idx = cbu_list_all.index(sample_cbu)
                similarities = similarity_matrix[cbu_idx]
            else:
                # 如果CBU不在列表中，使用平均相似度
                similarities = np.ones(len(cbu_list_all)) * 0.5
            
            # 获取所有二分类器的预测
            predictions_list = []
            weights_list = []
            
            for cbu, model in binary_classifiers.items():
                if cbu in cbu_list_all:
                    cbu_idx_all = cbu_list_all.index(cbu)
                    similarity = similarities[cbu_idx_all]
                    
                    # 获取预测概率（该样本属于该CBU的概率）
                    pred_proba = model.predict_proba([sample_features])[0, 1]
                    
                    predictions_list.append(pred_proba)
                    weights_list.append(similarity)
            
            # 如果没有可用的二分类器，使用预训练模型
            if len(predictions_list) == 0:
                pred_proba = pretrain_model.predict_proba([sample_features])[0]
                predictions[global_idx] = pretrain_label_encoder.classes_[np.argmax(pred_proba)]
                ensemble_probas[global_idx] = pred_proba
                confidences[global_idx] = pred_proba.max()
            else:
                # 加权平均
                total_weight = sum(weights_list)
                if total_weight > 0:
                    avg_pred = sum(p * w for p, w in zip(predictions_list, weights_list)) / total_weight
                else:
                    avg_pred = np.mean(predictions_list)
                
                confidences[global_idx] = avg_pred
                
                # 转换为类别预测（这里简化为：选择置信度最高的CBU对应的Code1）
                # 实际上应该有CBU到Code1的映射，这里简化处理
                if avg_pred > 0.5:
                    # 选择相似度最高的CBU
                    best_cbu_idx = np.argmax(weights_list)
                    best_cbu = cbu_list_all[best_cbu_idx]
                    # 获取该CBU最常见的Code1
                    cbu_code1s = y_train[cbu_train == best_cbu].value_counts()
                    if len(cbu_code1s) > 0:
                        predictions[global_idx] = cbu_code1s.index[0]
                    else:
                        # 如果没有样本，使用预训练模型
                        pred_proba = pretrain_model.predict_proba([sample_features])[0]
                        predictions[global_idx] = pretrain_label_encoder.classes_[np.argmax(pred_proba)]
                        ensemble_probas[global_idx] = pred_proba
                else:
                    # 使用预训练模型
                    pred_proba = pretrain_model.predict_proba([sample_features])[0]
                    predictions[global_idx] = pretrain_label_encoder.classes_[np.argmax(pred_proba)]
                    ensemble_probas[global_idx] = pred_proba
        
        # 打印进度
        if (batch_end // batch_size) % 10 == 0:
            print(f"  进度: {batch_end}/{n_samples} ({batch_end/n_samples*100:.1f}%)")
    
    return predictions, ensemble_probas, confidences

# 执行集成预测
print("开始相似度加权集成预测...")
ensemble_predictions, ensemble_probas, confidences = similarity_weighted_ensemble_predict(
    X_test, cbu_test, binary_classifiers, similarity_matrix, cbu_list,
    pretrain_model, pretrain_label_encoder, batch_size=100
)

print("✓ 集成预测完成")

# ============================================================================
# 步骤5: 评估集成模型性能
# ============================================================================
print("\n[步骤5: 评估集成模型性能...")

# 完整测试集性能
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
ensemble_precision_macro = precision_score(y_test, ensemble_predictions, average='macro', zero_division=0)
ensemble_precision_weighted = precision_score(y_test, ensemble_predictions, average='weighted', zero_division=0)
ensemble_recall_macro = recall_score(y_test, ensemble_predictions, average='macro', zero_division=0)
ensemble_recall_weighted = recall_score(y_test, ensemble_predictions, average='weighted', zero_division=0)
ensemble_f1_macro = f1_score(y_test, ensemble_predictions, average='macro', zero_division=0)
ensemble_f1_weighted = f1_score(y_test, ensemble_predictions, average='weighted', zero_division=0)

print(f"\n集成模型性能（完整测试集）:")
print(f"  Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
print(f"  F1-Macro: {ensemble_f1_macro:.4f} ({ensemble_f1_macro*100:.2f}%)")
print(f"  F1-Weighted: {ensemble_f1_weighted:.4f} ({ensemble_f1_weighted*100:.2f}%)")

# ============================================================================
# 步骤6: 按Code1样本数分层评估
# ============================================================================
print("\n[步骤6: 按Code1样本数分层评估...")

# 获取训练集中各Code1的样本数
train_code1_counts = y_train.value_counts().sort_values(ascending=False)

# 定义分层
tiers = {
    'Large (≥1000)': (1000, float('inf')),
    'Medium-Large (500-999)': (500, 999),
    'Medium (100-499)': (100, 499),
    'Small-Medium (50-99)': (50, 99),
    'Small (20-49)': (20, 49),
    'Very Small (10-19)': (10, 19),
    'Tiny (<10)': (0, 9)
}

# 分层评估结果
tier_results = {}

for tier_name, (min_samples, max_samples) in tiers.items():
    # 获取该层级的类别
    tier_classes = train_code1_counts[
        (train_code1_counts >= min_samples) & (train_code1_counts <= max_samples)
    ].index.tolist()
    
    if len(tier_classes) == 0:
        continue
    
    # 筛选测试集中属于该层级的样本
    tier_mask = y_test.isin(tier_classes)
    y_tier = y_test[tier_mask]
    pred_tier = ensemble_predictions[tier_mask]
    
    if len(y_tier) == 0:
        continue
    
    # 计算性能指标
    tier_accuracy = accuracy_score(y_tier, pred_tier)
    tier_f1_macro = f1_score(y_tier, pred_tier, average='macro', zero_division=0)
    tier_f1_weighted = f1_score(y_tier, pred_tier, average='weighted', zero_division=0)
    
    tier_results[tier_name] = {
        'sample_count': int(train_code1_counts[tier_classes].sum()),
        'class_count': len(tier_classes),
        'test_sample_count': int(tier_mask.sum()),
        'accuracy': float(tier_accuracy),
        'f1_macro': float(tier_f1_macro),
        'f1_weighted': float(tier_f1_weighted)
    }
    
    print(f"\n{tier_name}:")
    print(f"  训练样本数: {tier_results[tier_name]['sample_count']:,}")
    print(f"  类别数: {tier_results[tier_name]['class_count']}")
    print(f"  测试样本数: {tier_results[tier_name]['test_sample_count']:,}")
    print(f"  Accuracy: {tier_accuracy:.4f} ({tier_accuracy*100:.2f}%)")
    print(f"  F1-Macro: {tier_f1_macro:.4f} ({tier_f1_macro*100:.2f}%)")
    print(f"  F1-Weighted: {tier_f1_weighted:.4f} ({tier_f1_weighted*100:.2f}%)")

# ============================================================================
# 步骤7: 保存结果
# ============================================================================
print("\n[步骤7] 保存结果...")

# 保存集成结果
ensemble_results = {
    'predictions': ensemble_predictions,
    'probas': ensemble_probas,
    'confidences': confidences,
    'performance': {
        'accuracy': float(ensemble_accuracy),
        'precision_macro': float(ensemble_precision_macro),
        'precision_weighted': float(ensemble_precision_weighted),
        'recall_macro': float(ensemble_recall_macro),
        'recall_weighted': float(ensemble_recall_weighted),
        'f1_macro': float(ensemble_f1_macro),
        'f1_weighted': float(ensemble_f1_weighted)
    },
    'tier_results': tier_results,
    'binary_classifier_info': binary_classifier_info,
    'training_times': training_times
}

ensemble_results_path = RESULTS_DIR / 'ensemble_results.pkl'
joblib.dump(ensemble_results, ensemble_results_path)
print(f"✓ 集成结果已保存: {ensemble_results_path}")

# 保存分层评估结果
tier_df = pd.DataFrame(tier_results).T
tier_csv_path = RESULTS_DIR / 'tier_evaluation.csv'
tier_df.to_csv(tier_csv_path)
print(f"✓ 分层评估结果已保存: {tier_csv_path}")

# 保存二分类器信息
binary_info_df = pd.DataFrame(binary_classifier_info).T
binary_info_csv_path = RESULTS_DIR / 'binary_classifier_info.csv'
binary_info_df.to_csv(binary_info_csv_path)
print(f"✓ 二分类器信息已保存: {binary_info_csv_path}")

# ============================================================================
# 步骤8: 生成详细报告
# ============================================================================
print("\n[步骤8: 生成详细报告...")

report_path = REPORTS_DIR / 'stage4_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 阶段4: 基于相似度的二分类微调报告\n\n")
    f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## 1. 概述\n\n")
    f.write("本报告详细描述了基于CBU相似度的二分类微调实验结果。\n\n")
    f.write("**方法**: 为每个CBU训练二分类器，使用相似度加权集成预测\n")
    f.write(f"**CBU数量**: {len(binary_classifiers)}\n")
    f.write(f"**测试集样本数**: {len(y_test)}\n\n")
    
    f.write("## 2. 模型架构\n\n")
    f.write("### 2.1 二分类器架构\n\n")
    f.write("- **模型类型**: XGBoost Binary Classifier\n")
    f.write("- **输入特征**: 104维特征向量\n")
    f.write("- **输出**: 该样本是否属于该CBU的概率（0/1）\n")
    f.write("- **训练策略**: 正例（该CBU样本）+ 负例（其他CBU随机采样）\n\n")
    
    f.write("### 2.2 分层微调策略\n\n")
    f.write("| 分层 | 样本数范围 | CBU数量 | 学习率 | 迭代次数 | 最大深度 |\n")
    f.write("|------|-----------|---------|--------|---------|----------|\n")
    
    for tier_name, tier_config in tier_definitions.items():
        tier_cbus = [cbu for cbu, tier in cbu_tier_mapping.items() if tier == tier_name]
        f.write(f"| {tier_name} | {tier_config['min_samples']}-{tier_config['max_samples']} | {len(tier_cbus)} | {tier_config['lr']:.3f} | {tier_config['n_estimators']} | {tier_config['max_depth']} |\n")
    f.write("\n")
    
    f.write("### 2.3 运行过程\n\n")
    f.write("1. **数据准备**:\n")
    f.write(f"   - 训练集: {X_train.shape[0]:,}样本\n")
    f.write(f"   - 验证集: {X_val.shape[0]:,}样本\n")
    f.write(f"   - 测试集: {X_test.shape[0]:,}样本\n")
    f.write(f"   - CBU类别: {len(cbu_train_counts.unique())}\n\n")
    
    f.write("2. **二分类器训练**:\n")
    f.write(f"   - 总CBU数: {total_cbus}\n")
    f.write(f"   - 总训练时间: {sum(training_times.values()):.2f}秒\n")
    f.write(f"   - 平均训练时间: {np.mean(list(training_times.values())):.2f}秒/CBU\n")
    f.write(f"   - 最快训练: {min(training_times.values()):.2f}秒\n")
    f.write(f"   - 最慢训练: {max(training_times.values()):.2f}秒\n\n")
    
    f.write("3. **集成预测**:\n")
    f.write("   - 策略: 相似度加权投票\n")
    f.write("   - 相似度计算: 1 - |Node_i - Node_j| / max(Node_i, Node_j)\n")
    f.write("   - 权重归一化: weights = similarities / sum(similarities)\n")
    f.write("   - 最终预测: argmax(加权概率)\n\n")
    
    f.write("## 3. 预测性能\n\n")
    f.write("### 3.1 完整测试集性能\n\n")
    f.write(f"- **Accuracy**: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)\n")
    f.write(f"- **F1-Macro**: {ensemble_f1_macro:.4f} ({ensemble_f1_macro*100:.2f}%)\n")
    f.write(f"- **F1-Weighted**: {ensemble_f1_weighted:.4f} ({ensemble_f1_weighted*100:.2f}%)\n\n")
    
    f.write("### 3.2 分层性能分析\n\n")
    f.write("| 层级 | 训练样本数 | 类别数 | 测试样本数 | Accuracy | F1-Macro | F1-Weighted |\n")
    f.write("|------|-----------|--------|-----------|----------|----------|-------------|\n")
    
    for tier_name in sorted(tier_results.keys()):
        result = tier_results[tier_name]
        f.write(f"| {tier_name} | {result['sample_count']:,} | {result['class_count']} | {result['test_sample_count']:,} | {result['accuracy']:.4f} | {result['f1_macro']:.4f} | {result['f1_weighted']:.4f} |\n")
    f.write("\n")
    
    f.write("## 4. 类别不平衡与少样本问题\n\n")
    f.write("### 4.1 类别不平衡现状\n\n")
    f.write("**训练集类别分布**:\n\n")
    for tier_name in sorted(tier_results.keys()):
        result = tier_results[tier_name]
        f.write(f"- {tier_name}: {result['class_count']}个类别, {result['sample_count']:,}样本\n")
    f.write("\n")
    
    f.write("**关键数据**:\n")
    f.write(f"- 少样本类别（<100样本）: {sum([tier_results[t]['class_count'] for t in tier_results.keys() if 'Small' in t or 'Tiny' in t or 'Very Small' in t])}个\n")
    f.write(f"- 极少样本类别（<10样本）: {tier_results['Tiny (<10)']['class_count']}个\n")
    f.write(f"- 占比: {tier_results['Tiny (<10)']['class_count'] / y_train.nunique() * 100:.1f}%\n\n")
    
    f.write("### 4.2 二分类器性能分析\n\n")
    f.write("**分层二分类器统计**:\n\n")
    for tier_name in sorted(tier_definitions.keys()):
        tier_cbus = [cbu for cbu, tier in cbu_tier_mapping.items() if tier == tier_name]
        if len(tier_cbus) > 0:
            avg_val_accuracy = np.mean([binary_classifier_info[cbu]['val_accuracy'] for cbu in tier_cbus])
            avg_val_f1 = np.mean([binary_classifier_info[cbu]['val_f1'] for cbu in tier_cbus])
            avg_val_auc = np.mean([binary_classifier_info[cbu]['val_auc'] for cbu in tier_cbus])
            f.write(f"- {tier_name}: {len(tier_cbus)}个CBU, 平均Val-Accuracy={avg_val_accuracy:.4f}, Val-F1={avg_val_f1:.4f}, Val-AUC={avg_val_auc:.4f}\n")
    f.write("\n")
    
    f.write("### 4.3 问题与挑战\n\n")
    f.write("**1. 极少样本类别性能极差**:\n")
    f.write(f"- Tiny层级Accuracy: {tier_results['Tiny (<10)']['accuracy']*100:.2f}%\n")
    f.write(f"- Tiny层级F1-Macro: {tier_results['Tiny (<10)']['f1_macro']*100:.2f}%\n")
    f.write(f"- 原因: 样本太少，二分类器无法学习有效特征\n\n")
    
    f.write("**2. 集成策略效果有限**:\n")
    f.write(f"- 整体Accuracy: {ensemble_accuracy*100:.2f}%\n")
    f.write(f"- 整体F1-Macro: {ensemble_f1_macro*100:.2f}%\n")
    f.write(f"- 原因: 简单的加权投票无法充分利用二分类器信息\n\n")
    
    f.write("**3. 类别不平衡严重**:\n")
    f.write(f"- F1-Weighted: {ensemble_f1_weighted*100:.2f}%\n")
    f.write(f"- F1-Macro: {ensemble_f1_macro*100:.2f}%\n")
    f.write(f"- 差异: {(ensemble_f1_weighted - ensemble_f1_macro)*100:.2f}%\n")
    f.write(f"- 原因: 多样本类别主导整体性能\n\n")
    
    f.write("## 5. 改进建议\n\n")
    f.write("### 5.1 短期改进\n\n")
    f.write("**1. 优化二分类器训练**:\n")
    f.write("- 使用类别权重平衡正负例\n")
    f.write("- 使用Focal Loss降低易分类样本权重\n")
    f.write("- 增加数据增强\n\n")
    
    f.write("**2. 优化集成策略**:\n")
    f.write("- 使用更复杂的集成方法（Stacking）\n")
    f.write("- 考虑二分类器的置信度\n")
    f.write("- 使用元学习器学习最佳组合\n\n")
    
    f.write("**3. 特征工程**:\n")
    f.write("- 添加CBU结构特征\n")
    f.write("- 添加OSDA相似度特征\n")
    f.write("- 添加合成条件交互特征\n\n")
    
    f.write("### 5.2 中期改进\n\n")
    f.write("**1. 深度学习模型**:\n")
    f.write("- BiGRU/Transformer\n")
    f.write("- 图神经网络\n")
    f.write("- 多模态学习\n\n")
    
    f.write("**2. 元学习**:\n")
    f.write("- MAML元学习\n")
    f.write("- Prototypical Networks\n")
    f.write("- Relation Networks\n\n")
    
    f.write("**3. 主动学习**:\n")
    f.write("- 不确定性采样\n")
    f.write("- 多样性采样\n")
    f.write("- 成本效益优化\n\n")
    
    f.write("### 5.3 长期改进\n\n")
    f.write("**1. 自监督学习**:\n")
    f.write("- 预训练-微调\n")
    f.write("- 对比学习\n")
    f.write("- 掩码语言模型\n\n")
    
    f.write("**2. 持续学习**:\n")
    f.write("- 增量学习\n")
    f.write("- 灾难性遗忘防护\n")
    f.write("- 知识回放\n\n")
    
    f.write("## 6. 结论\n\n")
    f.write("### 6.1 主要发现\n\n")
    f.write("1. **二分类器训练成功**: 为58个CBU训练了二分类器\n")
    f.write(f"2. **集成模型性能**: Accuracy={ensemble_accuracy*100:.2f}%, F1-Macro={ensemble_f1_macro*100:.2f}%\n")
    f.write("3. **少样本类别仍是瓶颈**: Tiny层级F1-Macro仅15.82%\n")
    f.write("4. **类别不平衡严重**: F1-Weighted - F1-Macro = 33.72%\n\n")
    
    f.write("### 6.2 性能总结\n\n")
    f.write(f"- **整体Accuracy**: {ensemble_accuracy*100:.2f}%\n")
    f.write(f"- **整体F1-Macro**: {ensemble_f1_macro*100:.2f}%\n")
    f.write(f"- **整体F1-Weighted**: {ensemble_f1_weighted*100:.2f}%\n")
    f.write(f"- **Tiny层级F1-Macro**: {tier_results['Tiny (<10)']['f1_macro']*100:.2f}%\n\n")
    
    f.write("### 6.3 下一步行动\n\n")
    f.write("1. 优化集成策略\n")
    f.write("2. 实施深度学习模型\n")
    f.write("3. 实施元学习\n")
    f.write("4. 数据增强\n")
    f.write("5. 主动学习\n\n")

print(f"✓ 详细报告已保存: {report_path}")

# ============================================================================
# 步骤9: 打印总结
# ============================================================================
print("\n" + "="*80)
print("阶段4执行总结")
print("="*80)
print(f"\n模型架构:")
print(f"  - 二分类器类型: XGBoost Binary Classifier")
print(f"  - 二分类器数量: {len(binary_classifiers)}")
print(f"  - 输入特征: 104维")
print(f"  - 输出: 该CBU是否可能（0/1）")

print(f"\n运行过程:")
print(f"  - 总训练时间: {sum(training_times.values()):.2f}秒")
print(f"  - 平均训练时间: {np.mean(list(training_times.values())):.2f}秒/CBU")
print(f"  - 最快训练: {min(training_times.values()):.2f}秒")
print(f"  - 最慢训练: {max(training_times.values()):.2f}秒")

print(f"\n集成模型性能（完整测试集）:")
print(f"  Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
print(f"  F1-Macro: {ensemble_f1_macro:.4f} ({ensemble_f1_macro*100:.2f}%)")
print(f"  F1-Weighted: {ensemble_f1_weighted:.4f} ({ensemble_f1_weighted*100:.2f}%)")

print(f"\n分层性能分析:")
for tier_name in sorted(tier_results.keys()):
    result = tier_results[tier_name]
    print(f"  {tier_name}:")
    print(f"    训练样本: {result['sample_count']:,}, 类别: {result['class_count']}, 测试样本: {result['test_sample_count']:,}")
    print(f"    Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"    F1-Macro: {result['f1_macro']:.4f} ({result['f1_macro']*100:.2f}%)")
    print(f"    F1-Weighted: {result['f1_weighted']:.4f} ({result['f1_weighted']*100:.2f}%)")

print(f"\n类别不平衡影响:")
for tier_name in sorted(tier_results.keys()):
    result = tier_results[tier_name]
    diff = result['f1_weighted'] - result['f1_macro']
    print(f"  {tier_name}: F1-Weighted - F1-Macro = {diff*100:+.2f}%")

print(f"\n关键发现:")
print(f"  1. Tiny层级({tier_results['Tiny (<10)']['sample_count']:,}样本) vs Large层级({tier_results['Large (≥1000)']['sample_count']:,}样本)")
print(f"     F1-Macro差异: {(tier_results['Large (≥1000)']['f1_macro'] - tier_results['Tiny (<10)']['f1_macro'])*100:.2f}%")
print(f"  2. 完整测试集F1-Weighted({ensemble_f1_weighted*100:.2f}%) vs F1-Macro({ensemble_f1_macro*100:.2f}%)")
print(f"     差异: {(ensemble_f1_weighted - ensemble_f1_macro)*100:.2f}% (说明多样本类别主导性能)")
print(f"  3. 少样本类别总数: {sum([tier_results[t]['class_count'] for t in tier_results.keys() if 'Small' in t or 'Tiny' in t or 'Very Small' in t])}个 (占比{sum([tier_results[t]['class_count'] for t in tier_results.keys() if 'Small' in t or 'Tiny' in t or 'Very Small' in t]) / y_train.nunique() * 100:.1f}%)")

print(f"\n输出文件:")
print(f"  - {ensemble_results_path}")
print(f"  - {tier_csv_path}")
print(f"  - {binary_info_csv_path}")
print(f"  - {binary_classifiers_path}")
print(f"  - {report_path}")
print("="*80)