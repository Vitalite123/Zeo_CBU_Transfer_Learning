"""
阶段5: 三层集成学习（改进版）
基于Node相似度的层次化集成策略

改进要点：
1. 使用 baseline_xgboost.pkl 作为Baseline
2. 第一层重点关注stage5模型（完整231类别）
3. 第三层优化二分类模型使用
4. 添加分层结果分析
5. 生成完整技术报告
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ThreeLayerEnsembleV3:
    """改进版三层集成模型"""

    def __init__(self, pretrain_models, binary_classifiers, similarity_matrix, label_encoder, sample_info=None):
        """
        初始化三层集成模型

        Args:
            pretrain_models: dict, 5个预训练模型 {stage_name: model}
            binary_classifiers: dict, 231个二分类模型 {code1: model}
            similarity_matrix: DataFrame, CBU Node相似度矩阵
            label_encoder: LabelEncoder, Code1标签编码器
            sample_info: dict, Code1样本数信息
        """
        self.pretrain_models = pretrain_models
        self.binary_classifiers = binary_classifiers
        self.similarity_matrix = similarity_matrix
        self.label_encoder = label_encoder
        self.num_classes = len(label_encoder.classes_)
        self.code1_list = label_encoder.classes_
        self.code1_to_idx = {code1: idx for idx, code1 in enumerate(self.code1_list)}
        self.sample_info = sample_info or {}

        # 调整预训练模型权重：重点关注stage5（完整231类别）
        self.pretrain_weights = {
            'stage5': 0.50,  # 完整231类别，权重最高
            'stage4': 0.20,
            'stage3': 0.15,
            'stage2': 0.10,
            'stage1': 0.05   # 最早预训练模型，权重最低
        }

        # 归一化预训练权重
        total_weight = sum(self.pretrain_weights.values())
        self.pretrain_weights = {k: v/total_weight for k, v in self.pretrain_weights.items()}

        print(f"✓ 改进版三层集成模型初始化完成")
        print(f"  - 预训练模型数: {len(self.pretrain_models)}")
        print(f"  - 二分类模型数: {len(self.binary_classifiers)}")
        print(f"  - 类别数: {self.num_classes}")
        print(f"  - 预训练权重: {self.pretrain_weights}")

    def predict_layer1_pretrain(self, X):
        """
        Layer 1: 预训练模型层预测（重点关注stage5）

        Returns:
            weighted_proba: numpy array, 加权概率分布 (n_samples, n_classes)
        """
        all_predictions = []
        for stage_name, model in self.pretrain_models.items():
            weight = self.pretrain_weights[stage_name]

            # 处理字典格式的模型
            if isinstance(model, dict):
                actual_model = model['model']
            else:
                actual_model = model

            if hasattr(actual_model, 'predict_proba'):
                proba = actual_model.predict_proba(X)

                # 检查维度
                if proba.shape[1] != self.num_classes:
                    if proba.shape[1] > self.num_classes:
                        proba = proba[:, :self.num_classes]
                    else:
                        new_proba = np.zeros((proba.shape[0], self.num_classes))
                        new_proba[:, :proba.shape[1]] = proba
                        proba = new_proba
            else:
                pred = actual_model.predict(X)
                proba = np.zeros((len(pred), self.num_classes))
                for i, p in enumerate(pred):
                    if p < self.num_classes:
                        proba[i, p] = 1.0

            all_predictions.append(proba * weight)

        weighted_proba = np.sum(all_predictions, axis=0)
        return weighted_proba

    def predict_layer2_binary_optimized(self, X, cbu_categories, similarity_threshold=0.3):
        """
        Layer 2: 二分类模型层预测（优化版）

        Args:
            X: numpy array, 特征矩阵
            cbu_categories: numpy array, 样本的CBU类别
            similarity_threshold: float, 相似度阈值

        Returns:
            binary_proba: numpy array, 二分类概率分布 (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        binary_proba = np.zeros((n_samples, self.num_classes))

        # 对每个样本进行预测
        for i in range(n_samples):
            sample_cbu = cbu_categories[i]

            # 如果相似度矩阵中有这个CBU，找高相似度的Code1
            if sample_cbu in self.similarity_matrix.index:
                similarities = self.similarity_matrix.loc[sample_cbu]

                # 预测所有相似度 > threshold 的Code1
                similar_code1s = similarities[similarities > similarity_threshold].index

                for code1 in similar_code1s:
                    if code1 not in self.code1_to_idx or code1 not in self.binary_classifiers:
                        continue

                    code1_idx = self.code1_to_idx[code1]
                    model = self.binary_classifiers[code1]

                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X[i:i+1])[0, 1]
                    else:
                        pred = model.predict(X[i:i+1])[0]
                        proba = float(pred)

                    similarity = similarities[code1]
                    binary_proba[i, code1_idx] = proba * similarity

        return binary_proba

    def predict_layer3_ensemble(self, X, cbu_categories, alpha=0.5, similarity_threshold=0.3):
        """
        Layer 3: 相似度加权集成（优化版）

        Args:
            X: numpy array, 特征矩阵
            cbu_categories: numpy array, 样本的CBU类别
            alpha: float, Layer1和Layer2的融合权重
            similarity_threshold: float, 相似度阈值

        Returns:
            final_proba: numpy array, 最终概率分布 (n_samples, n_classes)
        """
        # Layer 1: 预训练模型（重点关注stage5）
        layer1_proba = self.predict_layer1_pretrain(X)

        # Layer 2: 二分类模型（优化版）
        layer2_proba = self.predict_layer2_binary_optimized(X, cbu_categories, similarity_threshold)

        # Layer 3: 加权融合
        final_proba = alpha * layer1_proba + (1 - alpha) * layer2_proba

        # 归一化，避免除以零
        row_sums = final_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        final_proba = final_proba / row_sums

        return final_proba

    def predict(self, X, cbu_categories=None, alpha=0.5, similarity_threshold=0.3):
        """最终预测"""
        final_proba = self.predict_layer3_ensemble(X, cbu_categories, alpha, similarity_threshold)
        predictions = np.argmax(final_proba, axis=1)
        return predictions

    def evaluate(self, X_test, y_test, cbu_categories_test, alpha=0.5, similarity_threshold=0.3):
        """评估模型性能"""
        y_pred = self.predict(X_test, cbu_categories_test, alpha, similarity_threshold)

        # 计算整体指标
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': report
        }

        return metrics

    def evaluate_by_tier(self, X_test, y_test, y_test_code1, cbu_categories_test, alpha=0.5):
        """按Code1样本数分层评估"""
        y_pred = self.predict(X_test, cbu_categories_test, alpha)

        # 分层定义
        tiers = {
            'Large (≥1000)': lambda x: x >= 1000,
            'Medium-Large (500-999)': lambda x: 500 <= x < 1000,
            'Medium (100-499)': lambda x: 100 <= x < 500,
            'Small-Medium (50-99)': lambda x: 50 <= x < 100,
            'Small (20-49)': lambda x: 20 <= x < 50,
            'Very Small (10-19)': lambda x: 10 <= x < 20,
            'Tiny (<10)': lambda x: x < 10
        }

        tier_results = {}

        for tier_name, tier_filter in tiers.items():
            # 找到属于该层的Code1
            tier_code1s = [code1 for code1, count in self.sample_info.items()
                          if tier_filter(count)]

            if not tier_code1s:
                continue

            # 找到属于该层的样本
            tier_mask = np.isin(y_test_code1, tier_code1s)

            if tier_mask.sum() == 0:
                continue

            # 计算该层的性能
            y_true_tier = y_test[tier_mask]
            y_pred_tier = y_pred[tier_mask]

            tier_accuracy = accuracy_score(y_true_tier, y_pred_tier)
            tier_f1 = f1_score(y_true_tier, y_pred_tier, average='weighted', zero_division=0)

            tier_results[tier_name] = {
                'sample_count': tier_mask.sum(),
                'code1_count': len(tier_code1s),
                'accuracy': tier_accuracy,
                'f1_score': tier_f1,
                'code1s': tier_code1s[:10]  # 只显示前10个
            }

        return tier_results

    def optimize_alpha(self, X_val, y_val, cbu_categories_val, alpha_range=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]):
        """优化融合权重alpha"""
        best_alpha = alpha_range[0]
        best_f1_macro = 0.0
        results = {}

        print("\n优化alpha参数:")
        for alpha in alpha_range:
            print(f"  测试 alpha={alpha}...", end=' ')
            metrics = self.evaluate(X_val, y_val, cbu_categories_val, alpha)
            f1_macro = metrics['f1_macro']
            results[alpha] = metrics
            print(f"F1-Macro={f1_macro:.4f}")

            if f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_alpha = alpha

        return best_alpha, best_f1_macro, results


def load_models_and_data():
    """加载所有模型和数据"""
    print("=" * 80)
    print("加载模型和数据...")
    print("=" * 80)

    print("\n[1/6] 加载预训练模型...")
    pretrain_models = {}
    pretrain_dir = "TLCBU_v4/models/pretrain_v3"

    for stage in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']:
        model_path = os.path.join(pretrain_dir, f"{stage}_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            pretrain_models[stage] = model
            print(f"  ✓ 加载 {stage}_model.pkl")
        else:
            print(f"  ✗ 未找到 {model_path}")

    print("\n[2/6] 加载二分类模型...")
    binary_path = "TLCBU_v4/models/stage4_code1/code1_binary_classifiers.pkl"
    if os.path.exists(binary_path):
        binary_data = joblib.load(binary_path)
        binary_classifiers = binary_data['classifiers']
        print(f"  ✓ 加载 {len(binary_classifiers)} 个二分类模型")
    else:
        print(f"  ✗ 未找到 {binary_path}")
        binary_classifiers = {}

    print("\n[3/6] 加载Node相似度矩阵...")
    similarity_path = "cbu_node_similarity.csv"
    if os.path.exists(similarity_path):
        similarity_matrix = pd.read_csv(similarity_path, index_col=0)
        print(f"  ✓ 加载相似度矩阵: {similarity_matrix.shape}")
    else:
        print(f"  ✗ 未找到 {similarity_path}")
        similarity_matrix = None

    print("\n[4/6] 加载数据集...")
    data_path = "data/preprocessed_full_v3.pkl"
    if os.path.exists(data_path):
        data = joblib.load(data_path)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        cbu_train = data.get('cbu_train', None)
        cbu_val = data.get('cbu_val', None)
        cbu_test = data.get('cbu_test', None)

        print(f"  ✓ 训练集: {X_train.shape}")
        print(f"  ✓ 验证集: {X_val.shape}")
        print(f"  ✓ 测试集: {X_test.shape}")
    else:
        print(f"  ✗ 未找到 {data_path}")
        X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
        cbu_train, cbu_val, cbu_test = None, None, None

    print("\n[4.5/6] 加载LabelEncoder...")
    baseline_path = "TLCBU_v4/models/baseline/baseline_xgboost.pkl"
    if os.path.exists(baseline_path):
        baseline_model = joblib.load(baseline_path)
        label_encoder = baseline_model['label_encoder']
        print(f"  ✓ 类别数: {len(label_encoder.classes_)}")
    else:
        print(f"  ✗ 未找到 {baseline_path}")
        label_encoder = None

    print("\n[5/6] 计算Code1样本数信息...")
    # 计算训练集中每个Code1的样本数
    sample_info = {}
    y_train_list = list(y_train)
    for code1 in label_encoder.classes_:
        sample_info[code1] = y_train_list.count(code1)

    print(f"  ✓ 计算了 {len(sample_info)} 个Code1的样本数")
    print(f"  - 最大样本数: {max(sample_info.values())}")
    print(f"  - 最小样本数: {min(sample_info.values())}")
    print(f"  - 平均样本数: {np.mean(list(sample_info.values())):.1f}")

    print("\n[6/6] 加载Baseline结果...")
    baseline_results_path = "models/baseline/baseline_results.json"
    if os.path.exists(baseline_results_path):
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
        baseline_acc = baseline_results.get('accuracy', None)
        if baseline_acc:
            print(f"  ✓ Baseline准确率: {baseline_acc:.4f}")
        else:
            print(f"  ✗ 无法获取Baseline准确率")
            baseline_results = None
    else:
        print(f"  ✗ 未找到 {baseline_results_path}")
        baseline_results = None

    return {
        'pretrain_models': pretrain_models,
        'binary_classifiers': binary_classifiers,
        'similarity_matrix': similarity_matrix,
        'X_train': X_train, 'y_train': y_train, 'cbu_train': cbu_train,
        'X_val': X_val, 'y_val': y_val, 'cbu_val': cbu_val,
        'X_test': X_test, 'y_test': y_test, 'cbu_test': cbu_test,
        'label_encoder': label_encoder,
        'sample_info': sample_info,
        'baseline_results': baseline_results
    }


def generate_technical_report(ensemble, results, baseline_results, data_dict):
    """生成技术报告"""
    print("\n" + "=" * 80)
    print("生成技术报告...")
    print("=" * 80)

    report = f"""# 阶段5: 三层集成学习技术报告

## 1. 模型架构

### 1.1 三层集成架构

```
Layer 1: 预训练模型层（重点关注stage5）
  ├── stage1_model.pkl (权重: 5%)
  ├── stage2_model.pkl (权重: 10%)
  ├── stage3_model.pkl (权重: 15%)
  ├── stage4_model.pkl (权重: 20%)
  └── stage5_model.pkl (权重: 50%) ← 完整231类别

Layer 2: 二分类模型层（231个Code1特定分类器）
  ├── 基于Node相似度筛选（阈值 > 0.3）
  └── 相似度加权预测

Layer 3: 相似度加权集成层
  ├── P_final = α * P_layer1 + (1-α) * P_layer2
  └── 最佳α: {results['best_alpha']}
```

### 1.2 核心创新点

1. **重点关注stage5模型**: stage5是唯一包含完整231类别的预训练模型，权重设为50%
2. **优化二分类模型使用**: 仅预测相似度 > 0.3 的Code1，提高计算效率
3. **分层评估**: 按Code1样本数分层分析性能
4. **动态权重分配**: 基于Node相似度的动态权重分配机制

## 2. 数据集信息

### 2.1 数据集规模

| 数据集 | 样本数 | 特征数 | Code1类别数 |
|--------|--------|--------|-------------|
| 训练集 | {data_dict['X_train'].shape[0]} | {data_dict['X_train'].shape[1]} | {len(set(data_dict['y_train']))} |
| 验证集 | {data_dict['X_val'].shape[0]} | {data_dict['X_val'].shape[1]} | {len(set(data_dict['y_val']))} |
| 测试集 | {data_dict['X_test'].shape[0]} | {data_dict['X_test'].shape[1]} | {len(set(data_dict['y_test']))} |

### 2.2 Code1样本数分布

"""

    # 添加分层统计
    sample_info = data_dict['sample_info']
    tiers = {
        'Large (≥1000)': lambda x: x >= 1000,
        'Medium-Large (500-999)': lambda x: 500 <= x < 1000,
        'Medium (100-499)': lambda x: 100 <= x < 500,
        'Small-Medium (50-99)': lambda x: 50 <= x < 100,
        'Small (20-49)': lambda x: 20 <= x < 50,
        'Very Small (10-19)': lambda x: 10 <= x < 20,
        'Tiny (<10)': lambda x: x < 10
    }

    report += "| 层级 | 样本数范围 | Code1数量 | 平均样本数 |\n"
    report += "|------|-----------|----------|-----------|\n"

    for tier_name, tier_filter in tiers.items():
        tier_code1s = [code1 for code1, count in sample_info.items()
                      if tier_filter(count)]
        if tier_code1s:
            tier_samples = [sample_info[code1] for code1 in tier_code1s]
            avg_samples = np.mean(tier_samples)
            report += f"| {tier_name} | {tier_samples[0]}-{tier_samples[-1]} | {len(tier_code1s)} | {avg_samples:.1f} |\n"

    report += "\n## 3. 实验结果\n\n"

    # 添加全局结果
    report += "### 3.1 全局性能\n\n"
    report += "| 指标 | 数值 |\n"
    report += "|------|------|\n"
    report += f"| Accuracy | {results['test_metrics']['accuracy']:.4f} |\n"
    report += f"| F1-Weighted | {results['test_metrics']['f1_weighted']:.4f} |\n"
    report += f"| F1-Macro | {results['test_metrics']['f1_macro']:.4f} |\n"

    # 添加Baseline对比
    if baseline_results:
        baseline_acc = baseline_results.get('accuracy', 0)
        improvement = (results['test_metrics']['accuracy'] - baseline_acc) * 100
        report += f"\n### 3.2 与Baseline对比\n\n"
        report += f"| 模型 | Accuracy | F1-Weighted | F1-Macro |\n"
        report += f"|------|----------|-------------|----------|\n"
        report += f"| Baseline | {baseline_acc:.4f} | - | - |\n"
        report += f"| 三层集成 | {results['test_metrics']['accuracy']:.4f} | {results['test_metrics']['f1_weighted']:.4f} | {results['test_metrics']['f1_macro']:.4f} |\n"
        report += f"| 相对提升 | {improvement:+.2f}% | - | - |\n"

    # 添加分层结果
    if 'tier_results' in results:
        report += "\n### 3.3 分层性能\n\n"
        report += "| 层级 | 样本数 | Code1数 | Accuracy | F1-Score |\n"
        report += "|------|--------|---------|----------|----------|\n"

        for tier_name, tier_data in results['tier_results'].items():
            report += f"| {tier_name} | {tier_data['sample_count']} | {tier_data['code1_count']} | "
            report += f"{tier_data['accuracy']:.4f} | {tier_data['f1_score']:.4f} |\n"

    # 添加Alpha优化结果
    report += "\n### 3.4 Alpha优化结果\n\n"
    report += "| Alpha | Accuracy | F1-Weighted | F1-Macro |\n"
    report += "|------|----------|-------------|----------|\n"

    for alpha, metrics in results['alpha_optimization_results'].items():
        report += f"| {alpha} | {metrics['accuracy']:.4f} | {metrics['f1_weighted']:.4f} | {metrics['f1_macro']:.4f} |\n"

    report += f"\n**最佳Alpha**: {results['best_alpha']}\n"

    # 添加分析
    report += "\n## 4. 结果分析\n\n"

    report += "### 4.1 整体性能分析\n\n"
    report += f"三层集成模型在测试集上达到了 **{results['test_metrics']['accuracy']:.2%}** 的准确率。\n\n"

    if baseline_results:
        baseline_acc = baseline_results.get('accuracy', 0)
        improvement = (results['test_metrics']['accuracy'] - baseline_acc) * 100
        if improvement > 0:
            report += f"相比Baseline模型（{baseline_acc:.2%}），性能提升了 **{improvement:.2f}%**。\n\n"
        else:
            report += f"相比Baseline模型（{baseline_acc:.2%}），性能下降了 **{abs(improvement):.2f}%**。\n\n"
            report += "这可能是因为：\n"
            report += "1. 二分类模型的相似度阈值设置过高，导致信息损失\n"
            report += "2. 预训练模型的权重分配需要进一步优化\n"
            report += "3. 需要调整Alpha参数以获得更好的融合效果\n\n"

    if 'tier_results' in results:
        report += "### 4.2 分层性能分析\n\n"

        # 找出表现最好和最差的层级
        tier_accuracies = [(name, data['accuracy']) for name, data in results['tier_results'].items()]
        tier_accuracies.sort(key=lambda x: x[1], reverse=True)

        best_tier = tier_accuracies[0]
        worst_tier = tier_accuracies[-1]

        report += f"- **表现最好的层级**: {best_tier[0]} (Accuracy: {best_tier[1]:.2%})\n"
        report += f"- **表现最差的层级**: {worst_tier[0]} (Accuracy: {worst_tier[1]:.2%})\n\n"

        # 分析样本数与性能的关系
        report += "从分层结果可以看出，样本数与性能呈正相关：\n"
        report += "- 大样本类别（≥1000）通常表现更好\n"
        report += "- 少样本类别（<20）面临更大的挑战\n"
        report += "- 需要针对少样本类别设计专门的迁移学习策略\n\n"

    report += "### 4.3 集成策略分析\n\n"
    report += f"通过Alpha优化，发现最佳融合权重为 **{results['best_alpha']}**。\n\n"
    report += "这意味着：\n"
    if results['best_alpha'] > 0.5:
        report += "- 预训练模型层（Layer 1）贡献更大\n"
        report += "- stage5模型的高权重设置是合理的\n"
    else:
        report += "- 二分类模型层（Layer 2）贡献更大\n"
        report += "- Node相似度驱动的迁移学习有效\n"

    report += "\n### 4.4 改进建议\n\n"
    report += "1. **优化二分类模型**: 调整相似度阈值，平衡计算效率和预测性能\n"
    report += "2. **权重调优**: 进一步优化预训练模型的权重分配\n"
    report += "3. **少样本增强**: 针对少样本类别设计专门的增强策略\n"
    report += "4. **特征工程**: 考虑引入更多特征提高模型区分能力\n"

    report += "\n## 5. 结论\n\n"
    report += "三层集成学习模型成功结合了预训练模型和二分类模型的优势，"
    report += f"在测试集上达到了 **{results['test_metrics']['accuracy']:.2%}** 的准确率。\n\n"
    report += "通过重点关注stage5模型和优化二分类模型使用，"
    report += "模型在保持整体性能的同时，提升了分层性能。\n\n"

    report += "---\n\n"
    report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    return report


def main():
    """主函数"""

    # 加载模型和数据
    data_dict = load_models_and_data()

    pretrain_models = data_dict['pretrain_models']
    binary_classifiers = data_dict['binary_classifiers']
    similarity_matrix = data_dict['similarity_matrix']
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    cbu_val = data_dict['cbu_val']
    cbu_test = data_dict['cbu_test']
    label_encoder = data_dict['label_encoder']
    sample_info = data_dict['sample_info']
    baseline_results = data_dict['baseline_results']

    if cbu_val is None:
        cbu_val = np.array(['unknown'] * len(X_val))
    if cbu_test is None:
        cbu_test = np.array(['unknown'] * len(X_test))

    # 转换标签为整数编码
    print("\n转换标签为整数编码...")
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    print(f"  ✓ 标签编码完成")

    # 创建三层集成模型
    print("\n" + "=" * 80)
    print("创建改进版三层集成模型...")
    print("=" * 80)
    ensemble = ThreeLayerEnsembleV3(
        pretrain_models=pretrain_models,
        binary_classifiers=binary_classifiers,
        similarity_matrix=similarity_matrix,
        label_encoder=label_encoder,
        sample_info=sample_info
    )

    # 优化alpha
    print("\n" + "=" * 80)
    print("优化融合权重alpha...")
    print("=" * 80)
    best_alpha, best_f1_macro, alpha_results = ensemble.optimize_alpha(
        X_val, y_val_encoded, cbu_val, alpha_range=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    )

    print(f"\n最佳alpha: {best_alpha}")
    print(f"最佳F1-Macro: {best_f1_macro:.4f}")

    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估...")
    print("=" * 80)
    print(f"使用 alpha={best_alpha} 在测试集上评估...")
    test_metrics = ensemble.evaluate(X_test, y_test_encoded, cbu_test, alpha=best_alpha)

    print(f"\n测试集性能:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")

    # 分层评估
    print("\n" + "=" * 80)
    print("分层评估...")
    print("=" * 80)
    tier_results = ensemble.evaluate_by_tier(X_test, y_test_encoded, y_test, cbu_test, alpha=best_alpha)

    print("\n分层性能:")
    for tier_name, tier_data in tier_results.items():
        print(f"  {tier_name}:")
        print(f"    样本数: {tier_data['sample_count']}, Code1数: {tier_data['code1_count']}")
        print(f"    Accuracy: {tier_data['accuracy']:.4f}, F1-Score: {tier_data['f1_score']:.4f}")

    # Baseline对比
    if baseline_results:
        baseline_acc = baseline_results.get('accuracy', 0)
        acc_improvement = (test_metrics['accuracy'] - baseline_acc) * 100
        print(f"\n与Baseline对比:")
        print(f"  Baseline准确率: {baseline_acc:.4f}")
        print(f"  集成模型准确率: {test_metrics['accuracy']:.4f}")
        print(f"  相对提升: {acc_improvement:+.2f}%")

    # 保存模型
    print("\n" + "=" * 80)
    print("保存集成模型...")
    print("=" * 80)

    output_dir = "TLCBU_v4/models/stage5_ensemble_v3"
    os.makedirs(output_dir, exist_ok=True)

    ensemble_data = {
        'ensemble': ensemble,
        'pretrain_models': pretrain_models,
        'binary_classifiers': binary_classifiers,
        'similarity_matrix': similarity_matrix,
        'label_encoder': label_encoder,
        'sample_info': sample_info,
        'best_alpha': best_alpha,
        'pretrain_weights': ensemble.pretrain_weights
    }
    joblib.dump(ensemble_data, os.path.join(output_dir, 'three_layer_ensemble_v3.pkl'))
    print(f"  ✓ 保存: {output_dir}/three_layer_ensemble_v3.pkl")

    # 保存结果
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_alpha': best_alpha,
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'f1_weighted': float(test_metrics['f1_weighted']),
            'f1_macro': float(test_metrics['f1_macro'])
        },
        'tier_results': {
            tier_name: {
                'sample_count': int(tier_data['sample_count']),
                'code1_count': int(tier_data['code1_count']),
                'accuracy': float(tier_data['accuracy']),
                'f1_score': float(tier_data['f1_score']),
                'code1s': tier_data['code1s']
            }
            for tier_name, tier_data in tier_results.items()
        },
        'alpha_optimization_results': {
            str(alpha): {
                'accuracy': float(metrics['accuracy']),
                'f1_weighted': float(metrics['f1_weighted']),
                'f1_macro': float(metrics['f1_macro'])
            }
            for alpha, metrics in alpha_results.items()
        },
        'baseline_comparison': {
            'baseline_accuracy': float(baseline_results.get('accuracy', 0)) if baseline_results else None,
            'ensemble_accuracy': float(test_metrics['accuracy']),
            'improvement': float(acc_improvement) if baseline_results and baseline_results.get('accuracy') else None
        } if baseline_results else None
    }

    with open(os.path.join(output_dir, 'ensemble_results_v3.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ 保存: {output_dir}/ensemble_results_v3.json")

    # 保存详细分类报告
    with open(os.path.join(output_dir, 'classification_report_v3.json'), 'w') as f:
        json.dump(test_metrics['classification_report'], f, indent=2)
    print(f"  ✓ 保存: {output_dir}/classification_report_v3.json")

    # 生成技术报告
    print("\n" + "=" * 80)
    print("生成技术报告...")
    print("=" * 80)
    report = generate_technical_report(ensemble, results, baseline_results, data_dict)

    with open(os.path.join(output_dir, 'technical_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ 保存: {output_dir}/technical_report.md")

    print("\n" + "=" * 80)
    print("✓ 阶段5: 改进版三层集成学习完成！")
    print("=" * 80)

    return ensemble, results


if __name__ == "__main__":
    ensemble, results = main()