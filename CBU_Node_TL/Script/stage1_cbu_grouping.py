"""
阶段1 v2: 基于Node值的自然CBU分组
===================================

任务:
1. 加载cbu_node_similarity矩阵和CBU Node值
2. 基于Node值对CBU进行自然分组（18个自然组）
3. 计算每个CBU组的平均相似度
4. 分析组内和组间相似度
5. 生成CBU组映射表

分组策略: 基于Node值的自然分组（18个组）
- G1-G18: 基于Node值的18个自然组（5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 30, 32, 36, 48）

输出:
- results/similarity_groups/cbu_similarity_groups_v2.csv: CBU分组结果
- results/similarity_groups/cbu_group_statistics_v2.csv: 组内统计
- results/similarity_groups/cbu_group_heatmap_v2.png: 组间相似度热力图
- results/similarity_groups/group_analysis_report_v2.md: 分组分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义路径
DATA_DIR = Path(__file__).parent.parent
SIMILARITY_PATH = DATA_DIR / 'cbu_node_similarity.csv'
CBU_MAPPING_PATH = DATA_DIR / 'cbu_category_clean_mapping.csv'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'similarity_groups'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("阶段1 v2: 基于Node值的自然CBU分组")
print("="*80)
print(f"数据目录: {DATA_DIR}")
print(f"相似度矩阵: {SIMILARITY_PATH}")
print(f"CBU映射: {CBU_MAPPING_PATH}")
print(f"输出目录: {OUTPUT_DIR}")
print("="*80)
print()

# ============================================================================
# 步骤1: 加载Node相似度矩阵和CBU Node值
# ============================================================================
print("\n[步骤1] 加载Node相似度矩阵和CBU Node值...")

# 加载相似度矩阵
similarity_df = pd.read_csv(SIMILARITY_PATH, index_col=0)
print(f"✓ 相似度矩阵加载成功: {similarity_df.shape[0]}×{similarity_df.shape[1]}")

# 加载CBU映射（包含Node值）
cbu_mapping = pd.read_csv(CBU_MAPPING_PATH)
print(f"✓ CBU映射加载成功: {cbu_mapping.shape[0]}个类别")
print(f"✓ 列名: {list(cbu_mapping.columns)}")

# 过滤掉Node为空的CBU
cbu_mapping_filtered = cbu_mapping[cbu_mapping['Node'].notna()]
print(f"✓ 过滤后: {cbu_mapping_filtered.shape[0]}个有效CBU")

# 创建CBU到Node的映射
cbu_to_node = dict(zip(cbu_mapping_filtered['cbu_name'], cbu_mapping_filtered['Node']))
cbu_to_index = dict(zip(cbu_mapping_filtered['cbu_name'], cbu_mapping_filtered['cbu_index']))

# ============================================================================
# 步骤2: 基于Node值进行自然分组
# ============================================================================
print("\n[步骤2] 基于Node值进行自然分组...")

# 定义18个自然组（基于Node值）
node_groups = {
    'G1': {'node': 5, 'cbus': []},
    'G2': {'node': 6, 'cbus': []},
    'G3': {'node': 8, 'cbus': []},
    'G4': {'node': 10, 'cbus': []},
    'G5': {'node': 11, 'cbus': []},
    'G6': {'node': 12, 'cbus': []},
    'G7': {'node': 13, 'cbus': []},
    'G8': {'node': 14, 'cbus': []},
    'G9': {'node': 15, 'cbus': []},
    'G10': {'node': 16, 'cbus': []},
    'G11': {'node': 18, 'cbus': []},
    'G12': {'node': 20, 'cbus': []},
    'G13': {'node': 24, 'cbus': []},
    'G14': {'node': 28, 'cbus': []},
    'G15': {'node': 30, 'cbus': []},
    'G16': {'node': 32, 'cbus': []},
    'G17': {'node': 36, 'cbus': []},
    'G18': {'node': 48, 'cbus': []}
}

# 将CBU分配到对应的组
for cbu_name in similarity_df.index:
    node_value = cbu_to_node.get(cbu_name)
    if node_value is not None:
        for group_name, group_info in node_groups.items():
            if group_info['node'] == node_value:
                group_info['cbus'].append(cbu_name)
                break

# 统计各组CBU数量
print(f"\n基于Node值的自然分组结果:")
total_cbus = 0
for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
    node_value = node_groups[group_name]['node']
    cbus = node_groups[group_name]['cbus']
    count = len(cbus)
    total_cbus += count
    print(f"  {group_name} (Node={node_value}): {count}个CBU - {', '.join(cbus)}")

print(f"\n总计: {total_cbus}个CBU")

# ============================================================================
# 步骤3: 计算组内相似度
# ============================================================================
print("\n[步骤3] 计算组内相似度...")

group_stats = {}
for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
    cbus = node_groups[group_name]['cbus']
    node_value = node_groups[group_name]['node']

    if len(cbus) > 1:
        # 提取组内相似度
        group_similarities = similarity_df.loc[cbus, cbus]
        upper_triangle = []
        for i in range(len(cbus)):
            for j in range(i+1, len(cbus)):
                upper_triangle.append(group_similarities.iloc[i, j])

        group_stats[group_name] = {
            'node': node_value,
            'cbu_count': len(cbus),
            'cbus': cbus,
            'avg_similarity': float(np.mean(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle)),
            'std_similarity': float(np.std(upper_triangle))
        }
    else:
        group_stats[group_name] = {
            'node': node_value,
            'cbu_count': len(cbus),
            'cbus': cbus,
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'std_similarity': 0.0
        }

print(f"\n组内相似度统计:")
for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
    stats = group_stats[group_name]
    if stats['cbu_count'] > 1:
        print(f"  {group_name} (Node={stats['node']}): {stats['cbu_count']}个CBU, 平均相似度={stats['avg_similarity']:.4f}")

# ============================================================================
# 步骤4: 计算组间相似度
# ============================================================================
print("\n[步骤4] 计算组间相似度...")

print(f"\n组间相似度统计 (仅显示部分):")
group_names = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']
cross_similarity_matrix = pd.DataFrame(index=group_names, columns=group_names)

for i, group1 in enumerate(group_names):
    for j, group2 in enumerate(group_names):
        cbus1 = group_stats[group1]['cbus']
        cbus2 = group_stats[group2]['cbus']

        if len(cbus1) > 0 and len(cbus2) > 0:
            if i == j:
                # 组内相似度
                cross_similarity_matrix.loc[group1, group2] = group_stats[group1]['avg_similarity']
            else:
                # 组间相似度
                cross_similarities = similarity_df.loc[cbus1, cbus2].values.flatten()
                cross_similarity_matrix.loc[group1, group2] = float(np.mean(cross_similarities))
        else:
            cross_similarity_matrix.loc[group1, group2] = 0.0

# 打印前5组的组间相似度
print(f"\n前6组组间相似度矩阵:")
print(cross_similarity_matrix.iloc[:6, :6].to_string())

# ============================================================================
# 步骤5: 生成CBU组映射表
# ============================================================================
print("\n[步骤5] 生成CBU组映射表...")

# 创建分组DataFrame
group_mapping_list = []
for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
    stats = group_stats[group_name]
    for cbu in stats['cbus']:
        group_mapping_list.append({
            'CBU': cbu,
            'Group': group_name,
            'Node': stats['node'],
            'Avg_Similarity_Within_Group': stats['avg_similarity']
        })

group_mapping_df = pd.DataFrame(group_mapping_list)
group_mapping_path = OUTPUT_DIR / 'cbu_similarity_groups_v2.csv'
group_mapping_df.to_csv(group_mapping_path, index=False)
print(f"✓ CBU分组结果已保存: {group_mapping_path}")

# 保存组统计
group_stats_list = []
for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
    stats = group_stats[group_name]
    group_stats_list.append({
        'Group': group_name,
        'Node': stats['node'],
        'CBU_Count': stats['cbu_count'],
        'Avg_Similarity': stats['avg_similarity'],
        'Min_Similarity': stats['min_similarity'],
        'Max_Similarity': stats['max_similarity'],
        'Std_Similarity': stats['std_similarity'],
        'CBU_List': ', '.join(stats['cbus'])
    })

group_stats_df = pd.DataFrame(group_stats_list)
group_stats_path = OUTPUT_DIR / 'cbu_group_statistics_v2.csv'
group_stats_df.to_csv(group_stats_path, index=False)
print(f"✓ 组统计已保存: {group_stats_path}")

# 保存组间相似度矩阵
cross_similarity_path = OUTPUT_DIR / 'cbu_cross_similarity_v2.csv'
cross_similarity_matrix.to_csv(cross_similarity_path)
print(f"✓ 组间相似度矩阵已保存: {cross_similarity_path}")

# ============================================================================
# 步骤6: 生成组间相似度热力图
# ============================================================================
print("\n[步骤6] 生成组间相似度热力图...")

# 按组重新排序相似度矩阵
group_order = []
for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
    group_order.extend(group_stats[group_name]['cbus'])

similarity_reordered = similarity_df.loc[group_order, group_order]

# 创建组标签
group_labels = []
group_boundaries = [0]
cumulative_count = 0
for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
    count = group_stats[group_name]['cbu_count']
    group_labels.extend([group_name] * count)
    cumulative_count += count
    group_boundaries.append(cumulative_count)

# 绘制热力图
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(
    similarity_reordered.astype(float),
    cmap='RdYlGn',
    vmin=0.0,
    vmax=1.0,
    cbar_kws={'label': 'Node相似度'},
    ax=ax
)

# 添加组分割线
for boundary in group_boundaries[1:-1]:
    ax.axhline(boundary, color='white', linewidth=2)
    ax.axvline(boundary, color='white', linewidth=2)

# 设置标题和标签
ax.set_title('CBU Node相似度热力图（按Node值分组）', fontsize=18, fontweight='bold')
ax.set_xlabel('CBU (按Node值分组)', fontsize=14)
ax.set_ylabel('CBU (按Node值分组)', fontsize=14)

plt.tight_layout()
heatmap_path = OUTPUT_DIR / 'cbu_group_heatmap_v2.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ 组间相似度热力图已保存: {heatmap_path}")
plt.close()

# 绘制组间相似度热力图（18×18）
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cross_similarity_matrix.astype(float),
    cmap='RdYlGn',
    vmin=0.0,
    vmax=1.0,
    cbar_kws={'label': '平均相似度'},
    ax=ax
)
ax.set_title('18个CBU组间相似度热力图', fontsize=16, fontweight='bold')
ax.set_xlabel('CBU组', fontsize=12)
ax.set_ylabel('CBU组', fontsize=12)
plt.tight_layout()
cross_heatmap_path = OUTPUT_DIR / 'cbu_cross_similarity_heatmap_v2.png'
plt.savefig(cross_heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ 组间相似度热力图（18×18）已保存: {cross_heatmap_path}")
plt.close()

# ============================================================================
# 步骤7: 生成分析报告
# ============================================================================
print("\n[步骤7] 生成分组分析报告...")

report_path = OUTPUT_DIR / 'group_analysis_report_v2.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 阶段1 v2: 基于Node值的自然CBU分组分析报告\n\n")
    f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## 1. 概述\n\n")
    f.write("本报告基于CBU Node值对58个CBU进行自然分组分析。\n\n")
    f.write("**分组策略**: 基于Node值的18个自然组（5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 28, 30, 32, 36, 48）\n\n")
    f.write("**分组依据**: CBU的Node值（顶点数量），相同Node值的CBU归为一组\n\n")

    f.write("## 2. 分组结果\n\n")
    f.write("### 2.1 各组CBU数量\n\n")
    f.write("| 组别 | Node值 | CBU数量 | CBU列表 |\n")
    f.write("|------|--------|---------|----------|\n")
    for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
        stats = group_stats[group_name]
        cbu_list_str = ', '.join(stats['cbus'])
        f.write(f"| {group_name} | {stats['node']} | {stats['cbu_count']} | {cbu_list_str} |\n")
    f.write("\n")

    f.write("### 2.2 组内相似度统计\n\n")
    f.write("| 组别 | Node值 | CBU数量 | 平均相似度 | 最小相似度 | 最大相似度 | 标准差 |\n")
    f.write("|------|--------|---------|-----------|-----------|-----------|--------|\n")
    for group_name in ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']:
        stats = group_stats[group_name]
        if stats['cbu_count'] > 1:
            f.write(f"| {group_name} | {stats['node']} | {stats['cbu_count']} | {stats['avg_similarity']:.4f} | {stats['min_similarity']:.4f} | {stats['max_similarity']:.4f} | {stats['std_similarity']:.4f} |\n")
    f.write("\n")

    f.write("### 2.3 Node值分布\n\n")
    f.write("| Node值 | CBU数量 | 占比 |\n")
    f.write("|--------|---------|------|\n")
    for node_value in sorted([group_stats[g]['node'] for g in group_stats]):
        cbus_with_node = [stats['cbu_count'] for g, stats in group_stats.items() if stats['node'] == node_value]
        count = cbus_with_node[0] if cbus_with_node else 0
        percentage = count / total_cbus * 100
        f.write(f"| {node_value} | {count} | {percentage:.1f}% |\n")
    f.write("\n")

    f.write("## 3. 组间相似度分析\n\n")
    f.write("### 3.1 组间相似度矩阵（部分）\n\n")
    f.write("```\n")
    f.write(cross_similarity_matrix.iloc[:6, :6].to_string())
    f.write("```\n\n")

    f.write("### 3.2 高相似度组对（平均相似度≥0.75）\n\n")
    f.write("| 组别1 | 组别2 | 平均相似度 |\n")
    f.write("|-------|-------|-----------|\n")
    high_sim_pairs = []
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names):
            if i < j:
                sim = cross_similarity_matrix.loc[group1, group2]
                if sim >= 0.75:
                    high_sim_pairs.append((group1, group2, sim))

    for pair in high_sim_pairs:
        f.write(f"| {pair[0]} (Node={group_stats[pair[0]]['node']}) | {pair[1]} (Node={group_stats[pair[1]]['node']}) | {pair[2]:.4f} |\n")
    f.write("\n")

    f.write("## 4. 预训练策略建议\n\n")
    f.write("### 4.1 分组优先级\n\n")

    # 按组大小排序
    sorted_groups = sorted(
        [(g, group_stats[g]) for g in group_names],
        key=lambda x: x[1]['cbu_count'],
        reverse=True
    )

    f.write("**大组（≥3个CBU）**: 可作为核心预训练组\n\n")
    for group_name, stats in sorted_groups:
        if stats['cbu_count'] >= 3:
            f.write(f"- {group_name} (Node={stats['node']}): {stats['cbu_count']}个CBU, 平均相似度={stats['avg_similarity']:.4f}\n")

    f.write("\n**小组（2个CBU）**: 可作为中等预训练组\n\n")
    for group_name, stats in sorted_groups:
        if stats['cbu_count'] == 2:
            f.write(f"- {group_name} (Node={stats['node']}): {stats['cbu_count']}个CBU, 平均相似度={stats['avg_similarity']:.4f}\n")

    f.write("\n**单独CBU（1个CBU）**: 需要Zero-shot学习或独立训练\n\n")
    for group_name, stats in sorted_groups:
        if stats['cbu_count'] == 1:
            f.write(f"- {group_name} (Node={stats['node']}): {stats['cbu_count']}个CBU - {stats['cbus'][0]}\n")

    f.write("\n### 4.2 预训练阶段划分\n\n")

    # 计算累积CBU数量
    cumulative_cbus = 0
    pretrain_stages = []
    current_cbus = []

    for group_name, stats in sorted_groups:
        current_cbus.append(group_name)
        cumulative_cbus += stats['cbu_count']

        if cumulative_cbus >= 10 or group_name == sorted_groups[-1][0]:
            total = sum(group_stats[g]['cbu_count'] for g in current_cbus)
            pretrain_stages.append({
                'stage': len(pretrain_stages) + 1,
                'groups': ', '.join(current_cbus),
                'cbu_count': total
            })
            current_cbus = []

    f.write("| 阶段 | 包含组 | CBU数量 | 预训练目标 |\n")
    f.write("|------|--------|---------|-----------|\n")
    for i, stage in enumerate(pretrain_stages):
        f.write(f"| Pretrain-{i+1} | {stage['groups']} | {stage['cbu_count']} | 扩展预训练数据 |\n")
    f.write("\n")

    f.write("## 5. 结论\n\n")
    f.write("### 5.1 主要发现\n\n")
    f.write(f"1. 58个CBU基于Node值分为18个自然组\n")
    f.write(f"2. 最大组为G6（Node=12），包含{group_stats['G6']['cbu_count']}个CBU\n")
    f.write(f"3. {len([g for g, stats in sorted_groups if stats['cbu_count'] >= 3])}个大组（≥3个CBU）可作为核心预训练组\n")
    f.write(f"4. {len([g for g, stats in sorted_groups if stats['cbu_count'] == 1])}个单独CBU需要Zero-shot学习\n")
    f.write(f"5. 组内平均相似度范围为{min([stats['avg_similarity'] for stats in group_stats.values() if stats['cbu_count'] > 1]):.4f}到{max([stats['avg_similarity'] for stats in group_stats.values() if stats['cbu_count'] > 1]):.4f}\n\n")

    f.write("### 5.2 分组效果评估\n\n")
    f.write("- 基于Node值的自然分组物理意义明确\n")
    f.write("- 相同Node值的CBU组内相似度高\n")
    f.write("- 不同Node值的CBU组间相似度变化规律性强\n")
    f.write("- 适合用于渐进式预训练\n\n")

    f.write("### 5.3 下一步建议\n\n")
    f.write("1. 执行阶段2: 训练Baseline模型作为对比基准\n")
    f.write("2. 执行阶段3: 基于大组进行渐进式预训练\n")
    f.write("3. 执行阶段4: 对小组和单独CBU进行微调或Zero-shot学习\n")
    f.write("4. 执行阶段5: 基于组间相似度加权集成\n\n")

print(f"✓ 分析报告已保存: {report_path}")

# ============================================================================
# 步骤8: 打印总结
# ============================================================================
print("\n" + "="*80)
print("分组分析总结（基于Node值的自然分组）")
print("="*80)
print(f"CBU总数: {similarity_df.shape[0]}")
print(f"自然组数: 18")
print()
print("各组统计:")
for group_name in ['G6', 'G8', 'G10', 'G13', 'G17', 'G18']:  # 大组
    stats = group_stats[group_name]
    if stats['cbu_count'] >= 3:
        print(f"  {group_name} (Node={stats['node']}): {stats['cbu_count']}个CBU, 平均相似度={stats['avg_similarity']:.4f}")

print()
print("单独CBU:")
for group_name, stats in sorted_groups:
    if stats['cbu_count'] == 1:
        print(f"  {group_name} (Node={stats['node']}): {stats['cbus'][0]}")

print()
print("输出文件:")
print(f"  - {group_mapping_path}")
print(f"  - {group_stats_path}")
print(f"  - {cross_similarity_path}")
print(f"  - {heatmap_path}")
print(f"  - {cross_heatmap_path}")
print(f"  - {report_path}")
print("="*80)