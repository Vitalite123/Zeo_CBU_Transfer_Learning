"""
阶段1: CBU分组分析（TLCBU_vGlobe）
基于全局特征相似度的迁移学习项目

任务：
1. 分析全局相似度矩阵统计特性
2. 基于相似度阈值对CBU进行分组
3. 生成CBU组映射表

输出：
- results/similarity_groups/global_cbu_groups.csv
- results/similarity_groups/group_statistics.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ========== 路径配置 ==========
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'
SIMILARITY_DIR = RESULTS_DIR / 'similarity_groups'

# 相似度矩阵文件
SIMILARITY_PATH = Path(__file__).parent.parent / 'cbu_similarity_cleaned.csv'

# 相似度阈值 - 动态（基于实际分位数）
# 将在group_cbus_by_similarity中根据数据自动计算
SIMILARITY_THRESHOLDS = {
    'very_high': 'auto',  # Q75
    'high': 'auto',       # Q50
    'medium': 'auto',     # Q25
    'low': 'auto'        # 其余
}


def load_similarity_matrix():
    """加载相似度矩阵"""
    print("\n[1/3] 加载全局相似度矩阵...")

    similarity_df = pd.read_csv(SIMILARITY_PATH, index_col=0)
    print(f"  矩阵大小: {similarity_df.shape}")

    cbu_list = similarity_df.index.tolist()
    print(f"  CBU数量: {len(cbu_list)}")

    return similarity_df, cbu_list


def analyze_similarity_distribution(sim_df):
    """分析相似度分布"""
    print("\n[2/3] 分析相似度分布...")

    sim_matrix = sim_df.values.copy()
    np.fill_diagonal(sim_matrix, 0)

    upper_tri = np.triu(sim_matrix, k=1)
    upper_tri[upper_tri == 0] = np.nan
    upper_tri = upper_tri.flatten()
    upper_tri = upper_tri[~np.isnan(upper_tri)]

    stats = {
        'total_pairs': len(upper_tri),
        'mean': float(np.mean(upper_tri)),
        'std': float(np.std(upper_tri)),
        'min': float(np.min(upper_tri)),
        'max': float(np.max(upper_tri)),
        'median': float(np.median(upper_tri)),
        'q25': float(np.percentile(upper_tri, 25)),
        'q75': float(np.percentile(upper_tri, 75))
    }

    print(f"  总对数: {stats['total_pairs']}")
    print(f"  平均相似度: {stats['mean']:.4f}")
    print(f"  相似度范围: [{stats['min']:.4f}, {stats['max']:.4f}]")

    return stats, upper_tri


def group_cbus_by_similarity(sim_df, cbu_list):
    """基于相似度阈值对CBU进行分组 - 使用实际分位数"""
    print("\n[3/3] CBU分组...")

    sim_matrix = sim_df.values.copy()
    np.fill_diagonal(sim_matrix, 1.0)

    avg_sims_all = []
    for i, cbu_i in enumerate(cbu_list):
        avg_sims = []
        for j, cbu_j in enumerate(cbu_list):
            if i != j:
                avg_sims.append(sim_matrix[i, j])
        if avg_sims:
            avg_sims_all.append(np.mean(avg_sims))

    q75 = np.percentile(avg_sims_all, 75)
    q50 = np.percentile(avg_sims_all, 50)
    q25 = np.percentile(avg_sims_all, 25)

    print(f"  相似度分位数: Q75={q75:.4f}, Q50={q50:.4f}, Q25={q25:.4f}")

    groups = {
        'very_high': [],
        'high': [],
        'medium': [],
        'low': []
    }

    for i, cbu_i in enumerate(cbu_list):
        avg_sims = []
        for j, cbu_j in enumerate(cbu_list):
            if i != j:
                avg_sims.append(sim_matrix[i, j])

        if avg_sims:
            avg_sim = np.mean(avg_sims)

            if avg_sim >= q75:
                groups['very_high'].append(cbu_i)
            elif avg_sim >= q50:
                groups['high'].append(cbu_i)
            elif avg_sim >= q25:
                groups['medium'].append(cbu_i)
            else:
                groups['low'].append(cbu_i)

    for group_name, cbus in groups.items():
        print(f"  {group_name}组: {len(cbus)}个CBU")

    return groups


def create_group_mapping(sim_df, cbu_list, groups):
    """创建CBU组映射表"""
    print("\n创建组映射表...")

    group_data = []
    sim_matrix = sim_df.values.copy()

    for i, cbu in enumerate(cbu_list):
        avg_sims = []
        for j in range(len(cbu_list)):
            if i != j:
                avg_sims.append(sim_matrix[i, j])

        avg_sim = np.mean(avg_sims) if avg_sims else 0

        if cbu in groups['very_high']:
            group = 'G1_VeryHigh'
            tier = 1
        elif cbu in groups['high']:
            group = 'G2_High'
            tier = 2
        elif cbu in groups['medium']:
            group = 'G3_Medium'
            tier = 3
        else:
            group = 'G4_Low'
            tier = 4

        group_data.append({
            'cbu': cbu,
            'group': group,
            'tier': tier,
            'avg_similarity': avg_sim,
            'sample_count': 0
        })

    group_df = pd.DataFrame(group_data)
    group_df = group_df.sort_values('tier')

    group_df.to_csv(SIMILARITY_DIR / 'global_cbu_groups.csv', index=False)
    print(f"  保存: {SIMILARITY_DIR / 'global_cbu_groups.csv'}")

    tier_counts = group_df['tier'].value_counts().sort_index().to_dict()

    return group_df, tier_counts


def compute_group_statistics(sim_df, group_df, cbu_list):
    """计算组间/组内统计"""
    print("\n计算组统计...")

    sim_matrix = sim_df.values.copy()

    group_stats = {}

    for tier in range(1, 5):
        tier_cbus = group_df[group_df['tier'] == tier]['cbu'].tolist()
        tier_indices = [cbu_list.index(cbu) for cbu in tier_cbus if cbu in cbu_list]

        if len(tier_indices) >= 2:
            tier_sim = []
            for i in range(len(tier_indices)):
                for j in range(i+1, len(tier_indices)):
                    tier_sim.append(sim_matrix[tier_indices[i], tier_indices[j]])

            if tier_sim:
                group_stats[f'G{tier}'] = {
                    'count': len(tier_cbus),
                    'avg_within': float(np.mean(tier_sim)),
                    'std_within': float(np.std(tier_sim))
                }

    stats = {
        'thresholds': SIMILARITY_THRESHOLDS,
        'group_stats': group_stats
    }

    with open(SIMILARITY_DIR / 'group_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  保存: {SIMILARITY_DIR / 'group_statistics.json'}")

    return stats


def main():
    """主函数"""
    print("\n" + "="*60)
    print("TLCBU_vGlobe - 阶段1: CBU分组分析")
    print("="*60 + "\n")

    similarity_df, cbu_list = load_similarity_matrix()

    dist_stats, upper_tri = analyze_similarity_distribution(similarity_df)

    groups = group_cbus_by_similarity(similarity_df, cbu_list)

    group_df, tier_counts = create_group_mapping(similarity_df, cbu_list, groups)

    group_stats = compute_group_statistics(similarity_df, group_df, cbu_list)

    print("\n" + "="*60)
    print("分组结果摘要")
    print("="*60)

    print(f"\n相似度阈值:")
    for name, threshold in SIMILARITY_THRESHOLDS.items():
        print(f"  {name}: ≥{threshold}")

    print(f"\nCBU分组:")
    for group_name, cbus in groups.items():
        print(f"  {group_name}: {len(cbus)}个CBU")

    print(f"\n组内统计:")
    for group, stats in group_stats.get('group_stats', {}).items():
        print(f"  {group}: {stats['count']}个CBU, 组内平均{stats['avg_within']:.4f}")

    print("\n" + "="*60)
    print("阶段1完成!")
    print("="*60)

    return groups


if __name__ == '__main__':
    main()