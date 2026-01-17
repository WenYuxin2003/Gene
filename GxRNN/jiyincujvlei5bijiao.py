# -*- coding: utf-8 -*-
"""
Cluster 5 Interpretability Analysis
分析第 5 簇的独立性与可解释性：
1. α / β 调制相似度
2. gate 强度分布
3. 生成分子与训练集的新颖性
4. 基因特征空间的 PCA 可视化
Author: Yuxin
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# ========== 路径设置 ==========
BASE_DIR = os.path.dirname(__file__)
RESULT_DIR = os.path.join(BASE_DIR, "results", "film_cluster_analysis")

# FiLM 参数与聚类结果
alpha_path = os.path.join(RESULT_DIR, "film_alpha.npy")
beta_path = os.path.join(RESULT_DIR, "film_beta.npy")
gate_path = os.path.join(RESULT_DIR, "film_gate.npy")
cluster_path = os.path.join(RESULT_DIR, "clusters.npy")
gene_feat_path = os.path.join(RESULT_DIR, "gene_features.npy")

# 生成的分子
cluster5_file = os.path.join(RESULT_DIR, "cluster5_generated_molecules.csv")
train_file = os.path.join(BASE_DIR, "datasets", "LINCS", "mcf7.csv")

# 检查文件
for f in [alpha_path, beta_path, gate_path, cluster_path, gene_feat_path, cluster5_file]:
    assert os.path.exists(f), f"❌ 文件不存在: {f}"
print("✅ 所有必要文件已找到")

# ========== 加载数据 ==========
alpha = np.load(alpha_path)
beta = np.load(beta_path)
gate = np.load(gate_path)
clusters = np.load(cluster_path)
gene_feat = np.load(gene_feat_path)
cluster5_idx = np.where(clusters == 5)[0]

print(f"📊 Cluster 5 样本数: {len(cluster5_idx)} / {len(clusters)}")

# ========== 1️⃣ α / β 相似度矩阵 ==========
means_alpha = alpha  # alpha 本身是每簇均值
means_beta = beta    # beta 本身是每簇均值

sim_alpha = cosine_similarity(means_alpha)
sim_beta = cosine_similarity(means_beta)
plt.figure(figsize=(6, 5))
sns.heatmap(sim_alpha, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cosine Similarity between Cluster α Vectors")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "cluster_alpha_similarity.png"), dpi=300)

plt.figure(figsize=(6, 5))
sns.heatmap(sim_beta, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cosine Similarity between Cluster β Vectors")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "cluster_beta_similarity.png"), dpi=300)

print("📈 已保存 α / β 相似度热图")

# ========== 2️⃣ gate 强度分布 ==========
# ========== 2️⃣ gate 强度分布 ==========
plt.figure(figsize=(6, 4))

if len(gate) == len(clusters):
    # 每个样本都有 gate → 用 violinplot
    gate_mean = gate.mean(axis=1)
    sns.violinplot(x=clusters, y=gate_mean, inner="quartile", scale="width", cut=0)
    plt.title("Gate Strength Distribution across Clusters")
    plt.ylabel("Mean Gate Value")
else:
    # gate 只有每簇均值 → 用柱状图
    gate_mean = gate.mean(axis=1) if gate.ndim == 2 else gate
    plt.bar(range(len(gate_mean)), gate_mean, color="skyblue")
    plt.title("Average Gate Strength per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Gate Mean")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "cluster_gate_distribution.png"), dpi=300)
print("📈 已保存 Gate 分布图")

# ========== 3️⃣ Tanimoto 相似度（Cluster 5 vs 训练集） ==========
cluster5_smiles = pd.read_csv(cluster5_file)["SMILES"].dropna().tolist()
train_df = pd.read_csv(train_file, header=None)
# 自动添加列名
num_cols = train_df.shape[1]
if num_cols >= 2:
    train_df.columns = ["inchikey", "smiles"] + [f"gene{i}" for i in range(1, num_cols - 1)]
else:
    raise ValueError(f"文件格式错误：{train_file} 至少需要包含 inchikey 和 smiles 两列")

# 自动检测 SMILES 列名
smiles_col = None
for c in train_df.columns:
    if "smiles" in c.lower():
        smiles_col = c
        break
if smiles_col is None:
    raise KeyError(f"❌ 未找到 SMILES 列，请检查文件列名: {train_df.columns.tolist()}")

train_smiles = train_df[smiles_col].dropna().tolist()


def tanimoto_mean(smiles_a, smiles_b, n_a=200, n_b=2000):
    """计算平均 Tanimoto 相似度（随机采样）"""
    import random
    smiles_a = random.sample(smiles_a, min(len(smiles_a), n_a))
    smiles_b = random.sample(smiles_b, min(len(smiles_b), n_b))
    fps_a = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024) for s in smiles_a if Chem.MolFromSmiles(s)]
    fps_b = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024) for s in smiles_b if Chem.MolFromSmiles(s)]
    sims = []
    for fp in fps_a:
        sim = DataStructs.BulkTanimotoSimilarity(fp, fps_b)
        sims.append(np.mean(sim))
    return np.mean(sims)

sim_score = tanimoto_mean(cluster5_smiles, train_smiles)
print(f"🔹 Cluster 5 平均相似度（vs 训练集）: {sim_score:.3f}")

# ========== 4️⃣ 基因特征 PCA ==========
pca = PCA(n_components=2)
pca_feat = pca.fit_transform(gene_feat)
plt.figure(figsize=(6, 5))
palette = sns.color_palette("hls", 10)
sns.scatterplot(x=pca_feat[:, 0], y=pca_feat[:, 1], hue=clusters, palette=palette, s=10, alpha=0.6)
plt.title("Gene Feature PCA Colored by Cluster")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "gene_feature_pca.png"), dpi=300)
print("📈 已保存基因特征 PCA 图")

# ========== 总结输出 ==========
print("\n✅ 分析完成：结果文件已保存到 results/film_cluster_analysis/")
print("""
生成内容：
- cluster_alpha_similarity.png / cluster_beta_similarity.png → α、β 相似度热图
- cluster_gate_distribution.png → gate 平均值分布
- gene_feature_pca.png → 基因特征分布图
- 平均 Tanimoto 相似度值 printed on console
""")
