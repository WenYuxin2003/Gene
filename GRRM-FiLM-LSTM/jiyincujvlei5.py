import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ================================
# 路径设置
# ================================
GENE_FEAT_PATH = "results/film_cluster_analysis/gene_features.npy"
TRAIN_GEN_PATH = "results/generated_train_from_pkl.csv"
VALID_GEN_PATH = "results/generated_valid_from_pkl.csv"
OUTPUT_PATH = "results/film_cluster_analysis/cluster5_generated_molecules.csv"

# ================================
# 加载数据
# ================================
gene_features = np.load(GENE_FEAT_PATH)
train_gen = pd.read_csv(TRAIN_GEN_PATH)
valid_gen = pd.read_csv(VALID_GEN_PATH)
all_generated = pd.concat([train_gen, valid_gen], ignore_index=True)

print(f"🧬 gene_features shape: {gene_features.shape}")
print(f"📊 all_generated shape: {all_generated.shape}")

# 对齐长度
min_len = min(len(gene_features), len(all_generated))
gene_features = gene_features[:min_len]
all_generated = all_generated.iloc[:min_len]

# ================================
# 聚类
# ================================
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(gene_features)

# ================================
# 取 cluster 5
# ================================
cluster5_idx = np.where(clusters == 5)[0]
print(f"🔍 Cluster 5 contains {len(cluster5_idx)} samples (aligned to {min_len} total).")

cluster5_mols = all_generated.iloc[cluster5_idx]
cluster5_mols.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved Cluster 5 molecules to {OUTPUT_PATH}")
