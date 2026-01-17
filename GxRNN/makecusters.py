# -*- coding: utf-8 -*-
"""
重新生成 clusters.npy 文件，用于 interpretability 分析
"""
import numpy as np
from sklearn.cluster import KMeans
import os

RESULT_DIR = "results/film_cluster_analysis"
gene_feat_path = os.path.join(RESULT_DIR, "gene_features.npy")
cluster_path = os.path.join(RESULT_DIR, "clusters.npy")

assert os.path.exists(gene_feat_path), f"❌ 找不到基因特征文件: {gene_feat_path}"

print("✅ 加载基因特征中...")
gene_feat = np.load(gene_feat_path)
print("gene_features shape:", gene_feat.shape)

print("🔹 开始 KMeans 聚类 (n_clusters=10)...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(gene_feat)

np.save(cluster_path, clusters)
print(f"✅ 聚类完成，已保存: {cluster_path}")
print("簇样本数量统计：", np.bincount(clusters))
