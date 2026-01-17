# -*- coding: utf-8 -*-
"""
FiLM Gene Cluster Interpretability Analysis (with auto-folder creation)
Author: 文敔鑫
Date: 2025-10-28
Purpose:
    Visualize how different gene clusters modulate FiLM parameters (α, β, gate)
    in the trained GxRNN model under transcriptional conditioning.
    Automatically saves all results into results/film_cluster_analysis/
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================
# 1. Load Model & Gene Features
# ==============================
from xin4_GxRNNstar import GxRNN  # ✅ 确保同目录下有 GxRNN.py

# === User Parameters ===
MODEL_PATH = "results/saved_gxrnn.pkl_450.pkl"  # ✅ 你的最终模型
GENE_EXPRESSION_FILE = "datasets/LINCS/mcf7.csv"  # ✅ 你的表达矩阵路径
RESULT_DIR = "results/film_cluster_analysis"      # ✅ 自动创建保存路径
NUM_GENES = 978
NUM_CLUSTERS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Create result folder ===
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"📁 Results will be saved in: {RESULT_DIR}")

# ==============================
# 2. Load Gene Expression Data
# ==============================
data = pd.read_csv(
    GENE_EXPRESSION_FILE,
    names=['inchikey', 'smiles'] + [f'gene{i}' for i in range(1, NUM_GENES + 1)]
)
gene_matrix = data.iloc[:, 2:].values.astype('float32')
gene_tensor = torch.tensor(gene_matrix, dtype=torch.float32).to(DEVICE)

# ==============================
# 3. Load Model
# ==============================
# 伪tokenizer，只为加载模型
# ==============================
# 3. Load Model (match training params)
# ==============================
# 伪tokenizer，只为加载模型
tokenizer_stub = type('', (), {})()
tokenizer_stub.vocab_size = 54   # ✅ 改成你实际 tokenizer 的 vocab size
tokenizer_stub.char_to_int = {'<pad>': 0}
tokenizer_stub.pad = '<pad>'

model = GxRNN(
    tokenizer=tokenizer_stub,
    emb_size=128,           # ✅ 你的训练模型实际是128维embedding
    hidden_size=1024,       # ✅ 从文件名“hidden1024”也能验证
    gene_latent_size=978,
    num_layers=3,
    dropout=0.3,
    star_core_dim=256,
    gene_feature_dim=256,
    star_num_heads=4
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")


# ==============================
# 4. Encode Gene Features
# ==============================
with torch.no_grad():
    gene_feat = model._encode_gene(gene_tensor)  # [N, gene_feature_dim]

gene_feat_np = gene_feat.cpu().numpy()
np.save(os.path.join(RESULT_DIR, "gene_features.npy"), gene_feat_np)
print(f"💾 Encoded gene features saved to {RESULT_DIR}/gene_features.npy")

# ==============================
# 5. Cluster Gene Features
# ==============================
print(f"🔹 Clustering {gene_feat_np.shape[0]} samples into {NUM_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
cluster_ids = kmeans.fit_predict(gene_feat_np)
pd.DataFrame({"cluster_id": cluster_ids}).to_csv(os.path.join(RESULT_DIR, "cluster_assignments.csv"), index=False)

# 聚类中心均值
cluster_means = []
for i in range(NUM_CLUSTERS):
    cluster_means.append(gene_feat[cluster_ids == i].mean(dim=0))
cluster_means = torch.stack(cluster_means)  # [C, F]

# ==============================
# 6. Compute FiLM Parameters (α, β, gate)
# ==============================
with torch.no_grad():
    alpha = model.film.alpha_net(cluster_means).cpu().numpy()
    beta  = model.film.beta_net(cluster_means).cpu().numpy()
    gate  = model.film.gate_net(cluster_means).cpu().numpy()

np.save(os.path.join(RESULT_DIR, "film_alpha.npy"), alpha)
np.save(os.path.join(RESULT_DIR, "film_beta.npy"), beta)
np.save(os.path.join(RESULT_DIR, "film_gate.npy"), gate)

# ==============================
# 7. Visualization Functions
# ==============================
def plot_heatmap(matrix, title, save_name):
    plt.figure(figsize=(12, 5))
    sns.heatmap(matrix, cmap="coolwarm", center=0, cbar_kws={'label': 'Modulation Strength'})
    plt.xlabel("FiLM Channel (embedding dimension)")
    plt.ylabel("Gene Cluster ID")
    plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved: {save_path}")

# 绘制热图
plot_heatmap(alpha, "α (Scaling) - Gene Cluster Modulation", "film_alpha_heatmap.png")
plot_heatmap(beta,  "β (Shifting) - Gene Cluster Modulation", "film_beta_heatmap.png")
plot_heatmap(gate,  "Gate - Information Fusion Strength", "film_gate_heatmap.png")

# ==============================
# 8. PCA Visualization
# ==============================
pca = PCA(n_components=2)
film_flat = np.concatenate([alpha, beta], axis=1)
film_pca = pca.fit_transform(film_flat)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=film_pca[:, 0], y=film_pca[:, 1],
                hue=np.arange(NUM_CLUSTERS), palette="tab10", s=80, edgecolor='black')
plt.title("FiLM Parameter Space (α + β) Cluster Projection", fontsize=13)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Gene Cluster", loc='best')
plt.tight_layout()
pca_path = os.path.join(RESULT_DIR, "film_cluster_pca.png")
plt.savefig(pca_path, dpi=300)
plt.close()
print(f"📈 Saved PCA scatter: {pca_path}")

print("🎯 All analysis complete! Files saved to:", RESULT_DIR)
