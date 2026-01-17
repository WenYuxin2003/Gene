import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ 读取原始与聚类后数据
# ===============================
orig_df = pd.read_csv('datasets/LINCS/mcf7.csv', header=None)
reordered_df = pd.read_csv('datasets/LINCS_reordered/mcf7_reordered.csv', header=None)

# ===============================
# 2️⃣ 打印每个文件的前 5 行
# ===============================
print("原始文件前5行：")
print(orig_df.head())
print("\n聚类重排后文件前5行：")
print(reordered_df.head())
print("=" * 80)

# ===============================
# 3️⃣ 取基因表达矩阵部分绘制热力图
# ===============================
orig = orig_df.iloc[:, 2:]        # 去掉前两列 inchikey, smiles
reordered = reordered_df.iloc[:, 2:]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(orig.iloc[:100, :200], cmap='vlag', center=0)
plt.title('first (original order)')

plt.subplot(1, 2, 2)
sns.heatmap(reordered.iloc[:100, :200], cmap='vlag', center=0)
plt.title('second (cluster-reordered)')

plt.tight_layout()
plt.show()
