# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from xin4_GxRNNstar import GxRNN  # 若你保存成 GxRNN_star.py，请改成 from GxRNN_star import GxRNN
from utils import vocabulary
from types import SimpleNamespace
from tqdm import tqdm
import os

# ==============================
# 1️⃣ 参数配置
# ==============================
args = SimpleNamespace(
    gene_expression_file="datasets/LINCS_reordered/",
    cell_name="mcf7_reordered",
    gene_num=978,
    emb_size=128,
    hidden_size=512,
    num_layers=2,
    dropout=0.2,
)

BATCH_SIZE = 32
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ 使用设备: {device}")

# ==============================
# 2️⃣ 载入数据
# ==============================
path = os.path.join(args.gene_expression_file, f"{args.cell_name}.csv")
if not os.path.exists(path):
    raise FileNotFoundError(f"❌ 未找到文件: {path}")

data = pd.read_csv(path, header=None)
data = data.iloc[:1000]  # 为快速测试只取前 1000 条，可改大
print(f"✅ 数据加载成功: {data.shape[0]} 条样本, {data.shape[1]} 列")

smiles = data.iloc[:, 1].values
genes = torch.tensor(data.iloc[:, 2:].values, dtype=torch.float32)

# tokenizer
tokenizer = vocabulary(args)

def encode_smiles(s):
    return torch.tensor(tokenizer.encode(s), dtype=torch.long)

encoded = [encode_smiles(s) for s in smiles]
max_len = max(len(s) for s in encoded)
padded = torch.zeros(len(encoded), max_len, dtype=torch.long)
for i, s in enumerate(encoded):
    padded[i, :len(s)] = s

dataset = list(zip(padded, genes))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==============================
# 3️⃣ 初始化 STAR-GxRNN 模型
# ==============================
model = GxRNN(
    tokenizer=tokenizer,
    emb_size=args.emb_size,
    hidden_size=args.hidden_size,
    gene_latent_size=args.gene_num,
    num_layers=args.num_layers,
    dropout=args.dropout,
    star_core_dim=64,       # STAR 的核心维度
    gene_feature_dim=128,   # 基因融合后的特征维度
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.NLLLoss(ignore_index=tokenizer.char_to_int[tokenizer.pad])

# ==============================
# 4️⃣ 训练循环
# ==============================
loss_history = []
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for smiles_batch, genes_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        smiles_batch, genes_batch = smiles_batch.to(device), genes_batch.to(device)
        optimizer.zero_grad()
        outputs = model(smiles_batch, genes_batch)
        loss = criterion(outputs, smiles_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - 平均loss: {avg_loss:.4f}")

# ==============================
# 5️⃣ 绘制 loss 曲线
# ==============================
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
plt.title("Training Loss over Epochs (STAR-GxRNN)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/star_gxrnn_loss_curve.png", dpi=150)
plt.show()
print("📉 已保存 loss 曲线到 results/star_gxrnn_loss_curve.png")

# ==============================
# 6️⃣ 分子生成测试
# ==============================
model.eval()
with torch.no_grad():
    latent_vectors = genes[:5].to(device)
    generated = model.sample(max_len=80, latent_vectors=latent_vectors)

# decode back to smiles
def decode_smiles(tensor):
    return tokenizer.decode(tensor.tolist())

print("\n🧪 生成的前 5 条分子 SMILES：")
for i, gen in enumerate(generated):
    smiles_str = decode_smiles(gen)
    print(f"[{i+1}] {smiles_str}")

print("\n✅ STAR-GxRNN 测试完成！")
