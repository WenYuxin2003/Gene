import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from GxRNN import GxRNN    # 🔹 注意：导入原始论文版 GxRNN.py（不是 xin_GxRNN）
from utils import vocabulary
from types import SimpleNamespace
from tqdm import tqdm

# ===============================
# 1️⃣ 参数配置
# ===============================
args = SimpleNamespace(
    gene_expression_file="datasets/LINCS/",
    cell_name="mcf7",
    gene_num=978,
    emb_size=128,
    hidden_size=512,
    num_layers=2,
    dropout=0.2
)

BATCH_SIZE = 32
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ 使用设备: {device}")

# ===============================
# 2️⃣ 加载原始数据
# ===============================
path = f"{args.gene_expression_file}{args.cell_name}.csv"
data = pd.read_csv(path, header=None)
data = data.iloc[:1000]   # 可调整样本量
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

# ===============================
# 3️⃣ 初始化原始 GxRNN 模型
# ===============================
model = GxRNN(
    tokenizer=tokenizer,
    emb_size=args.emb_size,
    hidden_size=args.hidden_size,
    gene_latent_size=args.gene_num,  # gene 向量直接输入
    num_layers=args.num_layers,
    dropout=args.dropout
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.NLLLoss(ignore_index=tokenizer.char_to_int[tokenizer.pad])

# ===============================
# 4️⃣ 训练循环（论文版）
# ===============================
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

# ===============================
# 5️⃣ 训练完成后生成 SMILES
# ===============================
model.eval()
with torch.no_grad():
    latent_vectors = genes[:5].to(device)
    generated = model.sample(max_len=100, latent_vectors=latent_vectors)

def decode_smiles(tensor):
    return tokenizer.decode(tensor.tolist())

print("\n🧪 原始论文版模型生成的前 5 条 SMILES：")
for i, gen in enumerate(generated):
    smiles_str = decode_smiles(gen)
    print(f"[{i+1}] {smiles_str}")
