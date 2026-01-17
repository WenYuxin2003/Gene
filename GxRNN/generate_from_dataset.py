# -*- coding: utf-8 -*-
"""
使用已训练好的 GxRNN 模型 (.pkl) 重新生成训练集与验证集分子
并导出原始分子 (SMILES) 以便对比
Author: 敔鑫
"""

import torch
import pandas as pd
from rdkit import Chem
from utils import vocabulary, get_device
from xin4_GxRNNstar import GxRNN
from train_gxrnn import load_smiles_data

# ==============================
# 参数配置（与训练时保持完全一致）
# ==============================
class Args:
    gene_expression_file = "datasets/LINCS/"   # 数据路径
    cell_name = "mcf7"                         # 数据集名
    gene_num = 978
    gene_batch_size = 64
    train_rate = 0.9
    emb_size = 128
    hidden_size = 1024                         # 与训练时相同
    num_layers = 3
    smiles_dropout = 0.3
    max_len = 100
    saved_gxrnn = "results/saved_gxrnn.pkl_450.pkl"  # 模型路径
    variant=False

args = Args()

# ==============================
# 加载词典与数据集
# ==============================
print("📘 Loading dataset & tokenizer ...")
tokenizer = vocabulary(args)
train_loader, valid_loader = load_smiles_data(tokenizer, args)

# ==============================
# 加载模型
# ==============================
print(f"📦 Loading trained model from {args.saved_gxrnn} ...")
model = GxRNN(
    tokenizer,
    emb_size=args.emb_size,
    hidden_size=args.hidden_size,
    gene_latent_size=args.gene_num,
    num_layers=args.num_layers,
    dropout=args.smiles_dropout
).to(get_device())

model.load_model(args.saved_gxrnn)
model.eval()

# ==============================
# 辅助函数：生成 + 筛选 + 保存原始分子
# ==============================
def generate_and_save(loader, tag):
    print(f"\n🚀 Generating molecules for {tag} set ...")
    generated, original = [], []

    for _, (smiles, genes) in enumerate(loader):
        smiles, genes = smiles.to(get_device()), genes.to(get_device())
        dec_sampled_char = model.sample(args.max_len, genes)

        # 解码生成分子
        gen_smiles = [
            "".join(tokenizer.decode(dec_sampled_char[i].squeeze().detach().cpu().numpy())).strip("^$ ")
            for i in range(dec_sampled_char.size(0))
        ]

        # 解码真实分子
        true_smiles = [
            "".join(tokenizer.decode(smiles[i].squeeze().detach().cpu().numpy())).strip("^$ ")
            for i in range(smiles.size(0))
        ]

        # 过滤生成分子（只保留合法的）
        for g_smi, t_smi in zip(gen_smiles, true_smiles):
            mol = Chem.MolFromSmiles(g_smi)
            if mol is not None and mol.GetNumAtoms() > 1:
                generated.append(Chem.MolToSmiles(mol))
                original.append(t_smi)

    # 保存生成结果
    gen_df = pd.DataFrame({"SMILES": generated})
    gen_path = f"results/generated_{tag}_from_pkl.csv"
    gen_df.to_csv(gen_path, index=False)

    # 保存原始真实分子
    orig_df = pd.DataFrame({"SMILES": original})
    orig_path = f"results/original_{tag}_smiles.csv"
    orig_df.to_csv(orig_path, index=False)

    print(f"✅ {tag} 集生成完成: {len(generated)} 个合法分子")
    print(f"📁 生成分子保存到: {gen_path}")
    print(f"📁 原始分子保存到: {orig_path}")
    print("示例前5个生成分子:\n", gen_df.head().to_string(index=False))


# ==============================
# 执行生成
# ==============================
generate_and_save(train_loader, "train")
generate_and_save(valid_loader, "valid")

print("\n🎉 训练集与验证集生成与原始分子导出完成。")
