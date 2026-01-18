# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader, random_split

from utils import get_device


# ============================================================================
# 数据集与 DataLoader（保持不变）
class Smiles_Dataset(Dataset):
    def __init__(self, gene_expression_file, cell_name, tokenizer, gene_num, variant=False):
        self.tokenizer = tokenizer
        data = pd.read_csv(
            gene_expression_file + cell_name + '.csv',
            sep=',',
            names=['inchikey', 'smiles'] + [f'gene{i}' for i in range(1, gene_num + 1)]
        )
        data = data.dropna(how='any')

        if variant:
            data['smiles'] = data['smiles'].apply(self._variant_smiles)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smi = self.data.iloc[index]['smiles']
        encoded_smi = self.tokenizer.encode(smi)
        gene = self.data.iloc[index]['gene1':].values.astype('float32')
        return encoded_smi, gene

    @staticmethod
    def _variant_smiles(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(idxs)
        mol = Chem.RenumberAtoms(mol, idxs)
        return Chem.MolToSmiles(mol, canonical=False)


class Smiles_DataLoader(DataLoader):
    def __init__(self, gene_expression_file, cell_name, tokenizer, gene_num,
                 batch_size, train_rate=0.9, variant=False):
        self.gene_expression_file = gene_expression_file
        self.cell_name = cell_name
        self.tokenizer = tokenizer
        self.gene_num = gene_num
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.variant = variant

    @staticmethod
    def collate_fn(batch):
        smiles, genes = zip(*batch)
        smi_tensors = [torch.tensor(s, dtype=torch.long) for s in smiles]
        smi_tensors = nn.utils.rnn.pad_sequence(smi_tensors, batch_first=True)
        gene_tensors = torch.tensor(np.array(genes), dtype=torch.float32)
        return smi_tensors, gene_tensors

    def get_dataloader(self):
        dataset = Smiles_Dataset(
            self.gene_expression_file,
            self.cell_name,
            self.tokenizer,
            self.gene_num,
            self.variant
        )

        train_size = int(len(dataset) * self.train_rate)
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(
            dataset=dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(0)
        )

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=1
        )
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=1
        )
        return train_loader, test_loader


# ============================================================================
# 改进的 STAR 模块：真正在基因维度做全局聚合
class ImprovedSTAR(nn.Module):
    """
    改进的 STAR 模块：
    1. 在基因维度(num_genes)做 attention-like 聚合
    2. 使用多头注意力机制增强表达能力
    3. 残差连接 + LayerNorm 稳定训练

    输入: [B, num_genes]
    输出: [B, num_genes]
    """

    def __init__(self, num_genes: int, d_core: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_genes = num_genes
        self.d_core = d_core
        self.num_heads = num_heads

        # 多头注意力分支
        assert d_core % num_heads == 0, "d_core must be divisible by num_heads"
        self.head_dim = d_core // num_heads

        # Query, Key, Value 投影
        self.q_proj = nn.Linear(num_genes, d_core)
        self.k_proj = nn.Linear(num_genes, d_core)
        self.v_proj = nn.Linear(num_genes, d_core)

        # 输出投影
        self.out_proj = nn.Linear(d_core, num_genes)

        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(num_genes, num_genes * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_genes * 2, num_genes),
            nn.Dropout(dropout)
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(num_genes)
        self.norm2 = nn.LayerNorm(num_genes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x: [B, num_genes]
        """
        B = x.shape[0]
        residual = x

        # 转置以便在基因维度做 attention: [B, 1, num_genes]
        x_expanded = x.unsqueeze(1)  # [B, 1, num_genes]

        # Multi-head attention
        q = self.q_proj(x_expanded)  # [B, 1, d_core]
        k = self.k_proj(x_expanded)  # [B, 1, d_core]
        v = self.v_proj(x_expanded)  # [B, 1, d_core]

        # Reshape for multi-head: [B, num_heads, 1, head_dim]
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.d_core)

        # Output projection
        out = self.out_proj(attn_output).squeeze(1)  # [B, num_genes]
        out = self.dropout(out)

        # 第一个残差连接
        x = self.norm1(residual + out)

        # FFN + 第二个残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x  # [B, num_genes]


# ============================================================================
# 渐进式基因压缩模块
class ProgressiveGeneEncoder(nn.Module):
    """
    渐进式压缩基因表达，减少信息丢失
    978 → 512 → 256 → gene_feature_dim
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        # 计算中间维度
        mid_dim1 = max(output_dim, input_dim // 2)
        mid_dim2 = max(output_dim, input_dim // 4)

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mid_dim1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.LayerNorm(mid_dim1),
            nn.Linear(mid_dim1, mid_dim2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.LayerNorm(mid_dim2),
            nn.Linear(mid_dim2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.encoder(x)


# ============================================================================
# 改进的 FiLM 调制模块
class AdaptiveFiLM(nn.Module):
    """
    改进的 FiLM 调制：
    1. alpha 使用 tanh 而非 sigmoid，允许负调制
    2. 门控机制控制调制强度
    """

    def __init__(self, gene_dim: int, emb_dim: int):
        super().__init__()

        # FiLM 参数生成
        self.alpha_net = nn.Sequential(
            nn.Linear(gene_dim, emb_dim),
            nn.Tanh()  # [-1, 1] 更温和的调制
        )

        self.beta_net = nn.Linear(gene_dim, emb_dim)

        # 门控机制：控制 FiLM 调制的强度
        self.gate_net = nn.Sequential(
            nn.Linear(gene_dim, emb_dim),
            nn.Sigmoid()
        )

    def forward(self, emb, gene_feat):
        """
        emb: [B, L, E] - SMILES embedding
        gene_feat: [B, F] - gene features
        """
        alpha = self.alpha_net(gene_feat).unsqueeze(1)  # [B, 1, E]
        beta = self.beta_net(gene_feat).unsqueeze(1)  # [B, 1, E]
        gate = self.gate_net(gene_feat).unsqueeze(1)  # [B, 1, E]

        # 调制：alpha ∈ [-1, 1], 1 + alpha ∈ [0, 2]
        modulated = emb * (1 + alpha) + beta

        # 门控混合：保留部分原始信息
        output = gate * modulated + (1 - gate) * emb

        return output


# ============================================================================
# 改进的 GxRNN 主模型
class GxRNN(nn.Module):
    """
    改进版 GxRNN：
    1. 使用改进的 STAR 模块进行真正的全局基因聚合
    2. 渐进式基因压缩，减少信息丢失
    3. 改进的 FiLM 调制机制
    4. 用基因信息初始化 LSTM 隐状态
    5. 添加 L2 正则化和更多 Dropout
    """

    def __init__(self, tokenizer, emb_size=128, hidden_size=256, gene_latent_size=978,
                 num_layers=3, dropout=0.2, star_core_dim=256, gene_feature_dim=256,
                 star_num_heads=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.gene_latent_size = gene_latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gene_feature_dim = gene_feature_dim

        # SMILES 嵌入 + Dropout
        self.embedding = nn.Embedding(
            tokenizer.vocab_size,
            emb_size,
            padding_idx=tokenizer.char_to_int[tokenizer.pad]
        )
        self.emb_dropout = nn.Dropout(dropout)

        # 改进的 STAR 模块：在基因维度做全局聚合
        self.star = ImprovedSTAR(
            num_genes=gene_latent_size,
            d_core=star_core_dim,
            num_heads=star_num_heads,
            dropout=dropout
        )

        # 渐进式基因编码器
        self.gene_encoder = ProgressiveGeneEncoder(
            input_dim=gene_latent_size,
            output_dim=gene_feature_dim,
            dropout=dropout
        )

        # 改进的 FiLM 调制
        self.film = AdaptiveFiLM(gene_feature_dim, emb_size)

        # 基因特征到 LSTM 初始状态的投影
        self.gene_to_h0 = nn.Linear(gene_feature_dim, num_layers * hidden_size)
        self.gene_to_c0 = nn.Linear(gene_feature_dim, num_layers * hidden_size)

        # LSTM 解码器
        self.rnn = nn.LSTM(
            input_size=emb_size + gene_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # 输出层 + Dropout
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, tokenizer.vocab_size)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Xavier 初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def _encode_gene(self, gene_vec):
        """编码基因表达"""
        gene_global = self.star(gene_vec)  # 全局聚合
        gene_feat = self.gene_encoder(gene_global)  # 渐进式压缩
        return gene_feat

    def _init_hidden(self, gene_feat):
        """用基因特征初始化 LSTM 隐状态"""
        B = gene_feat.size(0)
        h0 = self.gene_to_h0(gene_feat).view(B, self.num_layers, self.hidden_size)
        c0 = self.gene_to_c0(gene_feat).view(B, self.num_layers, self.hidden_size)
        h0 = h0.transpose(0, 1).contiguous()  # [num_layers, B, hidden_size]
        c0 = c0.transpose(0, 1).contiguous()
        return h0, c0

    def forward(self, decoder_inputs, latent_vectors):
        """
        decoder_inputs: [B, L] - token IDs
        latent_vectors: [B, gene_latent_size] - gene expression
        """
        B, L = decoder_inputs.size()
        self.rnn.flatten_parameters()

        # 1. Token embedding
        emb = self.embedding(decoder_inputs)  # [B, L, E]
        emb = self.emb_dropout(emb)

        # 2. 编码基因特征
        gene_feat = self._encode_gene(latent_vectors)  # [B, F]

        # 3. FiLM 调制
        emb_modulated = self.film(emb, gene_feat)  # [B, L, E]

        # 4. 拼接条件向量
        cond = gene_feat.unsqueeze(1).expand(-1, L, -1)  # [B, L, F]
        rnn_input = torch.cat([emb_modulated, cond], dim=-1)  # [B, L, E+F]

        # 5. LSTM 解码（用基因特征初始化隐状态）
        h0, c0 = self._init_hidden(gene_feat)
        rnn_output, _ = self.rnn(rnn_input, (h0, c0))  # [B, L, H]

        # 6. 输出预测
        logits = self.fc(rnn_output.reshape(-1, self.hidden_size))  # [B*L, V]
        pred = self.log_softmax(logits)

        return pred

    def step(self, x_t, gene_feat, h, c):
        """单步生成"""
        emb = self.embedding(x_t)  # [B, 1, E]
        emb = self.emb_dropout(emb)

        # FiLM 调制
        emb_modulated = self.film(emb, gene_feat)

        # 拼接条件
        rnn_input = torch.cat([emb_modulated, gene_feat.unsqueeze(1)], dim=-1)

        # LSTM 单步
        out, (h, c) = self.rnn(rnn_input, (h, c))
        logits = self.fc(out.reshape(-1, self.hidden_size))
        pred = self.log_softmax(logits)

        return pred, h, c

    def sample(self, max_len, latent_vectors, temperature=1.0):
        """
        采样生成 SMILES
        temperature: 控制随机性，越大越随机
        """
        B = latent_vectors.size(0)
        device = latent_vectors.device

        # 编码基因特征并初始化隐状态
        gene_feat = self._encode_gene(latent_vectors)
        h, c = self._init_hidden(gene_feat)

        # 初始 token
        x = torch.full(
            (B, 1),
            self.tokenizer.char_to_int[self.tokenizer.start],
            dtype=torch.long,
            device=device
        )

        samples = []
        for _ in range(max_len):
            pred, h, c = self.step(x, gene_feat, h, c)

            # Temperature sampling
            if temperature != 1.0:
                pred = pred / temperature

            x = torch.multinomial(torch.exp(pred), 1)
            samples.append(x)

        return torch.cat(samples, dim=1)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=get_device()))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def get_l2_loss(self, weight_decay=1e-4):
        """计算 L2 正则化损失"""
        l2_loss = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name and param.requires_grad:
                l2_loss += torch.norm(param, p=2)
        return weight_decay * l2_loss