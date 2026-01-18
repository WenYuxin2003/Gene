# -*- coding: utf-8 -*-
"""
小提琴图对比生成分子与原始分子属性分布
Author: Wen Yuxin
"""

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def calc_properties(smiles_list, label):
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            logp = Descriptors.MolLogP(mol)
            sas = rdMolDescriptors.CalcNumHBA(mol) + rdMolDescriptors.CalcNumHBD(mol)  # 可换为实际 SAS 计算
            tpsa = Descriptors.TPSA(mol)
            qed = Descriptors.qed(mol)
            mw = Descriptors.MolWt(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            records.append([label, logp, sas, tpsa, qed, mw, rotb])
    return pd.DataFrame(records, columns=["type", "logP", "SAS", "TPSA", "QED", "MolWt", "RotBonds"])


def load_and_compute(gen_path, orig_path, label_gen, label_orig):
    gen = pd.read_csv(gen_path)["SMILES"].dropna().tolist()
    orig = pd.read_csv(orig_path)["SMILES"].dropna().tolist()
    df_gen = calc_properties(gen, label_gen)
    df_orig = calc_properties(orig, label_orig)
    return pd.concat([df_gen, df_orig], ignore_index=True)


def plot_violin(df, name):
    props = ["logP", "SAS", "TPSA", "QED", "MolWt", "RotBonds"]
    labels = df["type"].unique().tolist()
    colors = ["#66c2a5", "#fc8d62"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, prop in enumerate(props):
        ax = axes[i]
        data = [df[df["type"] == lbl][prop] for lbl in labels]
        vp = ax.violinplot(data, showmedians=True, widths=0.8)
        for j, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[j])
            body.set_edgecolor("black")
            body.set_alpha(0.7)
        ax.set_title(prop, fontsize=12)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(f"{name}_violin_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}_violin_plot.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {name}_violin_plot.png")


if __name__ == "__main__":
    # 1️⃣ Train 对比
    df_train = load_and_compute(
        "generated_train_from_pkl.csv",
        "original_train_smiles.csv",
        "Generated_Train",
        "Original_Train"
    )
    plot_violin(df_train, "train")

    # 2️⃣ Valid 对比
    df_valid = load_and_compute(
        "generated_valid_from_pkl.csv",
        "original_valid_smiles.csv",
        "Generated_Valid",
        "Original_Valid"
    )
    plot_violin(df_valid, "valid")
