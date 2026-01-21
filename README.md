# Transcriptionally Controlled De Novo Molecular Generation through Hierarchical Gene Encoding and Adaptive Feature Modulation
This repository contains the official implementation of the paper:

**Transcriptionally Controlled De Novo Molecular Generation through Hierarchical Gene Encoding and Adaptive Feature Modulation**
Wen Yuxin,Tian Yi, Nguyen Quoc Khanh Le*, Matthew Chin Heng Chua*

📄 *Manuscript  submitted*

---

## 🔬 Overview

This work proposes a **gene-expression–conditioned de novo molecular generation framework**, which integrates:

- **Gene Regulatory Representation Module (GRRM)**
  Hierarchical encoding of high-dimensional transcriptomic profiles
- **Adaptive Feature-wise Linear Modulation (FiLM)**
  Dynamic gene-conditioned modulation of SMILES embeddings
- **LSTM-based autoregressive decoder**
  Generation of chemically valid and biologically relevant molecules

The model is evaluated on **LINCS L1000 transcriptomic data** and demonstrates state-of-the-art performance on **10 therapeutic protein targets**, outperforming existing transcriptome-conditioned baselines.

---

## 🧠 Model Architecture

Gene Expression (978-dim)
 │
 ▼
 Gene Regulatory Representation Module (GRRM)
 │
 ▼
 Adaptive FiLM Modulation
 │
 ▼
 FiLM-modulated SMILES Embeddings
 │
 ▼
 LSTM Decoder
 │
 ▼
 Generated SMILES

The proposed model is implemented in `xin4_GxRNNstar.py`, which contains the complete definition of the **GRRM-FiLM-LSTM** framework described in the paper.

The module names used in the manuscript do not always exactly match the class or function names in the current implementation. These differences are purely nominal and do not affect the implemented functionality. The codebase will be further refactored in future updates to ensure full consistency with the paper’s module nomenclature.

The PBS scripts are retained as a complete and reproducible record of the model execution commands, providing evidence of the authenticity of the implementation.(PBS commands supported by the National Supercomputing Centre, Singapore)