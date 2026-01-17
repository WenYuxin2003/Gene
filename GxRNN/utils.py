import json
import re
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ============================================================================
# Define gene expression dataset
class GeneExpressionDataset(torch.utils.data.Dataset):

    def __init__(self, data):

        self.data = data
        self.data_num = len(data)

    def __len__(self):

        return self.data_num

    def __getitem__(self, idx):
        gene_data = torch.tensor(self.data.iloc[idx]).float()

        return gene_data

# ============================================================================
# Load testing gene expression dataset
def load_test_gene_data(args):

    # Load data, which contains gene values
    data = pd.read_csv(
        args.test_gene_data + args.protein_name + '.csv',
        sep=','
    )
    if args.protein_name == 'AKT1':
        data = data.iloc[:,1:]
    # Get a batch of gene data
    test_data = GeneExpressionDataset(data)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.gene_batch_size,
        shuffle=False
    )

    return test_loader

# ============================================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
def show_other_hyperparameters(args):

    # Hyper-parameters
    params = {}
    print('\n\nOther Hyperparameter Information:')
    print('='*50)
    params['RANDOM_SEEDS'] = args.random_seeds
    params['PROTEIN_NAME'] = args.protein_name
    params['SOURCE_PATH'] = args.source_path
    params['GEN_PATH'] = args.gen_path
    params['candidate_num'] = args.candidate_num

    for param in params:
        string = param + ' ' * (5 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('='*50)

# ============================================================================
# Build vocabulary for SMILES data
'''
class Tokenizer():

    def __init__(self):
        self.start = '^'
        self.end = '$'
        self.pad = ' '
    
    def build_vocab(self):
        chars=[]
        # atoms 
        chars = chars + ['H', 'B', 'C', 'c', 'N', 'n', 'O', 'o', 'P', 'S', 's', 'F', 'I']
        # replace Si for Q, Cl for R, Br for V
        chars = chars + ['Q', 'R', 'V', 'Y', 'Z', 'G', 'T', 'U']
        # hidrogens: H2 to W, H3 to X
        chars = chars + ['[', ']', '+', 'W', 'X']
        # bounding
        chars = chars + ['-', '=', '#', '.', '/', '@', '\\']
        # branches
        chars = chars + ['(', ')']
        # cycles
        chars = chars + ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
        #padding value is 0
        self.tokenlist = [self.pad, self.start, self.end] + list(chars)
        # create the dictionaries      
        self.char_to_int = {c:i for i,c in enumerate(self.tokenlist)}
        self.int_to_char = {i:c for c,i in self.char_to_int.items()}


    @property
    def vocab_size(self):
        return len(self.int_to_char)
    
    def encode(self, smi):
        encoded = []
        smi = smi.replace('Si', 'Q')
        smi = smi.replace('Cl', 'R')
        smi = smi.replace('Br', 'V')
        smi = smi.replace('Pt', 'Y')
        smi = smi.replace('Se', 'Z')
        smi = smi.replace('Li', 'T')
        smi = smi.replace('As', 'U')
        smi = smi.replace('Hg', 'G')
        # hydrogens
        smi = smi.replace('H2', 'W')
        smi = smi.replace('H3', 'X')

        return [self.char_to_int[self.start]] + [self.char_to_int[s] for s in smi] + [self.char_to_int[self.end]]
    
    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords]) 
        # hydrogens
        smi = smi.replace('W', 'H2')
        smi = smi.replace('X', 'H3')
        # replace proxy atoms for double char atoms symbols
        smi = smi.replace('Q', 'Si')
        smi = smi.replace('R', 'Cl')
        smi = smi.replace('V', 'Br')
        smi = smi.replace('Y', 'Pt')
        smi = smi.replace('Z', 'Se')
        smi = smi.replace('T', 'Li')
        smi = smi.replace('U', 'As')
        smi = smi.replace('G', 'Hg')
        
        return smi
'''
class Tokenizer():

    def __init__(self):
        self.start = '^'
        self.end = '$'
        self.pad = ' '

    def build_vocab(self):
        chars = []
        # 常见元素
        chars += ['H', 'B', 'b', 'C', 'c', 'N', 'n', 'O', 'o', 'P', 'p', 'S', 's', 'F', 'I']
        # 其他常见原子或标志
        chars += ['Q', 'R', 'V', 'Y', 'Z', 'G', 'T', 'U']
        # 氢同位素
        chars += ['[', ']', '+', '-', 'W', 'X']
        # 键符号
        chars += ['=', '#', '.', '/', '\\', '@']
        # 分支符号
        chars += ['(', ')']
        # 环编号 (补上 '0')
        chars += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # 其他偶尔出现的符号
        chars += ['%', '*', ':', ';']
        # padding + 起止符
        self.tokenlist = [self.pad, self.start, self.end] + list(sorted(set(chars)))
        # 建立映射
        self.char_to_int = {c: i for i, c in enumerate(self.tokenlist)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}

    @property
    def vocab_size(self):
        return len(self.int_to_char)

    def encode(self, smi):
        # 确保输入是字符串
        smi = str(smi)
        # 替换多字符原子标记
        smi = smi.replace('Si', 'Q').replace('Cl', 'R').replace('Br', 'V')
        smi = smi.replace('Pt', 'Y').replace('Se', 'Z').replace('Li', 'T')
        smi = smi.replace('As', 'U').replace('Hg', 'G')
        smi = smi.replace('H2', 'W').replace('H3', 'X')

        # 忽略未知字符（安全模式）
        tokens = [self.char_to_int[self.start]]
        for s in smi:
            if s in self.char_to_int:
                tokens.append(self.char_to_int[s])
            else:
                # 打印一次警告（可选）
                # print(f"[WARN] Unknown token ignored: {s}")
                continue
        tokens.append(self.char_to_int[self.end])
        return tokens

    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords if o in self.int_to_char])
        smi = smi.replace('W', 'H2').replace('X', 'H3')
        smi = smi.replace('Q', 'Si').replace('R', 'Cl').replace('V', 'Br')
        smi = smi.replace('Y', 'Pt').replace('Z', 'Se').replace('T', 'Li')
        smi = smi.replace('U', 'As').replace('G', 'Hg')
        return smi.strip()

# ============================================================================
def vocabulary(args):

    # Build the vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab()
    #print('\n')
    #print('Vocabulary Information:')
    #print('='*50)
    #print(tokenizer.char_to_int)
    #print('='*50)

    return tokenizer

# ============================================================================
def tanimoto_similarity(smi1, smi2):
    """
    smi1: SMILES string 1
    smi2: SMILES string 2
    returns:
        Tanimoto similarity score
    """
    mols = [Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024) for mol in mols]
    sim_score = FingerprintSimilarity(fps[0],fps[1])

    return sim_score

def mean_similarity(pred_smiles, label_smiles):
    
    all_scores = [tanimoto_similarity(pred, label) for pred, label in zip(pred_smiles, label_smiles)]
    mean_score = np.mean(all_scores)

    return mean_score
