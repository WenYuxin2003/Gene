import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import torch
import warnings
import argparse
import numpy as np
import torch.nn as nn
warnings.filterwarnings("ignore")

import os
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from utils import *
#修改
#from GxRNN import GxRNN
from xin4_GxRNNstar import GxRNN
from train_gxrnn import load_smiles_data, train_gxrnn

# =================================================================================
# Default settings
parser = argparse.ArgumentParser()

parser.add_argument('--cell_name', type=str, default='mcf7', help='Cell name of LINCS files, e.g., mcf7')
parser.add_argument('--protein_name', type=str, default='AKT1', help='10 proteins are AKT1, AKT2, AURKB, CTSK, EGFR, HDAC1, MTOR, PIK3CA, SMAD3, and TP53')
parser.add_argument('--generation', action='store_true', help='Generate new molecules') # Add --generation to generate new molecules using trained model
parser.add_argument('--gene_num', type=int, default=978, help='Number of gene values') # MCF7: 978
parser.add_argument('--gene_batch_size', type=int, default=64, help='Batch size for training GeneVAE') # 64
parser.add_argument('--gene_expression_file', type=str, default='datasets/LINCS/', help='Path of the training gene expression profile dataset for the VAE')
parser.add_argument('--test_gene_data', type=str, default='datasets/test_protein/', help='Path of the gene expression profile dataset for test proteins or test disease')

# ===========================
# SmilesDecoder
parser.add_argument('--train_gxrnn', action='store_true', help='Train GxRNN') # Add --train_smiles_decoder to train GxRNN

parser.add_argument('--smiles_epochs', type=int, default=300, help='SmilesDecoder training epochs')
parser.add_argument('--emb_size', type=int, default=128, help='Embedding size of SmilesDecoder')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size of SmilesDecoder')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers for training SmilesDecoder')
parser.add_argument('--smiles_lr', type=float, default=5e-4, help='Learning rate of SmilesDecoder')
parser.add_argument('--smiles_dropout', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--train_rate', type=float, default=0.9, help='Split training and validating subsets by training rate')
parser.add_argument('--max_len', type=int, default=100, help='Maximum length of SMILES strings')
parser.add_argument('--saved_gxrnn', type=str, default='results/saved_gxrnn.pkl', help='Save the trained model')
parser.add_argument('--valid_smiles_file', type=str, default='results/predicted_valid_smiles.csv', help='Save the valid SMILES into file')
parser.add_argument('--gxrnn_train_results', type=str, default='results/gxrnn_train_results.csv', help='Path to save the results of GxRNN training')
parser.add_argument('--variant', action='store_true', help='Apply variant smiles') # Add --variant to apply variant smiles

# ===========================
# Molecule selection with similar ligands
parser.add_argument('--calculate_tanimoto', action='store_true', help='Calculate tanimoto similarity for the source ligand and generated SMILES') # Add --calculate_tanimoto to calculate Tanimoto similarity
parser.add_argument('--random_seeds', type=int, default=1000, help='Generate n samples for each protein') 
parser.add_argument('--candidate_num', type=int, default=50, help='Number of candidate SMILES strings')

#parser.add_argument('--gene_type', type=str, default='gene_symbol', help='Gene types')
parser.add_argument('--source_path', type=str, default='datasets/ligands/source_', help='Load the source SMILES strings of known ligands')
parser.add_argument('--gen_path', type=str, default='results/generation/', help='Save the generated SMILES strings')

parser.add_argument('--count_known_ligands', action='store_true')

args = parser.parse_args()


# =================================================================================
def main(args):

    # Apply the seed to reproduct the results
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Print vocabulary information
    tokenizer = vocabulary(args)
    # Get train and valid dataloader
    train_dataloader, valid_dataloader = load_smiles_data(tokenizer, args)


    # ========================================================= #
    #                      1. Train GxRNN                       #
    # ========================================================= #
    if args.train_gxrnn:
        # Train GxRNN
        trained_gxrnn = train_gxrnn(
            train_dataloader,
            valid_dataloader,
            tokenizer, 
            args
        ).to(get_device())

    # ========================================================= #
    #                       2. Generation                       #
    # ========================================================= #
    if args.generation:
        # Print other hyperparameter information
        show_other_hyperparameters(args)

        # Load the trained GxRNN  
        trained_gxrnn = GxRNN(
            tokenizer,
            emb_size=args.emb_size,
            hidden_size=args.hidden_size,
            gene_latent_size=args.gene_num,
            num_layers=args.num_layers,
            dropout=args.smiles_dropout
        ).to(get_device())
        trained_gxrnn.load_model(args.saved_gxrnn)

        # Test mode
        trained_gxrnn.eval()

        # Load testing data
        test_gene_loader = load_test_gene_data(args)

        res_smiles = []
        # Write generated SMILES into file with random seeds
        for sd in range(args.random_seeds):
            np.random.seed(sd)
            torch.manual_seed(sd)
            torch.cuda.manual_seed(sd)

            for _, genes in enumerate(test_gene_loader):
                genes = genes.to(get_device())
                dec_sampled_char = trained_gxrnn.sample(args.max_len, genes)
                output_smiles = ["".join(tokenizer.decode(\
                    dec_sampled_char[i].squeeze().detach().cpu().numpy()
                    )).strip("^$ ") for i in range(dec_sampled_char.size(0))]
                res_smiles.append(output_smiles)

        test_data = pd.DataFrame(columns=['SMILES'], data=res_smiles)
        if not os.path.exists(args.gen_path):
            os.makedirs(args.gen_path)
        test_data.to_csv(args.gen_path + 'res-{}.csv'.format(args.protein_name), index=False)

    # ========================================================= #
    #                       3. Tanimoto                         #
    # ========================================================= #
    if args.calculate_tanimoto:     
        # Read training data
        train_data = pd.read_csv(
            args.gene_expression_file + args.cell_name + '.csv', 
            sep=',', 
            names=['inchikey','smiles'] + ['gene'+str(i) for i in range(1,args.gene_num+1)]
        )
        train_data = train_data['smiles']
        # Canonical SMILES
        train_data = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in train_data]

        # Read source ligands
        source_path = args.source_path + args.protein_name + '.csv'
        source_data = pd.read_csv(source_path, names=['smiles'])
        # Remove the SMILES string from the source ligands that are the same as the training dataset
        canonical_source_data = []
        for smi in source_data['smiles']:
            try:
                cano_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            except Exception:
                cano_smi = smi
            if cano_smi not in train_data:
                canonical_source_data.append(cano_smi)

        # Read generated SMILES string according to a protein
        if not os.path.exists(args.gen_path):
            print('generated path {} does not exist!'.format(args.gen_path))
        else:
            gen_path = args.gen_path + 'res-' + args.protein_name + '.csv'
            gen_data = pd.read_csv(gen_path)
            
            # Tanimoto similarity
            tanimoto = []

            for i in range(args.candidate_num):
                m1= Chem.MolFromSmiles(gen_data['SMILES'][i])
                # Remove generated molecules identical training data
                if (m1 != None) and (Chem.MolToSmiles(m1) in train_data):
                    continue
                if m1:
                    try:
                        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
                    except Exception:
                        break
                    else:
                        for j in range(len(canonical_source_data)):
                            try:
                                m2 = Chem.MolFromSmiles(canonical_source_data[j])
                                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
                            except Exception:
                                tanimoto.append([0, canonical_source_data[j], gen_data['SMILES'][i]])
                            else:
                                tanimoto.append([DataStructs.BulkTanimotoSimilarity(fp1, [fp2])[0], canonical_source_data[j], gen_data['SMILES'][i]])

            res = pd.DataFrame(tanimoto)   
            max_res = res.iloc[res[0].idxmax()]
            print('protein name:', args.protein_name)
            print('Source ligand:', max_res[1])
            print('Best generation:', max_res[2])
            print('Tanimoto similarity: {}'.format(max_res[0]))

    if args.count_known_ligands:
        # Read training data
        train_data = pd.read_csv(
            args.gene_expression_file + args.cell_name + '.csv',
            sep=',',
            names=['inchikey','smiles'] + ['gene'+str(i) for i in range(1,args.gene_num+1)]
        )
        train_data = train_data['smiles']
        # Canonical SMILES
        train_data = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in train_data]

        # Read source ligands
        source_path = args.source_path + args.protein_name + '.csv'
        source_data = pd.read_csv(source_path, names=['smiles'])
        # Remove the SMILES string from the source ligands that are the same as the training dataset
        count = 0
        for smi in source_data['smiles']:
            mol = Chem.MolFromSmiles(smi)
            if (mol != None) and (Chem.MolToSmiles(mol) not in train_data):
                count += 1
        print('protein name:', args.protein_name)
        print('The number of known ligands:', count)



if __name__ == '__main__':
    main(args)
