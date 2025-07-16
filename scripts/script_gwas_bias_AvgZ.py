# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# Scripts_BiasCal.py
# ========================================================================================================

import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
sys.path.insert(1, '/home/jw3514/Work/UNIMED/src')
from CellType_PSY import *
from UNIMED import *

import multiprocessing
from multiprocessing import Pool
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import logging
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_cluster_AvgBias(Disorder_Series, SpecMat, TopGenes, Anno, mode):
    """Calculate correlations for all clusters efficiently."""
    GeneZSTAT = Disorder_Series.sort_values(ascending=False).head(TopGenes)
    GeneZSTAT = GeneZSTAT.to_dict()
    #print(GeneZSTAT)
    if mode == "HumanCT":
        results_df = HumanCT_AvgZ_Weighted(SpecMat, GeneZSTAT)
        if Anno is None:
            raise ValueError("Annotation data is missing for HumanCT mode")
        results_df = AnnotateCTDat(results_df, Anno)
        results_df = results_df.sort_values(by='EFFECT', ascending=False)

    elif mode == "MouseSTR":
        if Anno is None:
            raise ValueError("Annotation data is missing for MouseSTR mode")
        results_df["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in results_df['Structure'].values]
        results_df = results_df.reset_index(drop=True)
        results_df["Rank"] = np.arange(1, len(results_df) + 1)
    elif mode == "MouseCT":
        results_df = MouseCT_AvgZ_Weighted(SpecMat, GeneZSTAT)
        print(results_df.head())
        if Anno is None:
            raise ValueError("Annotation data is missing for MouseCT mode")
        results_df['class_id_label'] = Anno.loc[results_df['ct_idx'].values, "class_id_label"].values
        results_df['subclass_id_label'] = Anno.loc[results_df['ct_idx'].values, "subclass_id_label"].values
    results_df = results_df.set_index("ct_idx")
    return results_df

def GetOption():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InpFil', type=str, required=True, help='file contains list of gene weight files')
    parser.add_argument('-m', '--mode', type=str, choices=["HumanCT", "MouseCT", "MouseSTR"], required=True, help='model to use')
    parser.add_argument('--biasMat', type=str, help='bias matrix file')
    parser.add_argument('--outDir', type=str)
    parser.add_argument('--processes', type=int, default=20, help='Number of processes to use')
    parser.add_argument('--exclude', type=str, help='file containing list of genes to exclude from analysis')
    return parser.parse_args()

def process_file(filename, SpecMat, TopGenes, outDIR, mode, Anno, exclude_genes):
    name = filename.split("/")[-1].split(".")[0]
    logging.info(f"Processing file: {filename}")
    
    GWAS_DF = pd.read_csv(filename, sep="\t", index_col="GENE")
        
    print("shape of Disorder GWAS: ", GWAS_DF.shape)
    if exclude_genes:
        GWAS_DF = GWAS_DF[~GWAS_DF.index.isin(exclude_genes)]
        print("shape of Disorder GWAS after exclusion: ", GWAS_DF.shape)
        
    if "ZSTAT" not in GWAS_DF.columns:
        logging.error(f"ZSTAT column missing in file {filename}")
        return
        
    results = calculate_cluster_AvgBias(GWAS_DF["ZSTAT"], SpecMat, TopGenes, Anno, mode)
    
    outname = f"{outDIR}/{mode}.Bias.{name}.Z2.csv"
    results.to_csv(outname, index_label="ct_idx")
        
    logging.info(f"Completed processing file: {filename}")

def process_batch(files, SpecMat,TopGenes, outDIR, mode, Anno, exclude_genes, num_processes=20):
    logging.info(f"Starting batch processing with {num_processes} processes")
    
    if not files:
        logging.warning("No files to process")
        return
        
    # Validate output directory exists
    if not os.path.exists(outDIR):
        os.makedirs(outDIR)
            
    # Create a partial function with fixed arguments
    process_func = partial(process_file, 
                         SpecMat=SpecMat,
                         TopGenes=TopGenes,
                         outDIR=outDIR,
                         mode=mode,
                         Anno=Anno,
                         exclude_genes=exclude_genes)
    
    # Use context manager for proper cleanup
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_func, files)
        
    logging.info("Batch processing completed successfully")

def main():
    args = GetOption()
    
    with open(args.InpFil, 'rt') as f:
        input_files = [x.strip() for x in f.readlines()]
    
    outDIR = args.outDir if args.outDir else "."
    
    # Load genes to exclude if specified
    exclude_genes = set()
    if args.exclude:
        with open(args.exclude, 'rt') as f:
            exclude_genes = set(int(x.strip()) for x in f.readlines())
        logging.info(f"Excluding {len(exclude_genes)} genes from analysis")
    
    # Load appropriate data based on mode
    if args.mode == "HumanCT":
        Annotat = Anno
        Z2Mat = pd.read_csv(args.biasMat if args.biasMat else 
                           "/home/jw3514/Work/CellType_Psy/dat/HumanCTExpressionMats/Human.Cluster.Log2Mean.Z1clip5.Z2.clip3.Dec30.csv",
                           index_col=0)
        Z2Mat.columns = Z2Mat.columns.astype(int)
    elif args.mode == "MouseSTR":
        Annotat = STR2Region()
        Z2Mat = pd.read_csv(args.biasMat if args.biasMat else 
                            "/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv", index_col=0)
    elif args.mode == "MouseCT":
        Annotat = pd.read_csv("../dat/MouseCT_Cluster_Anno.csv", index_col="cluster_id_label")
        Z2Mat = pd.read_csv(args.biasMat if args.biasMat else 
                           "/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/Cluster_Z2Mat_ISHMatch.z1clip3.csv", index_col=0)
    
    # Remove excluded genes from expression matrix
    print(list(exclude_genes)[:10])
    print(Z2Mat.index[:10])
    logging.info(f"Expression matrix shape: {Z2Mat.shape}")
    if len(exclude_genes) > 0:
        # Get genes to keep by finding genes not in exclude list
        genes_to_keep = [g for g in Z2Mat.index if g not in exclude_genes]
        # Filter matrix to only keep allowed genes
        Z2Mat = Z2Mat.loc[genes_to_keep]
        logging.info(f"Expression matrix shape after excluding genes: {Z2Mat.shape}")
    
    if Z2Mat.empty:
        logging.error("Expression matrix is empty after filtering")
        return
        
    # Get top genes using vectorized operations
    TopGenes = 160000
    
    # Process files
    process_batch(input_files, Z2Mat, TopGenes, outDIR, args.mode, Annotat, exclude_genes, args.processes)

if __name__ == '__main__':
    main()
