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

def calculate_cluster_topGene_Enrichment(Disorder_Series, top_genes_by_ct, Anno, mode):
    """Calculate correlations for all clusters efficiently."""
    results = []
    for ct_idx, genes in top_genes_by_ct.items():
        # Check if genes list is empty
        if not genes:
            logging.warning(f"Empty gene list for cluster {ct_idx}")
            continue
            
        # Check if all genes exist in Disorder_Series
        valid_genes = [g for g in genes if g in Disorder_Series.index]
        #if len(valid_genes) < len(genes):
        #    logging.warning(f"Some genes not found in GWAS data for cluster {ct_idx}")
            
        result = gene_set_enrichment_test(Disorder_Series, valid_genes)
        results.append({
            'ct_idx' if mode != "MouseSTR" else 'Structure': ct_idx,
            'beta': result['effect_size'],
            'pvalue': result['pvalue']
        })

    if not results:
        raise ValueError("No valid results generated from enrichment analysis")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='beta', ascending=False)

    if mode == "HumanCT":
        if Anno is None:
            raise ValueError("Annotation data is missing for HumanCT mode")
        results_df = AnnotateCTDat(results_df, Anno)

    elif mode == "MouseSTR":
        if Anno is None:
            raise ValueError("Annotation data is missing for MouseSTR mode")
        results_df["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in results_df['Structure'].values]
        results_df = results_df.reset_index(drop=True)
        results_df["Rank"] = np.arange(1, len(results_df) + 1)
        
    elif mode == "MouseCT":
        if Anno is None:
            raise ValueError("Annotation data is missing for MouseCT mode")
        results_df['class_id_label'] = Anno.loc[results_df['ct_idx'].values, "class_id_label"].values
        results_df['subclass_id_label'] = Anno.loc[results_df['ct_idx'].values, "subclass_id_label"].values

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

def process_file(filename, top_genes_by_ct, outDIR, mode, Anno, exclude_genes):
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
        
    results = calculate_cluster_topGene_Enrichment(GWAS_DF["ZSTAT"], top_genes_by_ct, Anno, mode)
    
    outname = f"{outDIR}/{mode}.{name}.csv"
    results.to_csv(outname, index=False)
        
    logging.info(f"Completed processing file: {filename}")

def process_batch(files, top_genes_by_ct, outDIR, mode, Anno, exclude_genes, num_processes=20):
    logging.info(f"Starting batch processing with {num_processes} processes")
    
    if not files:
        logging.warning("No files to process")
        return
        
    # Validate output directory exists
    if not os.path.exists(outDIR):
        os.makedirs(outDIR)
            
    # Create a partial function with fixed arguments
    process_func = partial(process_file, 
                         top_genes_by_ct=top_genes_by_ct,
                         outDIR=outDIR,
                         mode=mode,
                         Anno=Anno,
                         exclude_genes=exclude_genes)
    
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
        BiasMat = pd.read_parquet(args.biasMat)
        BiasMat.columns = BiasMat.columns.astype(int)
    elif args.mode == "MouseSTR":
        Annotat = STR2Region()
        BiasMat = pd.read_parquet(args.biasMat)
    elif args.mode == "MouseCT":
        Annotat = pd.read_csv("/home/jw3514/Work/UNIMED/dat/MouseCT_Cluster_Anno.csv", index_col="cluster_id_label")
        BiasMat = pd.read_parquet(args.biasMat)
    
    # Remove excluded genes from expression matrix
    print(list(exclude_genes)[:10])
    print(BiasMat.index[:10])
    logging.info(f"Expression matrix shape: {BiasMat.shape}")
    if len(exclude_genes) > 0:
        # Get genes to keep by finding genes not in exclude list
        genes_to_keep = [g for g in BiasMat.index if g not in exclude_genes]
        # Filter matrix to only keep allowed genes
        BiasMat = BiasMat.loc[genes_to_keep]
        logging.info(f"Expression matrix shape after excluding genes: {BiasMat.shape}")
    
    if BiasMat.empty:
        logging.error("Expression matrix is empty after filtering")
        return
        
    # Get top genes using vectorized operations
    topFrac = 0.1
    topN = int(topFrac * BiasMat.shape[0])
    top_genes_by_ct = {col: BiasMat[col].nlargest(topN).index.tolist() for col in BiasMat.columns}
    
    # Process files
    process_batch(input_files, top_genes_by_ct, outDIR, args.mode, Annotat, exclude_genes, args.processes)

if __name__ == '__main__':
    main()
