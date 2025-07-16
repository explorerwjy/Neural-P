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

def plot_cluster_correlation(cluster, SCZMutDF, specificity_scores, eff_label = "ZSTAT", plot=False):
    """
    Plot correlation between cluster bias and LGD odds ratio.
    
    Args:
        cluster (int): Cluster number to analyze
        SCZMutDF (pd.DataFrame): DataFrame containing mutation data
        specificity_scores (pd.DataFrame): Matrix of bias/specificity scores
    """
    entrez_list = SCZMutDF.index.values
    Zscore_list = SCZMutDF[eff_label].values
    Bias_list = specificity_scores.loc[entrez_list, cluster]
    valid_mask = ~np.isnan(Zscore_list) & ~np.isnan(Bias_list)

    # Calculate correlations
    spearman_corr, spearman_p = stats.spearmanr(Zscore_list[valid_mask], Bias_list[valid_mask])
    pearson_corr, pearson_p = stats.pearsonr(Zscore_list[valid_mask], Bias_list[valid_mask])
    #print(f"Spearman correlation: {spearman_corr}")
    #print(f"Pearson correlation: {pearson_corr}")

    # Clean data
    Zscore_list_clean = Zscore_list[valid_mask]
    Bias_list_clean = Bias_list[valid_mask]

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(Bias_list_clean, Zscore_list_clean, alternative="greater")
    
    if plot:
        # Create scatter plot
        plt.scatter(Bias_list_clean, Zscore_list_clean, s=1)

        # Add regression line
        x_range = np.array([min(Bias_list_clean), max(Bias_list_clean)])
        plt.plot(x_range, slope * x_range + intercept, 'r', 
                label=f'β = {slope:.2f}\nR² = {r_value**2:.3f}\np = {p_value:.2e}')

        plt.xlabel("Bias")
        plt.ylabel("Z-score") 
        plt.legend()
        plt.show()
    return spearman_corr, spearman_p, pearson_corr, pearson_p, slope,std_err, r_value, p_value, 

def cell_type_bias_Linear_fit(specificity_scores, Spark_Meta_test, Anno, eff_label="ZSTAT"):

    intercept, beta, ci_low, ci_high, r_value, p_value, std_err, pho, p = linear_fit(specificity_scores, Spark_Meta_test, Anno, eff_label=eff_label)
    return beta, r_value, std_err, p_value

def calculate_cluster_correlation_sub(specificity_scores, Spark_Meta_test, eff_label="ZSTAT"):
    # Create lists to store results
    clusters = []
    spearman_correlations = []
    spearman_pvalues = []
    slope_values = []
    std_err_values = []
    r_value_values = []
    p_value_values = []

    for cluster in specificity_scores.columns.values:
        spearman_corr, spearman_p, pearson_corr, pearson_p, slope, std_err, r_value, p_value = plot_cluster_correlation(cluster, Spark_Meta_test, specificity_scores, eff_label=eff_label)
        
        clusters.append(cluster)
        spearman_correlations.append(spearman_corr)
        spearman_pvalues.append(spearman_p)
        slope_values.append(slope)
        std_err_values.append(std_err)
        r_value_values.append(r_value)
        p_value_values.append(p_value)

    # Create DataFrame with results
    corr_df_ASD = pd.DataFrame({
        'Cluster': clusters,
        'Spearman_Correlation': spearman_correlations,
        'Spearman_P_value': spearman_pvalues,
        'Slope': slope_values,
        'Std_err': std_err_values,
        'R_value': r_value_values,
        'P_value': p_value_values
    })
    corr_df_ASD = corr_df_ASD.sort_values(by="Spearman_P_value", ascending=True)
    return corr_df_ASD

def calculate_cluster_Correlation(Disorder_DF, SpecMat, TopGenes, Anno, mode):
    intersect_genes = np.array(list(set(Disorder_DF.index) & set(SpecMat.index)))
    SpecMat = SpecMat.loc[intersect_genes]
    Disorder_DF = Disorder_DF.loc[intersect_genes]
    if mode == "HumanCT":
        #results_df = HumanCT_AvgZ_Weighted(SpecMat, GeneZSTAT)
        results_df = calculate_cluster_correlation_sub(SpecMat, Disorder_DF, eff_label="ZSTAT")
        if Anno is None:
            raise ValueError("Annotation data is missing for HumanCT mode")
        results_df = AnnotateCTDat(results_df, Anno)
        results_df = results_df.sort_values(by='Slope', ascending=False)

    elif mode == "MouseSTR":
        if Anno is None:
            raise ValueError("Annotation data is missing for MouseSTR mode")
        results_df["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in results_df['Structure'].values]
        results_df = results_df.reset_index(drop=True)
        results_df["Rank"] = np.arange(1, len(results_df) + 1)
    elif mode == "MouseCT":
        if Anno is None:
            raise ValueError("Annotation data is missing for MouseCT mode")
        results_df = calculate_cluster_correlation_sub(SpecMat, Disorder_DF, eff_label="ZSTAT")
        if Anno is None:
            raise ValueError("Annotation data is missing for HumanCT mode")
        results_df = AnnotateCTDat(results_df, Anno)
        results_df = results_df.sort_values(by='Slope', ascending=False)
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
        
    results = calculate_cluster_Correlation(GWAS_DF, SpecMat, TopGenes, Anno, mode)
    
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
