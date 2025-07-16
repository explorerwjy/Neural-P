# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# Scripts_BiasCal.py
# ========================================================================================================
import multiprocessing
from multiprocessing import Pool
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
from CellType_PSY import *


#### HumanCT 
def plot_cluster_correlation_HumanCT(cluster, SCZMutDF, specificity_scores, eff_label="ZSTAT", plot=False):
    """Calculate correlations between cluster bias and effect sizes."""
    entrez_list = SCZMutDF.index.values
    Zscore_list = SCZMutDF[eff_label].values
    Bias_list = specificity_scores.loc[entrez_list, cluster]
    valid_mask = ~np.isnan(Zscore_list) & ~np.isnan(Bias_list)

    Zscore_list_clean = Zscore_list[valid_mask]
    Bias_list_clean = Bias_list[valid_mask]

    spearman_corr, spearman_p = stats.spearmanr(Zscore_list_clean, Bias_list_clean)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Bias_list_clean, Zscore_list_clean, alternative="greater")
    
    if plot:
        plt.figure(figsize=(6,4))
        plt.scatter(Bias_list_clean, Zscore_list_clean, s=1)
        x_range = np.array([min(Bias_list_clean), max(Bias_list_clean)])
        plt.plot(x_range, slope * x_range + intercept, 'r', 
                label=f'β = {slope:.2f}\nR² = {r_value**2:.3f}\np = {p_value:.2e}')
        plt.xlabel("Bias")
        plt.ylabel("Z-score")
        plt.legend()
        plt.show()

    return spearman_corr, spearman_p, slope, std_err, r_value, p_value

def calculate_cluster_correlations_HumanCT(specificity_scores, Spark_Meta_test, Anno, eff_label="ZSTAT"):
    """Calculate correlations for all clusters."""
    results = []
    for cluster in specificity_scores.columns:
        spearman_corr, spearman_p, slope, std_err, r_value, p_value = plot_cluster_correlation_HumanCT(
            cluster, Spark_Meta_test, specificity_scores, eff_label=eff_label
        )
        results.append({
            'Cluster': cluster,
            'SuperCluster': Anno.loc[cluster, "Supercluster"],
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_value': spearman_p,
            'Slope': slope,
            'Std_err': std_err,
            'R_value': r_value, 
            'P_value': p_value
        })
    return pd.DataFrame(results).sort_values(by="Spearman_P_value", ascending=True)

#### MouseSTR
def plot_cluster_correlation_MouseSTR(cluster, SCZMutDF, specificity_scores, eff_label="ZSTAT", plot=False):
    """Plot correlation between cluster bias and effect sizes."""
    entrez_list = SCZMutDF.index.values
    Zscore_list = SCZMutDF[eff_label].values
    Bias_list = specificity_scores.loc[entrez_list, cluster]
    valid_mask = ~np.isnan(Zscore_list) & ~np.isnan(Bias_list)

    Zscore_list_clean = Zscore_list[valid_mask]
    Bias_list_clean = Bias_list[valid_mask]

    spearman_corr, spearman_p = stats.spearmanr(Zscore_list_clean, Bias_list_clean)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Bias_list_clean, Zscore_list_clean, alternative="greater")
    
    if plot:
        plt.scatter(Bias_list_clean, Zscore_list_clean, s=1)
        x_range = np.array([min(Bias_list_clean), max(Bias_list_clean)])
        plt.plot(x_range, slope * x_range + intercept, 'r',
                label=f'β = {slope:.2f}\nR² = {r_value**2:.3f}\np = {p_value:.2e}')
        plt.xlabel("Bias")
        plt.ylabel("Z-score")
        plt.legend()
        plt.show()
        
    return spearman_corr, spearman_p, slope, std_err, r_value, p_value

def calculate_str_correlations_MouseSTR(specificity_scores, Spark_Meta_test, Anno, eff_label="ZSTAT"):
    results = []
    for structure in specificity_scores.columns:
        spearman_corr, spearman_p, slope, std_err, r_value, p_value = plot_cluster_correlation_MouseSTR(
            structure, Spark_Meta_test, specificity_scores, eff_label=eff_label)
        results.append({
            'structure': structure,
            'regions': Anno[structure],
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_value': spearman_p,
            'Slope': slope,
            'Std_err': std_err,
            'R_value': r_value,
            'P_value': p_value
        })
    return pd.DataFrame(results).sort_values(by="Spearman_P_value", ascending=True)

#### MouseCT
def plot_cluster_correlation_MouseCT(cluster, SCZMutDF, specificity_scores, eff_label="ZSTAT", plot=False):
    """Plot correlation between cluster bias and effect sizes."""
    entrez_list = SCZMutDF.index.values
    Zscore_list = SCZMutDF[eff_label].values
    Bias_list = specificity_scores.loc[entrez_list, cluster]
    valid_mask = ~np.isnan(Zscore_list) & ~np.isnan(Bias_list)

    Zscore_list_clean = Zscore_list[valid_mask]
    Bias_list_clean = Bias_list[valid_mask]

    spearman_corr, spearman_p = stats.spearmanr(Zscore_list_clean, Bias_list_clean)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Bias_list_clean, Zscore_list_clean, alternative="greater")
    
    if plot:
        plt.scatter(Bias_list_clean, Zscore_list_clean, s=1)
        x_range = np.array([min(Bias_list_clean), max(Bias_list_clean)])
        plt.plot(x_range, slope * x_range + intercept, 'r',
                label=f'β = {slope:.2f}\nR² = {r_value**2:.3f}\np = {p_value:.2e}')
        plt.xlabel("Bias")
        plt.ylabel("Z-score")
        plt.legend()
        plt.show()
        
    return spearman_corr, spearman_p, slope, std_err, r_value, p_value

def calculate_cluster_correlations_MouseCT(specificity_scores, Spark_Meta_test, ClusterAnn, eff_label="ZSTAT"):
    results = []
    for cluster in specificity_scores.columns:
        spearman_corr, spearman_p, slope, std_err, r_value, p_value = plot_cluster_correlation_MouseCT(
            cluster, Spark_Meta_test, specificity_scores, eff_label=eff_label)
        results.append({
            'Cluster': cluster,
            'CT_Class': ClusterAnn.loc[cluster, "class_id_label"],
            'CT_Subclass': ClusterAnn.loc[cluster, "subclass_id_label"],
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_value': spearman_p,
            'Slope': slope,
            'Std_err': std_err,
            'R_value': r_value,
            'P_value': p_value
        })
    return pd.DataFrame(results).sort_values(by="Spearman_P_value", ascending=True)

def GetOption():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InpFil', type=str, required=True, help='file contains list of gene weight files')
    parser.add_argument('-m', '--mode', type=str, choices=["HumanCT", "MouseCT", "MouseSTR"], required=True, help='model to use')
    parser.add_argument('--biasMat', type=str, help='bias matrix file')
    parser.add_argument('--outDir', type=str)
    args = parser.parse_args()
    return args

def process_file(filename, Z2Mat, outDIR, Model, Anno):

    name = ".".join(filename.split("/")[-1].split(".")[0:1])
    GWAS_DF = pd.read_csv(filename, sep="\t", index_col="GENE")
    GWAS_DF = GWAS_DF[GWAS_DF.index.isin(Z2Mat.index.values)]

    print(filename)
    print(name)

    if Model == "HumanCT":
        outname = f"{outDIR}/HumanCT.Bias.{name}.Z2.csv"
        HumanCellTypeBiasCal(Z2Mat, GWAS_DF, Anno, outname)
    elif Model == "MouseCT":
        outname = f"{outDIR}/MouseCT.Bias.{name}.Z2.csv"
        ABC_CellTypeBiasCal(Z2Mat, GWAS_DF, Anno, outname)
    elif Model == "MouseSTR":
        outname = f"{outDIR}/MouseSTR.Bias.{name}.Z2.csv"
        MouseSTRBiasCal(Z2Mat, GWAS_DF, Anno, outname)

def process_batch(files, Z2Mat, outDIR, Model, Anno):
    pool = multiprocessing.Pool(processes=20)
    pool.starmap(process_file, [(filename, Z2Mat, outDIR, Model, Anno) for filename in files])
    pool.close()
    pool.join()

def ABC_CellTypeBiasCal(Z2Mat, GWAS_DF, Anno, outname):
    GWAS_DF = GWAS_DF[GWAS_DF.index.isin(Z2Mat.index.values)]
    Disorder_GWAS_spec = calculate_cluster_correlations_MouseCT(Z2Mat, GWAS_DF, Anno, eff_label="ZSTAT")
    Disorder_GWAS_spec.to_csv(outname)
    return Disorder_GWAS_spec

def HumanCellTypeBiasCal(Z2Mat, GWAS_DF, Anno, outname):
    GWAS_DF = GWAS_DF[GWAS_DF.index.isin(Z2Mat.index.values)]
    Disorder_GWAS_spec = calculate_cluster_correlations_HumanCT(Z2Mat, GWAS_DF, Anno, eff_label="ZSTAT")
    Disorder_GWAS_spec.to_csv(outname)
    return Disorder_GWAS_spec

def MouseSTRBiasCal(Z2Mat, GWAS_DF, Anno, outname):
    GWAS_DF = GWAS_DF[GWAS_DF.index.isin(Z2Mat.index.values)]
    Disorder_GWAS_spec = calculate_str_correlations_MouseSTR(Z2Mat, GWAS_DF, Anno, eff_label="ZSTAT")
    Disorder_GWAS_spec.to_csv(outname)
    return Disorder_GWAS_spec

def main():
    args = GetOption()
    input_files = [x.strip() for x in open(args.InpFil, 'rt').readlines()]
    outDIR = args.outDir if args.outDir else "."
    
    if args.mode == "HumanCT":
        Annotat = Anno
        if args.biasMat == None:
            Z2Mat = pd.read_csv("/home/jw3514/Work/CellType_Psy/dat/HumanCTExpressionMats/Human.Cluster.Log2Mean.Z1clip5.Z2.clip3.Dec30.csv", index_col=0)
        else:
            Z2Mat = pd.read_csv(args.biasMat, index_col=0)
        Z2Mat.columns = Z2Mat.columns.astype(int)
    elif args.mode == "MouseSTR":
        Annotat = STR2Region()
        Z2Mat = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv", index_col=0)
    elif args.mode == "MouseCT":
        Annotat = pd.read_excel("../../data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx",
                          sheet_name="cluster_annotation", index_col="cluster_id_label")
        Z2Mat = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/Cluster_Z2Mat_ISHMatch.z1clip3.csv", index_col=0)
    
    process_batch(input_files, Z2Mat, outDIR, args.mode, Annotat)

if __name__ == '__main__':
    main()
