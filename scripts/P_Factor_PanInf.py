import sys
import numpy as np
import pandas as pd
import multiprocessing
import argparse
from functools import partial

sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
sys.path.insert(1, '/home/jw3514/Work/UNIMED/src')
from CellType_PSY import *
from UNIMED import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze GWAS data with gene dropping')
    parser.add_argument('--disorder', type=str, required=True, help='Name of disorder to analyze')
    parser.add_argument('--processes', type=int, default=20, help='Number of processes to use')
    parser.add_argument('--start', type=int, default=1000, help='Start number of genes to drop')
    parser.add_argument('--end', type=int, default=17000, help='End number of genes to drop')
    parser.add_argument('--step', type=int, default=1000, help='Step size for gene dropping')
    parser.add_argument('--top_n', type=int, default=2000, help='Number of top genes to analyze')
    return parser.parse_args()

def calculate_cluster_topGene_Enrichment(Disorder_Series, top_genes_by_ct, Anno, mode):
    """Calculate correlations for all clusters efficiently."""
    results = []
    for ct_idx, genes in top_genes_by_ct.items():
        result = gene_set_enrichment_test(Disorder_Series, genes)
        results.append({
            'ct_idx' if mode != "MouseSTR" else 'Structure': ct_idx,
            'beta': result['effect_size'],
            'pvalue': result['pvalue']
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='beta', ascending=False)

    if mode == "HumanCT":
        results_df = AnnotateCTDat(results_df, Anno)
    elif mode == "MouseSTR":
        results_df["Region"] = [Anno[ct_idx] for ct_idx in results_df['Structure'].values]
        results_df = results_df.reset_index(drop=True)
        results_df["Rank"] = np.arange(1, len(results_df) + 1)
    elif mode == "MouseCT":
        results_df['class_id_label'] = Anno.loc[results_df['ct_idx'].values, "class_id_label"].values
        results_df['subclass_id_label'] = Anno.loc[results_df['ct_idx'].values, "subclass_id_label"].values

    return results_df
    
def load_data():
    # Constants
    DIR = "/home/jw3514/Work/UNIMED/dat/Genetics/GWAS/MagmaGeneRes/"
    LOEUF_PATH = "/home/jw3514/Work/UNIMED/dat/LOEUF_Rankings.csv"
    PSD_PATH = "/home/jw3514/Work/UNIMED/dat/Pfactor/HumanCT_GenePSD_Correlation.csv"
    
    HUMANCT_PATH = "/home/jw3514/Work/CellType_Psy/dat/Test.BiasMat/HumanCT.TPMFilt.Spec.Percentile.csv"
    
    # Load expression data
    HumanCT_Z2_scores = pd.read_csv(HUMANCT_PATH, index_col=0)
    HumanCT_Z2_scores.columns = HumanCT_Z2_scores.columns.astype(int)
    
    # Load LOEUF rankings
    LOEUF_RankingDF = pd.read_csv(LOEUF_PATH, index_col="EntrezID")
    PSD_RankingDF = pd.read_csv(PSD_PATH, index_col=0)

    return DIR, HumanCT_Z2_scores, LOEUF_RankingDF, PSD_RankingDF

def load_gwas_data(disorder, dir_path):
    """Load GWAS data for a specific disorder"""
    gwas_df = pd.read_csv(f"{dir_path}{disorder}.magma.genes.out.tsv", sep="\t", index_col="GENE")
    return gwas_df.sort_values("ZSTAT", ascending=False)

def process_n_genes_LOEUF(N_Genes_Drop, disorder, HumanCT_Z2_scores, LOEUF_RankingDF, GWAS_DF, top_n):
    """Process analysis for a given number of genes to drop"""
    try:
        # Get top genes to drop
        top_genes = LOEUF_RankingDF.head(N_Genes_Drop).index.values
        
        # Filter expression matrix
        genes_to_drop = [g for g in top_genes if g in HumanCT_Z2_scores.index]
        Z2Mat = HumanCT_Z2_scores.copy().drop(genes_to_drop)
        
        # Filter GWAS data
        genes_to_drop_DZ = [g for g in top_genes if g in GWAS_DF.index]
        GWAS_DF_filtered = GWAS_DF.copy().drop(genes_to_drop_DZ)
        
        # Calculate enrichment
        top_genes_by_ct = {col: Z2Mat[col].nlargest(top_n).index.tolist() for col in Z2Mat.columns}
        results = calculate_cluster_topGene_Enrichment(GWAS_DF_filtered["ZSTAT"], top_genes_by_ct, Anno, "HumanCT")
        results = results.set_index("ct_idx")
        
        # Save results
        results.to_csv(f"../dat/KillPSD/LOEUF/{disorder}_DropTop_{N_Genes_Drop}_HumanCT.csv")
    except Exception as e:
        print(f"Error in process_n_genes_LOEUF: {str(e)}")
        raise

def process_n_genes_PSD(N_Genes_Drop, disorder, HumanCT_Z2_scores, PSD_RankingDF, GWAS_DF, top_n):
    """Process analysis for a given number of genes to drop"""
    try:
        # Get top genes to drop
        top_genes = PSD_RankingDF.head(N_Genes_Drop).index.values
        
        # Filter expression matrix
        genes_to_drop = [g for g in top_genes if g in HumanCT_Z2_scores.index]
        Z2Mat = HumanCT_Z2_scores.copy().drop(genes_to_drop)
        
        # Filter GWAS data
        genes_to_drop_DZ = [g for g in top_genes if g in GWAS_DF.index]     
        GWAS_DF_filtered = GWAS_DF.copy().drop(genes_to_drop_DZ)
        
        # Calculate enrichment
        top_genes_by_ct = {col: Z2Mat[col].nlargest(top_n).index.tolist() for col in Z2Mat.columns}
        results = calculate_cluster_topGene_Enrichment(GWAS_DF_filtered["ZSTAT"], top_genes_by_ct, Anno, "HumanCT")
        results = results.set_index("ct_idx")
        
        # Save results
        results.to_csv(f"../dat/KillPSD/PSD/{disorder}_DropTop_{N_Genes_Drop}_HumanCT.csv")
    except Exception as e:
        print(f"Error in process_n_genes_PSD: {str(e)}")
        raise

def process_n_genes_DZ(N_Genes_Drop, disorder, HumanCT_Z2_scores, GWAS_DF, top_n):
    """Process analysis for a given number of genes to drop based on disorder GWAS ranking"""
    try:
        # Get top genes to drop from GWAS ranking
        top_genes = GWAS_DF.head(N_Genes_Drop).index.values
        
        # Filter expression matrix
        genes_to_drop = [g for g in top_genes if g in HumanCT_Z2_scores.index]
        Z2Mat = HumanCT_Z2_scores.copy().drop(genes_to_drop)
        
        # Filter GWAS data
        genes_to_drop_DZ = [g for g in top_genes if g in GWAS_DF.index]
        GWAS_DF_filtered = GWAS_DF.copy().drop(genes_to_drop_DZ)
        
        # Calculate enrichment
        top_genes_by_ct = {col: Z2Mat[col].nlargest(top_n).index.tolist() for col in Z2Mat.columns}
        results = calculate_cluster_topGene_Enrichment(GWAS_DF_filtered["ZSTAT"], top_genes_by_ct, Anno, "HumanCT")
        results = results.set_index("ct_idx")
        
        # Save results
        results.to_csv(f"../dat/KillPSD/DZ/{disorder}_DropTop_{N_Genes_Drop}_HumanCT.csv")
    except Exception as e:
        print(f"Error in process_n_genes_DZ: {str(e)}")
        raise

def main():
    try:
        args = parse_arguments()
        
        # Load all required data
        DIR, HumanCT_Z2_scores, LOEUF_RankingDF, PSD_RankingDF = load_data()
        GWAS_DF = load_gwas_data(args.disorder, DIR)
        
        # Define range of genes to drop
        topN_to_drop = np.arange(args.start, args.end, args.step)
        
        # Create partial functions with fixed arguments for each analysis type
        process_loeuf = partial(process_n_genes_LOEUF,  # Fixed function name
                              disorder=args.disorder,
                              HumanCT_Z2_scores=HumanCT_Z2_scores,
                              LOEUF_RankingDF=LOEUF_RankingDF,
                              GWAS_DF=GWAS_DF,
                              top_n=args.top_n)
                              
        process_psd = partial(process_n_genes_PSD,
                            disorder=args.disorder,
                            HumanCT_Z2_scores=HumanCT_Z2_scores,
                            PSD_RankingDF=PSD_RankingDF,
                            GWAS_DF=GWAS_DF,
                            top_n=args.top_n)
                            
        process_dz = partial(process_n_genes_DZ,
                           disorder=args.disorder,
                           HumanCT_Z2_scores=HumanCT_Z2_scores,
                           GWAS_DF=GWAS_DF,
                           top_n=args.top_n)
        
        # Run parallel processing for each analysis
        with multiprocessing.Pool(processes=args.processes) as pool:
            pool.map(process_loeuf, topN_to_drop)
            pool.map(process_psd, topN_to_drop)
            pool.map(process_dz, topN_to_drop)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
# Example usage:
# python P_Factor_PanInf.py --disorder "Bipolar" --processes 20 --start 1000 --end 17000 --step 1000 --top_n 2000
