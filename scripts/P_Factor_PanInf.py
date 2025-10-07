import sys
import numpy as np
import pandas as pd
import multiprocessing
import argparse
import yaml
import os
import logging
from functools import partial
from pathlib import Path

sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
sys.path.insert(1, '/home/jw3514/Work/UNIMED/src')
from CellType_PSY import *
from UNIMED import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze GWAS data with gene dropping')
    parser.add_argument('--disorder', type=str, required=True, help='Name of disorder to analyze')
    parser.add_argument('--processes', type=int, default=20, help='Number of processes to use')
    parser.add_argument('--start', type=int, default=1000, help='Start number of genes to drop')
    parser.add_argument('--end', type=int, default=17000, help='End number of genes to drop')
    parser.add_argument('--step', type=int, default=1000, help='Step size for gene dropping')
    parser.add_argument('--top_genes_mode', type=str, choices=['fixed', 'percentage'], default='fixed', help='Mode for selecting top genes: fixed number or percentage')
    parser.add_argument('--top_n', type=int, default=2000, help='Number of top genes to analyze (when using fixed mode)')
    parser.add_argument('--top_percentage', type=float, default=0.1, help='Percentage of top genes to analyze (when using percentage mode)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['HumanCT', 'MouseCT', 'MouseSTR'], default='HumanCT', help='Analysis mode')
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
    
def load_config(config_path):
    """Load configuration from YAML file with validation"""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required config sections
        required_sections = ['bias_matrices', 'pfactor_data']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        logging.info(f"Successfully loaded configuration from {config_path}")
        return config

    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise

def load_data(config, mode):
    """Load data based on configuration and mode with comprehensive validation"""
    try:
        logging.info(f"Loading data for mode: {mode}")

        # Validate mode
        if mode not in config['bias_matrices']:
            raise ValueError(f"Mode '{mode}' not found in bias_matrices config")

        # Load bias matrix based on mode
        bias_matrix_path = config['bias_matrices'][mode]
        if not os.path.exists(bias_matrix_path):
            raise FileNotFoundError(f"Bias matrix file not found: {bias_matrix_path}")

        logging.info(f"Loading bias matrix from: {bias_matrix_path}")
        if bias_matrix_path.endswith('.parquet'):
            BiasMat = pd.read_parquet(bias_matrix_path)
        else:
            BiasMat = pd.read_csv(bias_matrix_path, index_col=0)

        if mode == "HumanCT":
            BiasMat.columns = BiasMat.columns.astype(int)

        logging.info(f"Bias matrix shape: {BiasMat.shape}")

        # Load ranking data from config
        pfactor_data = config['pfactor_data']

        # Load LOEUF rankings
        loeuf_path = pfactor_data['loeuf_rankings']
        if not os.path.exists(loeuf_path):
            raise FileNotFoundError(f"LOEUF rankings file not found: {loeuf_path}")
        LOEUF_RankingDF = pd.read_csv(loeuf_path, index_col="EntrezID")
        logging.info(f"LOEUF rankings shape: {LOEUF_RankingDF.shape}")

        # Load PSD rankings
        psd_path = pfactor_data['psd_rankings']
        if not os.path.exists(psd_path):
            raise FileNotFoundError(f"PSD rankings file not found: {psd_path}")
        PSD_RankingDF = pd.read_csv(psd_path, index_col=0)
        logging.info(f"PSD rankings shape: {PSD_RankingDF.shape}")

        # Validate GWAS directory
        DIR = pfactor_data['gwas_dir']
        if not os.path.exists(DIR):
            raise FileNotFoundError(f"GWAS directory not found: {DIR}")

        logging.info("Successfully loaded all data files")
        return DIR, BiasMat, LOEUF_RankingDF, PSD_RankingDF

    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise

def load_gwas_data(disorder, dir_path):
    """Load GWAS data for a specific disorder with validation"""
    try:
        gwas_file = f"{dir_path}{disorder}.magma.genes.out.tsv"
        if not os.path.exists(gwas_file):
            raise FileNotFoundError(f"GWAS file not found: {gwas_file}")

        logging.info(f"Loading GWAS data for disorder: {disorder}")
        gwas_df = pd.read_csv(gwas_file, sep="\t", index_col="GENE")

        # Validate required columns
        if "ZSTAT" not in gwas_df.columns:
            raise ValueError(f"ZSTAT column not found in GWAS file: {gwas_file}")

        gwas_df = gwas_df.sort_values("ZSTAT", ascending=False)
        logging.info(f"GWAS data shape: {gwas_df.shape}")
        return gwas_df

    except Exception as e:
        logging.error(f"Failed to load GWAS data for {disorder}: {str(e)}")
        raise

def get_top_genes_by_ct(BiasMat, top_genes_mode, top_n=None, top_percentage=None):
    """Get top genes by cell type using either fixed number or percentage"""
    if top_genes_mode == 'fixed':
        if top_n is None:
            raise ValueError("top_n must be specified when using fixed mode")
        top_genes_by_ct = {col: BiasMat[col].nlargest(top_n).index.tolist() for col in BiasMat.columns}
    elif top_genes_mode == 'percentage':
        if top_percentage is None:
            raise ValueError("top_percentage must be specified when using percentage mode")
        topN = int(top_percentage * BiasMat.shape[0])
        top_genes_by_ct = {col: BiasMat[col].nlargest(topN).index.tolist() for col in BiasMat.columns}
    else:
        raise ValueError("top_genes_mode must be either 'fixed' or 'percentage'")

    return top_genes_by_ct

def process_gene_dropping_analysis(N_Genes_Drop, disorder, BiasMat, gene_ranking_source, GWAS_DF,
                                 top_genes_mode, top_n, top_percentage, mode, Anno,
                                 analysis_type, output_base_dir="results/KillPSD"):
    """
    Unified function to process gene dropping analysis for different ranking sources.

    Args:
        N_Genes_Drop: Number of top genes to drop
        disorder: Disorder name
        BiasMat: Expression bias matrix
        gene_ranking_source: DataFrame with gene rankings (LOEUF, PSD, or GWAS)
        GWAS_DF: GWAS data
        top_genes_mode: 'fixed' or 'percentage'
        top_n: Number of top genes (for fixed mode)
        top_percentage: Percentage of top genes (for percentage mode)
        mode: Analysis mode (HumanCT, MouseCT, MouseSTR)
        Anno: Annotation data
        analysis_type: Type of analysis ('LOEUF', 'PSD', 'DZ')
        output_base_dir: Base directory for output files
    """
    try:
        logging.info(f"Processing {analysis_type} analysis: {disorder} (N_drop={N_Genes_Drop}, mode={mode})")

        # Validate inputs
        if N_Genes_Drop <= 0:
            raise ValueError(f"N_Genes_Drop must be positive, got: {N_Genes_Drop}")
        if N_Genes_Drop > len(gene_ranking_source):
            logging.warning(f"N_Genes_Drop ({N_Genes_Drop}) exceeds ranking source size ({len(gene_ranking_source)})")

        # Get top genes to drop from the specified ranking source
        top_genes = gene_ranking_source.head(N_Genes_Drop).index.values

        # Filter expression matrix by removing top genes (optimized with set operations)
        top_genes_set = set(top_genes)
        genes_to_drop_bias = [g for g in top_genes_set if g in BiasMat.index]
        BiasMat_filtered = BiasMat.drop(genes_to_drop_bias)

        # Filter GWAS data by removing top genes
        genes_to_drop_gwas = [g for g in top_genes_set if g in GWAS_DF.index]
        GWAS_DF_filtered = GWAS_DF.drop(genes_to_drop_gwas)

        logging.info(f"Dropped {len(genes_to_drop_bias)} genes from bias matrix, {len(genes_to_drop_gwas)} from GWAS")
        logging.info(f"Filtered bias matrix shape: {BiasMat_filtered.shape}, GWAS shape: {GWAS_DF_filtered.shape}")

        # Calculate top genes by cell type from filtered bias matrix
        top_genes_by_ct = get_top_genes_by_ct(BiasMat_filtered, top_genes_mode, top_n, top_percentage)

        # Log summary of top genes selected
        total_genes_selected = sum(len(genes) for genes in top_genes_by_ct.values())
        logging.info(f"Selected {total_genes_selected} total genes across {len(top_genes_by_ct)} cell types/regions")

        # Perform enrichment analysis
        results = calculate_cluster_topGene_Enrichment(GWAS_DF_filtered["ZSTAT"], top_genes_by_ct, Anno, mode)

        # Set appropriate index based on mode
        index_col = "Structure" if mode == "MouseSTR" else "ct_idx"
        results = results.set_index(index_col)

        # Create output directory and save results
        output_dir = Path(output_base_dir) / analysis_type
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{disorder}_DropTop_{N_Genes_Drop}_{mode}.csv"
        results.to_csv(output_file)

        logging.info(f"✓ Completed {analysis_type} analysis for {disorder} (N_drop={N_Genes_Drop}, mode={mode})")
        logging.info(f"Results saved to: {output_file}")

    except Exception as e:
        logging.error(f"✗ Error in {analysis_type} analysis for {disorder} (N_drop={N_Genes_Drop}): {str(e)}")
        raise

# Wrapper functions for backward compatibility and cleaner partial function creation
def process_n_genes_LOEUF(N_Genes_Drop, disorder, BiasMat, LOEUF_RankingDF, GWAS_DF,
                         top_genes_mode, top_n, top_percentage, mode, Anno):
    """Process LOEUF-based gene dropping analysis"""
    return process_gene_dropping_analysis(
        N_Genes_Drop, disorder, BiasMat, LOEUF_RankingDF, GWAS_DF,
        top_genes_mode, top_n, top_percentage, mode, Anno, "LOEUF"
    )

def process_n_genes_PSD(N_Genes_Drop, disorder, BiasMat, PSD_RankingDF, GWAS_DF,
                       top_genes_mode, top_n, top_percentage, mode, Anno):
    """Process PSD-based gene dropping analysis"""
    return process_gene_dropping_analysis(
        N_Genes_Drop, disorder, BiasMat, PSD_RankingDF, GWAS_DF,
        top_genes_mode, top_n, top_percentage, mode, Anno, "PSD"
    )

def process_n_genes_DZ(N_Genes_Drop, disorder, BiasMat, GWAS_DF,
                      top_genes_mode, top_n, top_percentage, mode, Anno):
    """Process disorder GWAS-based gene dropping analysis"""
    return process_gene_dropping_analysis(
        N_Genes_Drop, disorder, BiasMat, GWAS_DF, GWAS_DF,
        top_genes_mode, top_n, top_percentage, mode, Anno, "DZ"
    )

def create_analysis_summary(args, config, BiasMat, LOEUF_RankingDF, PSD_RankingDF, GWAS_DF):
    """Create and log analysis summary"""
    topN_to_drop = np.arange(args.start, args.end, args.step)

    logging.info("="*60)
    logging.info("ANALYSIS SUMMARY")
    logging.info("="*60)
    logging.info(f"Disorder: {args.disorder}")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Top genes mode: {args.top_genes_mode}")
    if args.top_genes_mode == 'fixed':
        logging.info(f"Top N genes: {args.top_n}")
    else:
        logging.info(f"Top percentage: {args.top_percentage}")
    logging.info(f"Gene drop range: {args.start} to {args.end} (step: {args.step})")
    logging.info(f"Number of drop points: {len(topN_to_drop)}")
    logging.info(f"Processes: {args.processes}")
    logging.info(f"Total analyses to run: {len(topN_to_drop) * 3}")  # 3 analysis types
    logging.info("="*60)

def main():
    """Main function with comprehensive error handling and progress tracking"""
    import time
    start_time = time.time()

    try:
        args = parse_arguments()
        logging.info("Starting P-Factor PanInf Analysis")

        # Load configuration
        config = load_config(args.config)

        # Load all required data based on mode
        DIR, BiasMat, LOEUF_RankingDF, PSD_RankingDF = load_data(config, args.mode)
        GWAS_DF = load_gwas_data(args.disorder, DIR)

        # Load appropriate annotation based on mode
        logging.info(f"Loading annotations for mode: {args.mode}")
        if args.mode == "HumanCT":
            Annotation = Anno  # Global annotation from CellType_PSY
        elif args.mode == "MouseSTR":
            Annotation = STR2Region()
        elif args.mode == "MouseCT":
            mouse_ct_path = config['goterm_data']['mouse_ct_annotation']
            if not os.path.exists(mouse_ct_path):
                raise FileNotFoundError(f"Mouse CT annotation file not found: {mouse_ct_path}")
            Annotation = pd.read_csv(mouse_ct_path, index_col="cluster_id_label")

        # Create analysis summary
        create_analysis_summary(args, config, BiasMat, LOEUF_RankingDF, PSD_RankingDF, GWAS_DF)

        # Define range of genes to drop
        topN_to_drop = np.arange(args.start, args.end, args.step)

        # Create partial functions with fixed arguments for each analysis type
        process_loeuf = partial(process_n_genes_LOEUF,
                              disorder=args.disorder,
                              BiasMat=BiasMat,
                              LOEUF_RankingDF=LOEUF_RankingDF,
                              GWAS_DF=GWAS_DF,
                              top_genes_mode=args.top_genes_mode,
                              top_n=args.top_n,
                              top_percentage=args.top_percentage,
                              mode=args.mode,
                              Anno=Annotation)

        process_psd = partial(process_n_genes_PSD,
                            disorder=args.disorder,
                            BiasMat=BiasMat,
                            PSD_RankingDF=PSD_RankingDF,
                            GWAS_DF=GWAS_DF,
                            top_genes_mode=args.top_genes_mode,
                            top_n=args.top_n,
                            top_percentage=args.top_percentage,
                            mode=args.mode,
                            Anno=Annotation)

        process_dz = partial(process_n_genes_DZ,
                           disorder=args.disorder,
                           BiasMat=BiasMat,
                           GWAS_DF=GWAS_DF,
                           top_genes_mode=args.top_genes_mode,
                           top_n=args.top_n,
                           top_percentage=args.top_percentage,
                           mode=args.mode,
                           Anno=Annotation)

        # Run parallel processing for each analysis type
        analysis_types = [("LOEUF", process_loeuf), ("PSD", process_psd), ("DZ", process_dz)]

        for analysis_name, process_func in analysis_types:
            logging.info(f"Starting {analysis_name} analysis...")
            analysis_start = time.time()

            with multiprocessing.Pool(processes=args.processes) as pool:
                pool.map(process_func, topN_to_drop)

            analysis_time = time.time() - analysis_start
            logging.info(f"Completed {analysis_name} analysis in {analysis_time:.2f} seconds")

        # Final summary
        total_time = time.time() - start_time
        logging.info("="*60)
        logging.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logging.info(f"Total execution time: {total_time:.2f} seconds")
        logging.info(f"Processed {len(topN_to_drop) * 3} total analyses")
        logging.info("="*60)

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
# Example usage:
# Fixed number of genes:
# python P_Factor_PanInf.py --disorder "Bipolar" --processes 20 --start 1000 --end 17000 --step 1000 --top_genes_mode fixed --top_n 2000 --mode HumanCT --config config/config.yaml
#
# Percentage of genes:
# python P_Factor_PanInf.py --disorder "Bipolar" --processes 20 --start 1000 --end 17000 --step 1000 --top_genes_mode percentage --top_percentage 0.1 --mode HumanCT --config config/config.yaml
