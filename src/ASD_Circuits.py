import pandas as pd
import csv 
import re
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os
import sys
import seaborn as sns
import random
import bisect
import collections
from collections import Counter
import scipy.stats 
import re
#import statsmodels.api as sm
import statsmodels.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
#from scipy.stats import binom_test
import itertools
#import mygene
import igraph as ig
import gzip as gz
import pickle as pk
import copy
import zipfile
from tabulate import tabulate
import logging
from typing import Dict, Any
import gc

def LoadGeneINFO():
    HGNC = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    HGNC["entrez_id"] = pd.to_numeric(HGNC["entrez_id"], errors='coerce').astype('Int64')
    HGNC_valid = HGNC.dropna(subset=['entrez_id'])
    ENSID2Entrez = dict(zip(HGNC_valid["ensembl_gene_id"].values, HGNC_valid["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC_valid["symbol"].values, HGNC_valid["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC_valid["entrez_id"].values, HGNC_valid["symbol"].values))
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol 

MajorBrainDivisions = "/home/jw3514/Work/ASD_Circuits/dat/structure2region.tsv" 
def STR2Region():
    str2reg_df = pd.read_csv(MajorBrainDivisions, delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    return str2reg

def GeneList2GW(GeneListFil):
    with open(GeneListFil, 'r') as f:
        GeneList = []
        for g in f.readlines():
            gene_symbol = g.strip()
            entrez_id = GeneSymbol2Entrez.get(gene_symbol)
            if entrez_id is not None:
                GeneList.append(int(entrez_id))
            else:
                GeneList.append(0)
    GW = dict(zip(GeneList, np.ones(len(GeneList))))
    return GW

def Dict2Fil(dict_, fil_):
    with open(fil_, 'wt') as f:
        writer = csv.writer(f)
        for k,v in dict_.items():
            writer.writerow([k, v])

def Fil2Dict(fil_):
    df = pd.read_csv(fil_, header=None)
    return dict(zip(df[0].values, df[1].values))

####
#####################################################################################
#####################################################################################
#### Annotation Functions
#####################################################################################
#####################################################################################
def add_class(BiasDF, ClusterAnn):
    for cluster, row in BiasDF.iterrows():
        if cluster in ClusterAnn.index:
            BiasDF.loc[cluster, "class_id_label"] = ClusterAnn.loc[cluster, "class_id_label"]
            BiasDF.loc[cluster, "subclass_id_label"] = ClusterAnn.loc[cluster, "subclass_id_label"]
            BiasDF.loc[cluster, "supertype_id_label"] = ClusterAnn.loc[cluster, "supertype_id_label"]
            if "CCF_broad.freq" in ClusterAnn.columns:
                BiasDF.loc[cluster, "CCF_broad.freq"] = ClusterAnn.loc[cluster, "CCF_broad.freq"]
            if "CCF_acronym.freq" in ClusterAnn.columns:
                BiasDF.loc[cluster, "CCF_acronym.freq"] = ClusterAnn.loc[cluster, "CCF_acronym.freq"]
            if "v3.size" in ClusterAnn.columns:
                BiasDF.loc[cluster, "v3.size"] = ClusterAnn.loc[cluster, "v3.size"]
            if "v2.size" in ClusterAnn.columns:
                BiasDF.loc[cluster, "v2.size"] = ClusterAnn.loc[cluster, "v2.size"]
        else:
            # Set default values if cluster not found in annotation
            BiasDF.loc[cluster, "class_id_label"] = "Unknown"
    return BiasDF

def AnnotateCTDat(df, Anno):
    for i, row in df.iterrows():
        df.loc[i, "Class"] = Anno.loc[int(i), "Class auto-annotation"]
        df.loc[i, "Supercluster"] = Anno.loc[int(i), "Supercluster"]
        df.loc[i, "Subtype"] = Anno.loc[int(i), "Subtype auto-annotation"]
        df.loc[i, "Neurotransmitter"] = Anno.loc[int(i), "Neurotransmitter auto-annotation"]
        df.loc[i, "Top three regions"] = Anno.loc[int(i), "Top three regions"]
        df.loc[i, "Top three dissections"] = Anno.loc[int(i), "Top three dissections"]
        df.loc[i, "Number of cells"] = Anno.loc[int(i), "Number of cells"] #Top three dissections
        df.loc[i, "Neuropeptide auto-annotation"] = Anno.loc[int(i), "Neuropeptide auto-annotation"] #Top three dissections
    df.index = [int(i) for i in df.index.values]
    return df

#####################################################################################
#####################################################################################
#### IQ function
#####################################################################################
#####################################################################################
# Global cache for expression matrices
_matrix_cache: Dict[str, pd.DataFrame] = {}

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting to appropriate dtypes.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Memory-optimized DataFrame
    """
    original_memory = df.memory_usage(deep=True).sum()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        # Check if column has only integers
        if df[col].dtype in ['int64', 'int32']:
            # Try to downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype in ['float64']:
            # Convert float64 to float32 if precision allows
            max_val = df[col].abs().max()
            if pd.notna(max_val) and max_val < np.finfo(np.float32).max:
                # Check if conversion preserves precision reasonably
                temp_col = df[col].astype(np.float32)
                if np.allclose(df[col].dropna(), temp_col.dropna(), rtol=1e-6):
                    df[col] = temp_col
    
    # Optimize categorical columns with low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    reduction = (original_memory - new_memory) / original_memory * 100
    
    logging.info(f"Memory optimization: {reduction:.1f}% reduction ({original_memory/1e6:.1f}MB → {new_memory/1e6:.1f}MB)")
    
    return df

def load_expression_matrix_cached(file_path: str, force_reload: bool = False) -> pd.DataFrame:
    """
    Load expression matrix with caching to avoid redundant I/O operations.
    
    Args:
        file_path: Path to the expression matrix file
        force_reload: Force reload even if cached
    
    Returns:
        Expression matrix DataFrame
    """
    cache_key = os.path.abspath(file_path)
    
    # Return cached version if available
    if cache_key in _matrix_cache and not force_reload:
        logging.info(f"Using cached matrix: {file_path}")
        return _matrix_cache[cache_key]
    
    # Load matrix
    logging.info(f"Loading expression matrix: {file_path}")
    if file_path.endswith('.parquet'):
        matrix = pd.read_parquet(file_path)
    elif file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
        matrix = pd.read_csv(file_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Optimize data types
    matrix = optimize_dataframe_dtypes(matrix)
    
    # Preserve column names as strings (cell type names)
    if matrix.columns.dtype == 'object':
        matrix.columns = matrix.columns.astype(str)
        logging.info(f"Column names preserved as strings: {len(matrix.columns)} cell types")
    
    # Cache the optimized matrix
    _matrix_cache[cache_key] = matrix
    logging.info(f"Cached matrix with shape {matrix.shape}")
    
    return matrix

def clear_matrix_cache():
    """Clear the expression matrix cache and free memory."""
    global _matrix_cache
    cache_size = len(_matrix_cache)
    _matrix_cache.clear()
    gc.collect()
    logging.info(f"Cleared matrix cache ({cache_size} matrices)")

def cleanup_memory():
    """Explicit memory cleanup."""
    gc.collect()
    logging.info("Memory cleanup completed")

def get_cache_info() -> Dict[str, Any]:
    """Get information about the current cache state."""
    cache_info = {}
    total_memory = 0
    
    for path, matrix in _matrix_cache.items():
        memory_mb = matrix.memory_usage(deep=True).sum() / 1e6
        cache_info[path] = {
            'shape': matrix.shape,
            'memory_mb': memory_mb,
            'dtypes': matrix.dtypes.value_counts().to_dict()
        }
        total_memory += memory_mb
    
    cache_info['total_memory_mb'] = total_memory
    cache_info['num_matrices'] = len(_matrix_cache)
    
    return cache_info
#####################################################################################
#####################################################################################
#### Bias Calculation
#####################################################################################
#####################################################################################
def MouseSTR_AvgZ_Weighted(ExpZscoreMat, Gene2Weights, csv_fil=None, results_dir="./results"):
    """
    Calculate weighted average bias scores for brain structures
    
    Parameters:
    -----------
    ExpZscoreMat : pd.DataFrame
        Expression z-score matrix (genes x structures)
    Gene2Weights : dict
        Gene weights dictionary (entrez_id -> weight)
    csv_fil : str, optional
        Filename to save results (will be saved in results_dir)
    results_dir : str, default "./results"
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame : Structure bias results with EFFECT and Rank columns
    """
    # Convert Gene2Weights to pandas Series for faster lookup
    weights_series = pd.Series(Gene2Weights)
    
    # Get intersection of genes using index operations
    valid_genes = ExpZscoreMat.index.intersection(weights_series.index)
    
    # Get weights and expression values using vectorized operations
    weights = weights_series[valid_genes].values
    expr_mat = ExpZscoreMat.loc[valid_genes]
    
    # Compute weighted averages for all cell types at once using matrix operations
    mask = ~np.isnan(expr_mat)
    # Broadcasting weights to match expr_mat shape
    weights_broadcast = weights[:, np.newaxis]
    # Multiply weights with expression values, handling nans
    weighted_vals = expr_mat.values * weights_broadcast * mask
    # Sum of weights where values are not nan
    weight_sums = weights_broadcast * mask
    # Compute weighted average across genes for each cell type
    EFFECTS = np.sum(weighted_vals, axis=0) / np.sum(weight_sums, axis=0)
    
    # Create results dataframe efficiently
    df = pd.DataFrame({
        'Structure': ExpZscoreMat.columns,
        'EFFECT': EFFECTS
    }).set_index('Structure')
    
    
    # Sort and add rank
    df = df.sort_values('EFFECT', ascending=False)
    df['Rank'] = np.arange(1, len(df) + 1)
    
    if csv_fil is not None:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, csv_fil)
        df.to_csv(output_path)
        print(f"Results saved to: {output_path}")
    
    return df

def HumanCT_AvgZ_Weighted(ExpZscoreMat, Gene2Weights, csv_fil=None):
    # Filter out genes not in expression matrix
    valid_genes = [g for g in Gene2Weights.keys() if g in ExpZscoreMat.index.values]
    
    # Get weights and expression values for valid genes
    weights = np.array([Gene2Weights[g] for g in valid_genes])
    expr_mat = ExpZscoreMat.loc[valid_genes]
    
    # Compute weighted average for each cell type
    CellTypes = ExpZscoreMat.columns.values
    EFFECTS = []
    for CT in CellTypes:
        expr_values = expr_mat[CT].values
        # Remove nan values and corresponding weights
        mask = ~np.isnan(expr_values)
        filtered_expr = expr_values[mask]
        filtered_weights = weights[mask]
        # Weighted average for this cell type, ignoring nans
        weighted_avg = np.average(filtered_expr, weights=filtered_weights)
        EFFECTS.append(weighted_avg)
        
    # Create results dataframe
    df = pd.DataFrame(data={"ct_idx": CellTypes, "EFFECT": EFFECTS}).set_index('ct_idx')
    df = df.sort_values("EFFECT", ascending=False)
    #df = df.reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    #df = df.set_index("ct_idx")
    
    if csv_fil is not None:
        df.to_csv(csv_fil)
        
    return df

def MouseCT_AvgZ_Weighted(ExpZscoreMat, Gene2Weights, csv_fil=None):
    # Convert Gene2Weights to pandas Series for faster lookup
    weights_series = pd.Series(Gene2Weights)
    
    # Get intersection of genes using index operations
    valid_genes = ExpZscoreMat.index.intersection(weights_series.index)
    
    # Get weights and expression values using vectorized operations
    weights = weights_series[valid_genes].values
    expr_mat = ExpZscoreMat.loc[valid_genes]
    
    # Compute weighted averages for all cell types at once using matrix operations
    mask = ~np.isnan(expr_mat)
    # Broadcasting weights to match expr_mat shape
    weights_broadcast = weights[:, np.newaxis]
    # Multiply weights with expression values, handling nans
    weighted_vals = expr_mat.values * weights_broadcast * mask
    # Sum of weights where values are not nan
    weight_sums = weights_broadcast * mask
    # Compute weighted average across genes for each cell type
    EFFECTS = np.sum(weighted_vals, axis=0) / np.sum(weight_sums, axis=0)
    
    # Create results dataframe efficiently
    df = pd.DataFrame({
        'ct_idx': ExpZscoreMat.columns,
        'EFFECT': EFFECTS
    }).set_index('ct_idx')
    
    # Sort and add rank
    df = df.sort_values('EFFECT', ascending=False)
    df['Rank'] = np.arange(1, len(df) + 1)
    
    if csv_fil is not None:
        df.to_csv(csv_fil)
        
    return df


def GetPermutationP_vectorized(null_matrix, observed_vals, greater_than=True):
    """Vectorized permutation p-value calculation for multiple tests."""
    # null_matrix shape: (n_permutations, n_cell_types)
    # observed_vals shape: (n_cell_types,)
    
    # Remove NaN values per cell type
    mask = ~np.isnan(null_matrix)
    n_valid = np.sum(mask, axis=0)
    
    # Calculate means and stds, handling NaNs
    means = np.nanmean(null_matrix, axis=0)
    stds = np.nanstd(null_matrix, axis=0)
    
    # Calculate z-scores
    z_scores = (observed_vals - means) / stds
    
    # Calculate p-values vectorized
    if greater_than:
        counts = np.sum(observed_vals[None, :] <= null_matrix, axis=0)
    else:
        counts = np.sum(observed_vals[None, :] >= null_matrix, axis=0)
    
    p_values = (counts + 1) / (n_valid + 1)
    obs_adjs = observed_vals - means
    
    return z_scores, p_values, obs_adjs

#####################################################################################
#####################################################################################
####  Circuit Related Functions
#####################################################################################
#####################################################################################

def calculate_circuit_scores(pc_scores_df, IpsiInfoMat, sort_by="PC1"):
    STR_Ranks = pc_scores_df.sort_values(sort_by, ascending=False).index.values
    topNs = list(range(200, 5, -1))
    SC_Agg_topN_score = []
    
    for topN in topNs:
        top_strs = STR_Ranks[:topN]
        score = ScoreCircuit_SI_Joint(top_strs, IpsiInfoMat)
        SC_Agg_topN_score.append(score)
        
    return np.array(SC_Agg_topN_score)
    
def ScoreCircuit_SI_Joint(STRs, InfoMat):
    CirInfo = InfoMat.loc[STRs, STRs]
    #N_events = len(STRs) * (len(STRs) - 1)
    N_events = np.count_nonzero(CirInfo)
    CirInfo = np.nan_to_num(CirInfo, nan=0)
    score = np.sum(CirInfo)
    return score/N_events

#####################################################################################
#####################################################################################
#### Analyze Neurotransmitter Systems
#####################################################################################
#####################################################################################
def create_gene_weights_by_system_category(neural_system_df, system=None, category=None, group=None, SOURCE_CATEGORIES=None, TARGET_CATEGORIES=None, save_dir="/home/jw3514/Work/ASD_Circuits_CellType/dat/Genetics/GeneWeights/"):
    """
    Create gene weights dictionary for specific neurotransmitter system and/or category
    
    Parameters:
    -----------
    neural_system_df : pd.DataFrame
        DataFrame with neurotransmitter system data
    system : str, optional
        Specific neurotransmitter system ('dopamine', 'serotonin', 'oxytocin', 'acetylcholine')
    category : str, optional  
        Specific category ('synthesis', 'receptor', etc.)
    group : str, optional
        Either 'source' or 'target' to group categories
    SOURCE_CATEGORIES : list, optional
        Categories for source genes (default: ['synthesis', 'transporter', 'storage_release'])
    TARGET_CATEGORIES : list, optional  
        Categories for target genes (default: ['receptor', 'degradation'])
    
    Returns:
    --------
    dict : Gene weights dictionary (entrez_id -> 1.0)
    """
    # Set default categories if not provided
    if SOURCE_CATEGORIES is None:
        SOURCE_CATEGORIES = ['synthesis', 'transporter', 'storage_release']
    if TARGET_CATEGORIES is None:
        TARGET_CATEGORIES = ['receptor', 'degradation']
        
    df = neural_system_df.copy()
    
    if system is not None:
        df = df[df['neurotransmitter_system'] == system]
    
    if category is not None:
        df = df[df['category'] == category]
    
    if group is not None:
        if group == 'source':
            df = df[df['category'].isin(SOURCE_CATEGORIES)]
        elif group == 'target':
            df = df[df['category'].isin(TARGET_CATEGORIES)]
    
    # Create gene weights dictionary
    gene_weights = {}
    for _, row in df.iterrows():
        entrez_id = row['entrez_id']
        if pd.notna(entrez_id) and entrez_id != 0:
            gene_weights[int(entrez_id)] = 1.0
    if save_dir is not None:
        fil_ = os.path.join(save_dir, f"NT_{system}_{group}.gw")
        Dict2Fil(gene_weights, fil_)
    return gene_weights


def analyze_neurotransmitter_systems_source_target(neural_system_df, bias_mat, SOURCE_CATEGORIES=None, TARGET_CATEGORIES=None, save_dir="/home/jw3514/Work/ASD_Circuits_CellType/dat/Genetics/GeneWeights/"):
    """
    Analyze all neurotransmitter systems with source, target, and combined biases
    Only uses genes from SOURCE_CATEGORIES and TARGET_CATEGORIES
    
    Parameters:
    -----------
    neural_system_df : pd.DataFrame
        DataFrame with neurotransmitter system data
    bias_mat : pd.DataFrame
        Structure bias matrix
    SOURCE_CATEGORIES : list, optional
        Categories for source genes (default: ['synthesis', 'transporter', 'storage_release'])
    TARGET_CATEGORIES : list, optional  
        Categories for target genes (default: ['receptor', 'degradation'])
        
    Returns:
    --------
    dict : Results for each system with 'source', 'target', 'combined' DataFrames
    """
    
    # Set default categories if not provided
    if SOURCE_CATEGORIES is None:
        SOURCE_CATEGORIES = ['synthesis', 'transporter', 'storage_release']
    if TARGET_CATEGORIES is None:
        TARGET_CATEGORIES = ['receptor', 'degradation']
    
    # Get the STR2Region mapping
    Anno = STR2Region()
    
    systems = neural_system_df['neurotransmitter_system'].unique()
    results = {}
    
    print("Analyzing neurotransmitter systems with SOURCE-TARGET grouping:")
    print(f"SOURCE categories: {SOURCE_CATEGORIES}")
    print(f"TARGET categories: {TARGET_CATEGORIES}")
    print("Note: Other categories (metabolism, processing) will be excluded\n")
    
    for system in systems:
        print(f"Analyzing {system.upper()} system...")
        results[system] = {}
        
        # Source genes analysis
        source_weights = create_gene_weights_by_system_category(neural_system_df, system=system, group='source', 
                                                               SOURCE_CATEGORIES=SOURCE_CATEGORIES, TARGET_CATEGORIES=TARGET_CATEGORIES)
        if source_weights:
            source_bias = MouseSTR_AvgZ_Weighted(bias_mat, source_weights)
            source_bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in source_bias.index.values]
            results[system]['source'] = source_bias
            print(f"  Source genes: {len(source_weights)}")
        else:
            print(f"  Source genes: 0 (no genes found)")
        
        # Target genes analysis
        target_weights = create_gene_weights_by_system_category(neural_system_df, system=system, group='target',
                                                               SOURCE_CATEGORIES=SOURCE_CATEGORIES, TARGET_CATEGORIES=TARGET_CATEGORIES)
        if target_weights:
            target_bias = MouseSTR_AvgZ_Weighted(bias_mat, target_weights)
            target_bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in target_bias.index.values]
            results[system]['target'] = target_bias
            print(f"  Target genes: {len(target_weights)}")
        else:
            print(f"  Target genes: 0 (no genes found)")
        
        # Combined analysis (source + target genes only)
        combined_weights = {}
        combined_weights.update(source_weights if source_weights else {})
        combined_weights.update(target_weights if target_weights else {})
        fil_ = os.path.join(save_dir, f"NT_{system}_combined.gw")
        Dict2Fil(combined_weights, fil_)
        
        if combined_weights:
            combined_bias = MouseSTR_AvgZ_Weighted(bias_mat, combined_weights)
            combined_bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in combined_bias.index.values]
            results[system]['combined'] = combined_bias
            print(f"  Combined genes: {len(combined_weights)} (source + target only)")
        else:
            print(f"  Combined genes: 0 (no genes found)")
        
        # Show which categories are present for this system
        system_df = neural_system_df[neural_system_df['neurotransmitter_system'] == system]
        available_categories = system_df['category'].unique()
        used_categories = [cat for cat in available_categories if cat in SOURCE_CATEGORIES + TARGET_CATEGORIES]
        excluded_categories = [cat for cat in available_categories if cat not in SOURCE_CATEGORIES + TARGET_CATEGORIES]
        
        print(f"  Used categories: {used_categories}")
        if excluded_categories:
            print(f"  Excluded categories: {excluded_categories}")
        print()
    
    return results


def plot_source_target_heatmap(results, top_n=15, save_plot=False, results_dir="./results"):
    """
    Create heatmap comparing source vs target across all systems
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    top_n : int, default 15
        Number of top structures to show
    save_plot : bool, default False
        Whether to save the plot to results directory
    results_dir : str, default "./results"
        Directory to save plots
    """
    comparison_data = {}
    all_structures = set()
    
    for system, system_data in results.items():
        for group_type in ['source', 'target', 'combined']:
            if group_type in system_data:
                key = f"{system}_{group_type}"
                comparison_data[key] = system_data[group_type]['EFFECT']
                all_structures.update(system_data[group_type].head(top_n).index)
    
    # Create comparison matrix
    comparison_df = pd.DataFrame(index=list(all_structures), columns=list(comparison_data.keys()))
    
    for key, effects in comparison_data.items():
        comparison_df[key] = effects
    
    # Fill NaN with 0 and sort by mean effect
    comparison_df = comparison_df.fillna(0)
    comparison_df = comparison_df.loc[comparison_df.mean(axis=1).sort_values(ascending=False).index]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(comparison_df) * 0.3)))
    sns.heatmap(comparison_df.head(top_n), 
                annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                yticklabels=[s.replace('_', ' ') for s in comparison_df.head(top_n).index],
                ax=ax)
    ax.set_title('Neurotransmitter Source vs Target Comparison - Top Structures')
    ax.set_xlabel('System_Group')
    ax.set_ylabel('Brain Structure')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_plot:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'neurotransmitter_heatmap_top{top_n}.svg')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {plot_path}")
    
    plt.show()
    return fig

def plot_source_target_only(results, top_n=15, save_plot=False, results_dir="./results"):
    """
    Visualize source vs target neurotransmitter system bias results
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    top_n : int, default 15
        Number of top structures to show
    save_plot : bool, default False
        Whether to save the plot to results directory
    results_dir : str, default "./results"
        Directory to save plots
    """
    
    systems = list(results.keys())
    n_systems = len(systems)
    
    # Create subplots
    fig, axes = plt.subplots(n_systems, 3, figsize=(20, 4*n_systems), dpi=480)
    if n_systems == 1:
        axes = axes.reshape(1, -1)
    
    for i, system in enumerate(systems):
        system_data = results[system]
        
        # Plot source
        if 'source' in system_data:
            ax = axes[i, 0] if n_systems > 1 else axes[0]
            top_source = system_data['source'].head(top_n)
            bars = ax.barh(range(len(top_source)), top_source['EFFECT'])
            ax.set_yticks(range(len(top_source)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_source.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Source\n(synthesis, transporter, storage_release)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('orange')
        
        # Plot target
        if 'target' in system_data:
            ax = axes[i, 1] if n_systems > 1 else axes[1]
            top_target = system_data['target'].head(top_n)
            bars = ax.barh(range(len(top_target)), top_target['EFFECT'])
            ax.set_yticks(range(len(top_target)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_target.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Target\n(receptor, degradation)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('blue')
        
        # Plot combined
        if 'combined' in system_data:
            ax = axes[i, 2] if n_systems > 1 else axes[2]
            top_combined = system_data['combined'].head(top_n)
            bars = ax.barh(range(len(top_combined)), top_combined['EFFECT'])
            ax.set_yticks(range(len(top_combined)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_combined.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Combined\n(source + target)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('gray')
    
    plt.tight_layout()
    
    if save_plot:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'neurotransmitter_barplots_top{top_n}.svg')
        fig.savefig(plot_path, dpi=480, bbox_inches='tight')
        print(f"Bar plots saved to: {plot_path}")
    
    plt.show()
    
    return fig

def save_neurotransmitter_results(results, results_dir="./results"):
    """
    Save neurotransmitter analysis results to CSV files in results directory
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    results_dir : str, default "./results"
        Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Saving neurotransmitter results to {results_dir}/")
    saved_files = []
    
    for system, system_data in results.items():
        for analysis_type, bias_df in system_data.items():
            filename = f"{system}_{analysis_type}_bias.csv"
            filepath = os.path.join(results_dir, filename)
            bias_df.to_csv(filepath)
            saved_files.append(filepath)
            print(f"  Saved: {filepath}")
    
    return saved_files
