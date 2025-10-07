import sys
from CellType_PSY import *
import pandas as pd
import numpy as np
import requests
import time
import re
import os
import scipy.stats
from statsmodels.api import OLS, add_constant
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.cluster import hierarchy

# Import plotting functions from separate plotting module and re-export them
try:
    # Try relative import first (when used as module)
    from .plotting import *
except ImportError:
    # Fall back to absolute import (when script run directly)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from plotting import *

MajorBrainDivisions = "/home/jw3514/Work/ASD_Circuits/dat/structure2region.tsv"

def STR2Region():
    str2reg_df = pd.read_csv(MajorBrainDivisions, delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    return str2reg

def LoadGeneINFO():
    #HGNC = pd.read_csv("../dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    HGNC = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    ENSID2Entrez = dict(zip(HGNC["ensembl_gene_id"].values, HGNC["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC["symbol"].values, HGNC["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC["entrez_id"].values, HGNC["symbol"].values))
    #allen_mouse_genes = loadgenelist("/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol

def map_gene_symbols(old_symbols, batch_size=500):
    """
    Map a list of gene symbols to current official HGNC symbols using the HGNC API

    Parameters:
    -----------
    old_symbols : list
        List of gene symbols to map
    batch_size : int
        Number of genes to query in each batch

    Returns:
    --------
    DataFrame with columns: input_symbol, official_symbol, status
    """
    # Create a mapping dictionary to store results
    mapped_genes = {
        'input_symbol': [],
        'official_symbol': [],
        'status': []
    }

    # Process genes in batches to avoid overwhelming the API
    for i in range(0, len(old_symbols), batch_size):
        batch = old_symbols[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(old_symbols)-1)//batch_size + 1}")

        # Process each gene in the batch
        for gene in batch:
            mapped_genes['input_symbol'].append(gene)

            try:
                # Query the HGNC REST API
                url = f"https://rest.genenames.org/search/symbol/{gene}"
                headers = {"Accept": "application/json"}
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()

                    # Check if any results were found
                    if data['response']['numFound'] > 0:
                        # Get the official symbol
                        for doc in data['response']['docs']:
                            # Check if it's the approved symbol
                            if 'status' in doc and doc['status'] == 'Approved':
                                mapped_genes['official_symbol'].append(doc['symbol'])
                                mapped_genes['status'].append('Approved')
                                break
                            # Check if it's a previous symbol that points to an approved one
                            elif 'symbol' in doc:
                                mapped_genes['official_symbol'].append(doc['symbol'])
                                mapped_genes['status'].append('Previous')
                                break
                        else:
                            # If no matching approved or previous symbol was found
                            mapped_genes['official_symbol'].append(None)
                            mapped_genes['status'].append('Not Found')
                    else:
                        mapped_genes['official_symbol'].append(None)
                        mapped_genes['status'].append('Not Found')
                else:
                    mapped_genes['official_symbol'].append(None)
                    mapped_genes['status'].append(f'API Error: {response.status_code}')

            except Exception as e:
                mapped_genes['official_symbol'].append(None)
                mapped_genes['status'].append(f'Error: {str(e)}')

            # Add a small delay to avoid overwhelming the API
            time.sleep(0.1)

    # Create a DataFrame from the mapping dictionary
    mapping_df = pd.DataFrame(mapped_genes)
    return mapping_df

def map_with_biomart(old_symbols, output_file="gene_mapping.csv"):
    """
    Alternative method using biomaRt through a simple Python interface
    """
    try:
        from pybiomart import Server

        # Connect to Ensembl BioMart
        server = Server(host='http://www.ensembl.org')
        ensembl = server.marts['ENSEMBL_MART_ENSEMBL']
        dataset = ensembl.datasets['hsapiens_gene_ensembl']

        # Prepare the query
        attributes = ['hgnc_symbol', 'external_gene_name', 'entrezgene_id',
                      'hgnc_id', 'previous_symbols', 'alias_symbol']

        # Get all human genes with their symbols and previous symbols
        gene_info = dataset.query(attributes=attributes)

        # Create a mapping dictionary
        mapping_dict = {}

        # Process each row to build a mapping from previous/alias symbols to current symbols
        for _, row in gene_info.iterrows():
            current_symbol = row['hgnc_symbol']
            if pd.notna(current_symbol):
                # Add current symbol to itself
                mapping_dict[current_symbol.upper()] = current_symbol

                # Add previous symbols
                if pd.notna(row['previous_symbols']):
                    for prev in str(row['previous_symbols']).split('|'):
                        if prev:
                            mapping_dict[prev.upper()] = current_symbol

                # Add aliases
                if pd.notna(row['alias_symbol']):
                    for alias in str(row['alias_symbol']).split('|'):
                        if alias:
                            mapping_dict[alias.upper()] = current_symbol

        # Map the old symbols to new ones
        results = []
        for symbol in old_symbols:
            if symbol.upper() in mapping_dict:
                results.append({
                    'input_symbol': symbol,
                    'official_symbol': mapping_dict[symbol.upper()],
                    'status': 'Mapped'
                })
            else:
                results.append({
                    'input_symbol': symbol,
                    'official_symbol': None,
                    'status': 'Not Found'
                })

        # Create a DataFrame from the results
        mapping_df = pd.DataFrame(results)

        # Save to CSV
        mapping_df.to_csv(output_file, index=False)
        return mapping_df

    except ImportError:
        print("pybiomart is not installed. Install it with 'pip install pybiomart'")
        return None

def map_using_local_data(old_symbols, hgnc_file=None):
    """
    Map gene symbols using a local HGNC complete set file

    Parameters:
    -----------
    old_symbols : list
        List of gene symbols to map
    hgnc_file : str
        Path to HGNC complete set file (will be downloaded if not provided)

    Returns:
    --------
    DataFrame with columns: input_symbol, official_symbol, status
    """
    # Download HGNC complete set if not provided
    if hgnc_file is None or not os.path.exists(hgnc_file):
        print("Downloading HGNC complete set...")
        url = "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
        hgnc_file = "hgnc_complete_set.txt"
        response = requests.get(url)
        with open(hgnc_file, 'wb') as f:
            f.write(response.content)

    # Load HGNC data
    hgnc_data = pd.read_csv(hgnc_file, sep='\t')

    # Create mapping dictionaries
    symbol_to_approved = {}

    # Map approved symbols to themselves
    for _, row in hgnc_data.iterrows():
        symbol = row.get('symbol')
        if pd.notna(symbol):
            symbol_to_approved[symbol.upper()] = symbol

    # Map previous symbols to approved symbols
    for _, row in hgnc_data.iterrows():
        approved = row.get('symbol')
        prev_symbols = row.get('prev_symbol')

        if pd.notna(prev_symbols) and pd.notna(approved):
            for prev in str(prev_symbols).split('|'):
                if prev.strip():
                    symbol_to_approved[prev.upper()] = approved

    # Map aliases to approved symbols
    for _, row in hgnc_data.iterrows():
        approved = row.get('symbol')
        aliases = row.get('alias_symbol')

        if pd.notna(aliases) and pd.notna(approved):
            for alias in str(aliases).split('|'):
                if alias.strip():
                    # Only add if not already in the dict or pointing to itself
                    if alias.upper() not in symbol_to_approved:
                        symbol_to_approved[alias.upper()] = approved

    # Map the old symbols to new ones
    mapped_genes = {
        'input_symbol': [],
        'official_symbol': [],
        'status': []
    }

    for symbol in old_symbols:
        mapped_genes['input_symbol'].append(symbol)

        if symbol.upper() in symbol_to_approved:
            mapped_genes['official_symbol'].append(symbol_to_approved[symbol.upper()])

            # Check if it's the same symbol or a mapping
            if symbol.upper() == symbol_to_approved[symbol.upper()].upper():
                mapped_genes['status'].append('Current')
            else:
                mapped_genes['status'].append('Updated')
        else:
            mapped_genes['official_symbol'].append(None)
            mapped_genes['status'].append('Not Found')

    # Create a DataFrame from the mapping dictionary
    mapping_df = pd.DataFrame(mapped_genes)
    return mapping_df

def create_bedfile_from_genes(gene_list, gene_ref, upstream=35000, downstream=10000):
    """
    Create BED file from a list of genes with upstream and downstream padding,
    accounting for strand direction

    Parameters:
    -----------
    gene_list : list
        List of gene IDs (e.g. ENSG IDs or gene symbols)
    gene_ref : pandas.DataFrame
        DataFrame with gene coordinates
        Must have columns: ensgid/gene_symbol, chr, start, end, strand
    upstream : int, optional
        Number of base pairs to extend upstream (default: 35000)
    downstream : int, optional
        Number of base pairs to extend downstream (default: 10000)

    Returns:
    --------
    pandas.DataFrame: DataFrame in BED format containing coordinates for input genes
    """
    # Filter gene reference data for genes in gene_list
    bed_data = gene_ref[gene_ref['GENE'].isin(gene_list)][['CHR', 'Start', 'End', 'GENE', 'Strand']]

    # Add upstream and downstream padding based on strand
    for idx, row in bed_data.iterrows():
        if row['Strand'] == '+':
            bed_data.at[idx, 'Start'] = row['Start'] - upstream
            bed_data.at[idx, 'End'] = row['End'] + downstream
        else:  # negative strand
            bed_data.at[idx, 'Start'] = row['Start'] - downstream
            bed_data.at[idx, 'End'] = row['End'] + upstream

    # Ensure start positions don't go below 0
    bed_data['Start'] = bed_data['Start'].clip(lower=0)

    bed_data['CHR'] = "chr" + bed_data['CHR'].astype(str)
    # Sort by chromosome and start position
    bed_data = bed_data.sort_values(['CHR', 'Start'])
    # remove "X" and "Y" from CHR
    bed_data = bed_data[bed_data['CHR'] != 'chrX']
    bed_data = bed_data[bed_data['CHR'] != 'chrY']

    return bed_data

def compute_burden(DF):
    Ncase = DF["Total Case"].values[0]
    Nctrl = DF["Total Ctrl"].values[0]
    LGD_case = DF["LGD Case"].sum()
    LGD_ctrl = DF["LGD Ctrl"].sum()
    Dmis_case = DF["Dmis Case"].sum()
    Dmis_ctrl = DF["Dmis Ctrl"].sum()
    # compute LGD odds ratio and p-value
    LGD_odds_ratio = (LGD_case/Ncase) / (LGD_ctrl/Nctrl)
    odds_ratio, LGD_p_value = scipy.stats.fisher_exact([[LGD_case, Ncase-LGD_case], [LGD_ctrl, Nctrl-LGD_ctrl]])
    # Calculate 95% CI
    alpha = 0.05
    se = np.sqrt(1/LGD_case + 1/(Ncase-LGD_case) + 1/LGD_ctrl + 1/(Nctrl-LGD_ctrl))
    ci_lower = np.exp(np.log(odds_ratio) - scipy.stats.norm.ppf(1-alpha/2) * se)
    ci_upper = np.exp(np.log(odds_ratio) + scipy.stats.norm.ppf(1-alpha/2) * se)
    return odds_ratio, LGD_p_value, ci_lower, ci_upper

def compute_burden_DNV(DF):
    Ncase = DF["Total Case"].values[0]
    Nctrl = DF["Total Ctrl"].values[0]
    LGD_case = DF["LGD Case"].sum()
    LGD_ctrl = DF["LGD Ctrl"].sum()
    Dmis_case = DF["Dmis Case"].sum()
    Dmis_ctrl = DF["Dmis Ctrl"].sum()

    # Compute rate ratio using Poisson model
    rate_ratio = (LGD_case/Ncase) / (LGD_ctrl/Nctrl)

    # Calculate standard error for log rate ratio
    se = np.sqrt(1/LGD_case + 1/LGD_ctrl)

    # Calculate 95% CI
    alpha = 0.05
    ci_lower = np.exp(np.log(rate_ratio) - scipy.stats.norm.ppf(1-alpha/2) * se)
    ci_upper = np.exp(np.log(rate_ratio) + scipy.stats.norm.ppf(1-alpha/2) * se)

    # Calculate p-value using Poisson test
    expected_case = (LGD_case + LGD_ctrl) * (Ncase/(Ncase + Nctrl))
    p_value = scipy.stats.poisson.sf(LGD_case-1, expected_case)

    return rate_ratio, p_value, ci_lower, ci_upper

def compute_regression(target_df, effect="BETA"):
    """
    Compute enrichment of beta coefficients between candidate and non-candidate genes
    using linear regression.

    Parameters:
    -----------
    test_df : pandas.DataFrame
        DataFrame containing 'in_candidates' and 'BETA' columns

    Returns:
    --------
    tuple: (odds_ratio, p_value, or_ci_lower, or_ci_upper)
        Regression results including odds ratio, p-value and confidence intervals
    """
    # Prepare data for regression
    X = target_df["in_candidates"].values.reshape(-1, 1)  # Independent variable
    y = target_df[effect].values  # Dependent variable

    # Fit regression
    reg = LinearRegression().fit(X, y)

    # Calculate standard error of coefficient
    n = len(X)
    y_pred = reg.predict(X)
    mse = np.sum((y - y_pred) ** 2) / (n - 2)
    std_err = np.sqrt(mse / np.sum((X - X.mean()) ** 2))

    # Calculate 95% CI for coefficient
    t_value = stats.t.ppf(0.975, n - 2)
    ci_lower = reg.coef_[0] - t_value * std_err
    ci_upper = reg.coef_[0] + t_value * std_err

    # Calculate odds ratio and CI
    odds_ratio = np.exp(reg.coef_[0])
    or_ci_lower = np.exp(ci_lower)
    or_ci_upper = np.exp(ci_upper)

    # Calculate p-value for odds ratio
    t_stat = reg.coef_[0] / std_err
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))

    return odds_ratio, p_value, or_ci_lower, or_ci_upper

def gene_set_enrichment_test(gene_zscores, gene_set, covariate=None):
    """
    Test if genes within a specified gene set have higher z-scores compared to genes outside the set.
    Uses linear regression to determine the effect size and p-value.

    Parameters:
    -----------
    gene_zscores : dict or pandas.Series
        A dictionary or Series where keys/indices are gene IDs and values are z-scores
        for a specific disorder or trait.

    gene_set : set or list
        A set or list of gene IDs that define the gene set of interest.

    covariate : dict or pandas.Series, optional
        Optional covariates for each gene (e.g., gene length, GC content) to include
        in the regression model. Should have the same keys/indices as gene_zscores.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'effect_size': The coefficient of the gene set indicator in the model
        - 'pvalue': The p-value for the gene set coefficient
        - 'model_summary': Full summary of the regression model
        - 'n_genes_in_set': Number of genes in the set that were found in the z-scores
        - 'mean_zscore_in_set': Mean z-score of genes in the set
        - 'mean_zscore_out_set': Mean z-score of genes outside the set
    """
    # Convert inputs to pandas Series if they are dictionaries
    if isinstance(gene_zscores, dict):
        gene_zscores = pd.Series(gene_zscores)

    # Create gene set indicator (1 if gene is in the set, 0 otherwise)
    gene_set = set(gene_set)  # Convert to set for faster lookup
    genes_in_data = set(gene_zscores.index)
    gene_set_overlap = gene_set.intersection(genes_in_data)

    if len(gene_set_overlap) == 0:
        raise ValueError("No genes from the provided gene set were found in the z-scores data")

    # Create indicator variable
    gene_set_indicator = pd.Series(
        [1 if gene in gene_set else 0 for gene in gene_zscores.index],
        index=gene_zscores.index
    )

    # Prepare data for regression
    X = pd.DataFrame({'gene_set': gene_set_indicator})

    # Add covariate if provided
    if covariate is not None:
        if isinstance(covariate, dict):
            covariate = pd.Series(covariate)

        # Ensure covariate has the same indices as gene_zscores
        covariate = covariate.loc[gene_zscores.index]
        X['covariate'] = covariate

    # Add constant term (intercept)
    X = add_constant(X)

    # Perform regression
    model = sm.OLS(gene_zscores, X)
    results = model.fit()

    # Calculate mean z-scores inside and outside the gene set
    mean_zscore_in_set = gene_zscores[gene_set_indicator == 1].mean()
    mean_zscore_out_set = gene_zscores[gene_set_indicator == 0].mean()

    # Return results
    return {
        'effect_size': results.params['gene_set'],
        'pvalue': results.pvalues['gene_set'],
        'model_summary': results.summary(),
        'n_genes_in_set': len(gene_set_overlap),
        'mean_zscore_in_set': mean_zscore_in_set,
        'mean_zscore_out_set': mean_zscore_out_set
    }

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
    df.index = df.index.astype(int)
    df = df.sort_values("EFFECT", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    
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

def MouseSTR_AvgZ_Weighted(ExpZscoreMat, Gene2Weights, csv_fil=None):
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
        df.to_csv(csv_fil)

    return df

def perform_pca_analysis(data, verbose=True):
    """
    Perform PCA analysis on input data and print explained variance and PC1 loadings

    Args:
        data: pandas DataFrame to analyze

    Returns:
        tuple: (pca object, pca results, loadings DataFrame)
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    if verbose:
        # Print explained variance ratios
        print("Explained variance ratios:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        print("\nCumulative variance explained:",
              f"{sum(pca.explained_variance_ratio_)*100:.1f}%")

    # Get component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=data.columns
    )

    if verbose:
        # Print PC1 loadings
        print("\nPC1 loadings:")
        print(loadings['PC1'].sort_values(ascending=False))

    return scaled_data, pca, pca_result, loadings


def load_pc_loadings(config, file_type, verbose=True):
    """
    Load PC loadings for all combinations of sources, modes, bias types, and exclude lists.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing gwas_sources, assoc_modes, bias_types, exclude_gene_list
    verbose : bool, default=True
        Whether to print loading messages
    
    Returns:
    --------
    dict
        Nested dictionary containing loaded PC loadings
    """
    # Configuration from the loaded config
    gwas_sources = config['gwas_sources']
    modes = config['assoc_modes'] 
    bias_types = config['bias_types']
    exclude_lists = config['exclude_gene_list']

    if verbose:
        print(f"Loading PC loadings for:")
        print(f"  Sources: {gwas_sources}")
        print(f"  Modes: {modes}")
        print(f"  Bias types: {bias_types}")
        print(f"  Exclude lists: {exclude_lists}")

    # Dictionary to store all PC loadings
    loadings_all = {}

    # Root directory for PCA results
    pca_root = "../results/pca"

    # Load all combinations
    for source in gwas_sources:
        loadings_all[source] = {}
        for mode in modes:
            loadings_all[source][mode] = {}
            for bias_type in bias_types:
                loadings_all[source][mode][bias_type] = {}
                for exclude in exclude_lists:
                    
                    # Construct file path
                    file_path = f"{pca_root}/{source}/{mode}/{bias_type}.{exclude}/{file_type}"
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path, sep=',', index_col=0)
                            loadings_all[source][mode][bias_type][exclude] = df
                            if verbose:
                                print(f"✓ Loaded {source}/{mode}/{bias_type}.{exclude}: {df.shape}")
                        except Exception as e:
                            if verbose:
                                print(f"✗ Error loading {file_path}: {e}")
                    else:
                        if verbose:
                            print(f"✗ File not found: {file_path}")
    
    return loadings_all


