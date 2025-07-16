
import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
from CellType_PSY import *
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
plt.style.use('seaborn-v0_8')


def LoadGeneINFO():
    #HGNC = pd.read_csv("../dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    HGNC = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    ENSID2Entrez = dict(zip(HGNC["ensembl_gene_id"].values, HGNC["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC["symbol"].values, HGNC["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC["entrez_id"].values, HGNC["symbol"].values))
    #allen_mouse_genes = loadgenelist("/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol 
    
# Create figure
def plot_HumanCT_boxplot(DF, Anno, ALL_CTs, label, title):
    plt.figure(figsize=(12,6), transparent=True)

    # Create list to store data for boxplot and calculate means for sorting
    box_data = []
    tick_labels = []
    means = []

    # Collect data and means for each cell type
    for CT in ALL_CTs:
        CT_idx = Anno[Anno["Supercluster"]==CT].index.values
        dat = DF.loc[CT_idx, label]
        box_data.append(dat)
        tick_labels.append(CT)
        means.append(dat.mean())

    # Sort everything by means
    sorted_indices = np.argsort(means)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]

    # Create boxplot    
    plt.boxplot(box_data, tick_labels=tick_labels, patch_artist=True)

    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(label)
    plt.title('Distribution of {} by Cell Type'.format(title))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def plot_correlation(values1, values2, name1, name2, title="", xlim=None):
    # Calculate correlations
    corr_all = stats.spearmanr(values1, values2)[0]
    corr_neur = stats.spearmanr(values1[Neur_idx], values2[Neur_idx])[0]
    corr_nonneur = stats.spearmanr(values1[NonNeur_idx], values2[NonNeur_idx])[0]
    
    # print(f"All cells correlation: {corr_all:.3f}")
    # print(f"Neuronal correlation: {corr_neur:.3f}")
    # print(f"Non-neuronal correlation: {corr_nonneur:.3f}")
    

    plt.figure(figsize=(6,6), dpi = 360, transparent=True)
    plt.scatter(values1[Neur_idx], values2[Neur_idx],
            color="red", alpha=0.6, s=80, label="Neuronal")
    plt.scatter(values1[NonNeur_idx], values2[NonNeur_idx], 
            color="blue", alpha=0.6, s=80, label="Non-neuronal")


                # Add text annotations on lower right
    plt.text(0.95, 0.05, 
             f'All: ρ = {corr_all:.3f}\n'
             f'Neuronal: ρ = {corr_neur:.3f}\n'
             f'Non-neuronal: ρ = {corr_nonneur:.3f}',
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='bottom',
             fontsize=15)

    plt.xlabel(name1, fontsize=22)
    
    plt.ylabel(name2, fontsize=22)
    if title == None:
        #title = "{} - {}".format(name1, name2)
        pass
    plt.title(f"{title}", fontsize=22)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(xlim)
    plt.legend(fontsize=15, loc="upper left")
    return corr_all

def plot_correlation_scatter_mouseCT(df1, df2, name1, name2, effect_col1="EFFECT", effect_col2="EFFECT", class_col="class_id_label"):
    """
    Plot correlation scatter plot between two dataframes' effect values, colored by neuronal vs non-neuronal.
    
    Args:
        df1: First pandas DataFrame containing effect values
        df2: Second pandas DataFrame containing effect values  
        effect_col: Column name containing effect values (default "EFFECT")
        class_col: Column name containing class labels (default "class_id_label")
    """
    # Get values for scatter plot and ensure matching indices
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    values1 = df1[effect_col1].values
    values2 = df2[effect_col2].values
    Non_NEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']

    # Create mask for Non-neuronal classes using matched indices
    if class_col in df1.columns:
        non_neur_mask = df1[class_col].isin(Non_NEUR)
    else:
        non_neur_mask = df2[class_col].isin(Non_NEUR)

    # Calculate correlations
    from scipy.stats import pearsonr, spearmanr

    # All cells correlation
    all_corr, _ = pearsonr(values1, values2)
    print(f"All cells correlation: {all_corr:.3f}")

    # Neuronal cells correlation 
    neur_corr, _ = pearsonr(values1[~non_neur_mask], values2[~non_neur_mask])
    print(f"Neuronal cells correlation: {neur_corr:.3f}")

    # Non-neuronal cells correlation
    non_neur_corr, _ = pearsonr(values1[non_neur_mask], values2[non_neur_mask])
    print(f"Non-neuronal cells correlation: {non_neur_corr:.3f}")

    # Plot scatter
    plt.figure(figsize=(6.5,6), dpi = 360, transparent=True)
    plt.scatter(values1[~non_neur_mask], values2[~non_neur_mask], 
               color="red", alpha=0.6, s=30, label="Neuronal")
    plt.scatter(values1[non_neur_mask], values2[non_neur_mask],
               color="blue", alpha=0.6, s=30, label="Non-neuronal")
    plt.xlabel(name1, fontsize=22)
    plt.ylabel(name2, fontsize=22)
    plt.legend(fontsize=15, loc="upper left")

    # Calculate and add Spearman correlation
    # Calculate Spearman correlations
    all_spearman, _ = spearmanr(values1, values2)
    neur_spearman, _ = spearmanr(values1[~non_neur_mask], values2[~non_neur_mask])
    non_neur_spearman, _ = spearmanr(values1[non_neur_mask], values2[non_neur_mask])

    # Add text annotations
    plt.text(0.95, 0.05,
             f'All: ρ = {all_spearman:.3f}\n'
             f'Neuronal: ρ = {neur_spearman:.3f}\n'
             f'Non-neuronal: ρ = {non_neur_spearman:.3f}',
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='bottom',
             fontsize=22)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()



import pandas as pd
import requests
import time
import re
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

# Genetic burnden related functions
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
        


def plot_disorder_burden_comparison_gwas(target_disorder, RareCoding_data, GWAS_data, top_n_values=[100, 500, 1000]):
    """
    Plot burden comparison of top genes from other disorders in target disorder.
    
    Args:
        target_disorder (str): Name of target disorder to analyze
        target_df (DataFrame): Case-control data for target disorder
        top_n_values (list): List of top N genes to analyze [default: [100,500,1000]]
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, len(top_n_values), figsize=(24,8), dpi=300, transparent=True)

    # Remove target disorder from comparison
    comparison_data = {k:v for k,v in RareCoding_data.items() if k != target_disorder}
    target_df = RareCoding_data[target_disorder]

    if target_disorder in ["ASD", "NDD"]:
        for i, TopN in enumerate(top_n_values):
            disorders = []
            odds_ratios = []
            ci_lowers = []
            ci_uppers = []
            pvalues = []

            for Disorder, DF in comparison_data.items():
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden_DNV(DF_filt)
                
                disorders.append(Disorder + " De Novo")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            for Disorder, DF in GWAS_data.items():
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden_DNV(DF_filt)
                
                disorders.append(Disorder + " GWAS")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            # Plot odds ratios and confidence intervals
            y_pos = np.arange(len(disorders))
            axes[i].errorbar(odds_ratios, y_pos, 
                            xerr=[np.array(odds_ratios)-np.array(ci_lowers),
                                np.array(ci_uppers)-np.array(odds_ratios)],
                            fmt='o', color='red', capsize=5)

            # Add disorder labels with larger font
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(disorders, fontsize=20)

            # Add p-values with larger font
            for j, (x, y, p) in enumerate(zip(odds_ratios, y_pos, pvalues)):
                axes[i].text(x, y-0.2, f'p={p:.2e}', ha='center', va='top', fontsize=12)

            # Add vertical line at x=1
            axes[i].axvline(x=1, color='gray', linestyle='--', alpha=0.5)

            # Customize plot with larger fonts
            axes[i].set_xlabel('Rate Ratio', fontsize=15)
            axes[i].set_title(f'{target_disorder} LGD Burden of Top {TopN} Genes from Other Disorders', fontsize=15)
            axes[i].margins(y=0.05)
            
            # Increase tick label font sizes
            axes[i].tick_params(axis='both', which='major', labelsize=12)

    elif target_disorder in ["MDD", "EDU", "RT", "VNR"]:
        for i, TopN in enumerate(top_n_values):
            disorders = []
            odds_ratios = []
            ci_lowers = []
            ci_uppers = []
            pvalues = []

            for Disorder, DF in comparison_data.items():
                if Disorder == "LOEUF":
                    disorders.append(Disorder + "Constraint")
                elif Disorder in ["ASD", "NDD"]:
                    disorders.append(Disorder + " De Novo")
                else:
                    disorders.append(Disorder + " Case Control")

                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                target_df['in_candidates'] = target_df['GeneSymbol'].isin(CandidateGenes)
                odds_ratio, pvalue, ci_lower, ci_upper = compute_regression(target_df)
                
                
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            for Disorder, DF in GWAS_data.items():
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                target_df['in_candidates'] = target_df['GeneSymbol'].isin(CandidateGenes)
                odds_ratio, pvalue, ci_lower, ci_upper = compute_regression(target_df)
                
                disorders.append(Disorder + " GWAS")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)


import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
from CellType_PSY import *
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
plt.style.use('seaborn-v0_8')


def LoadGeneINFO():
    #HGNC = pd.read_csv("../dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    HGNC = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    ENSID2Entrez = dict(zip(HGNC["ensembl_gene_id"].values, HGNC["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC["symbol"].values, HGNC["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC["entrez_id"].values, HGNC["symbol"].values))
    #allen_mouse_genes = loadgenelist("/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol 
    
# Create figure
def plot_HumanCT_boxplot(DF, Anno, ALL_CTs, label, title):
    plt.figure(figsize=(12,6))

    # Create list to store data for boxplot and calculate means for sorting
    box_data = []
    tick_labels = []
    means = []

    # Collect data and means for each cell type
    for CT in ALL_CTs:
        CT_idx = Anno[Anno["Supercluster"]==CT].index.values
        dat = DF.loc[CT_idx, label]
        box_data.append(dat)
        tick_labels.append(CT)
        means.append(dat.mean())

    # Sort everything by means
    sorted_indices = np.argsort(means)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]

    # Create boxplot    
    plt.boxplot(box_data, tick_labels=tick_labels, patch_artist=True)

    # Customize plot
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(label)
    plt.title('Distribution of {} by Cell Type'.format(title))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def plot_correlation(values1, values2, name1, name2, title="", xlim=None, dpi=360):
    # Calculate correlations
    corr_all = stats.spearmanr(values1, values2)[0]
    corr_neur = stats.spearmanr(values1[Neur_idx], values2[Neur_idx])[0]
    corr_nonneur = stats.spearmanr(values1[NonNeur_idx], values2[NonNeur_idx])[0]
    
    # print(f"All cells correlation: {corr_all:.3f}")
    # print(f"Neuronal correlation: {corr_neur:.3f}")
    # print(f"Non-neuronal correlation: {corr_nonneur:.3f}")
    

    plt.figure(figsize=(6,6), dpi = dpi)
    plt.scatter(values1[Neur_idx], values2[Neur_idx],
            color="red", alpha=0.6, s=80, label="Neuronal")
    plt.scatter(values1[NonNeur_idx], values2[NonNeur_idx], 
            color="blue", alpha=0.6, s=80, label="Non-neuronal")


                # Add text annotations on lower right
    plt.text(0.95, 0.05, 
             f'All: ρ = {corr_all:.3f}\n'
             f'Neuronal: ρ = {corr_neur:.3f}\n'
             f'Non-neuronal: ρ = {corr_nonneur:.3f}',
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='bottom',
             fontsize=15)

    plt.xlabel(name1, fontsize=22)
    
    plt.ylabel(name2, fontsize=22)
    if title == None:
        #title = "{} - {}".format(name1, name2)
        pass
    plt.title(f"{title}", fontsize=22)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(xlim)
    plt.legend(fontsize=15, loc="upper left")
    return corr_all

def plot_correlation_scatter_mouseCT(df1, df2, name1, name2, effect_col1="EFFECT", effect_col2="EFFECT", class_col="class_id_label"):
    """
    Plot correlation scatter plot between two dataframes' effect values, colored by neuronal vs non-neuronal.
    
    Args:
        df1: First pandas DataFrame containing effect values
        df2: Second pandas DataFrame containing effect values  
        effect_col: Column name containing effect values (default "EFFECT")
        class_col: Column name containing class labels (default "class_id_label")
    """
    # Get values for scatter plot and ensure matching indices
    df1 = df1.sort_index()
    df2 = df2.sort_index()

    values1 = df1[effect_col1].values
    values2 = df2[effect_col2].values
    Non_NEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']

    # Create mask for Non-neuronal classes using matched indices
    if class_col in df1.columns:
        non_neur_mask = df1[class_col].isin(Non_NEUR)
    else:
        non_neur_mask = df2[class_col].isin(Non_NEUR)

    # Calculate correlations
    from scipy.stats import pearsonr, spearmanr

    # All cells correlation
    all_corr, _ = pearsonr(values1, values2)
    print(f"All cells correlation: {all_corr:.3f}")

    # Neuronal cells correlation 
    neur_corr, _ = pearsonr(values1[~non_neur_mask], values2[~non_neur_mask])
    print(f"Neuronal cells correlation: {neur_corr:.3f}")

    # Non-neuronal cells correlation
    non_neur_corr, _ = pearsonr(values1[non_neur_mask], values2[non_neur_mask])
    print(f"Non-neuronal cells correlation: {non_neur_corr:.3f}")

    # Plot scatter
    plt.figure(figsize=(6.5,6), dpi = 360)
    plt.scatter(values1[~non_neur_mask], values2[~non_neur_mask], 
               color="red", alpha=0.6, s=30, label="Neuronal")
    plt.scatter(values1[non_neur_mask], values2[non_neur_mask],
               color="blue", alpha=0.6, s=30, label="Non-neuronal")
    plt.xlabel(name1, fontsize=22)
    plt.ylabel(name2, fontsize=22)
    plt.legend(fontsize=15, loc="upper left")

    # Calculate and add Spearman correlation
    # Calculate Spearman correlations
    all_spearman, _ = spearmanr(values1, values2)
    neur_spearman, _ = spearmanr(values1[~non_neur_mask], values2[~non_neur_mask])
    non_neur_spearman, _ = spearmanr(values1[non_neur_mask], values2[non_neur_mask])

    # Add text annotations
    plt.text(0.95, 0.05,
             f'All: ρ = {all_spearman:.3f}\n'
             f'Neuronal: ρ = {neur_spearman:.3f}\n'
             f'Non-neuronal: ρ = {non_neur_spearman:.3f}',
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='bottom',
             fontsize=22)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()



import pandas as pd
import requests
import time
import re
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

# Genetic burnden related functions
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
        


def plot_disorder_burden_comparison_gwas(target_disorder, RareCoding_data, GWAS_data, top_n_values=[100, 500, 1000]):
    """
    Plot burden comparison of top genes from other disorders in target disorder.
    
    Args:
        target_disorder (str): Name of target disorder to analyze
        target_df (DataFrame): Case-control data for target disorder
        top_n_values (list): List of top N genes to analyze [default: [100,500,1000]]
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, len(top_n_values), figsize=(24,8), dpi=300)

    # Remove target disorder from comparison
    comparison_data = {k:v for k,v in RareCoding_data.items() if k != target_disorder}
    target_df = RareCoding_data[target_disorder]

    if target_disorder in ["ASD", "NDD"]:
        for i, TopN in enumerate(top_n_values):
            disorders = []
            odds_ratios = []
            ci_lowers = []
            ci_uppers = []
            pvalues = []

            for Disorder, DF in comparison_data.items():
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden_DNV(DF_filt)
                
                disorders.append(Disorder + " De Novo")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            for Disorder, DF in GWAS_data.items():
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden_DNV(DF_filt)
                
                disorders.append(Disorder + " GWAS")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            # Plot odds ratios and confidence intervals
            y_pos = np.arange(len(disorders))
            axes[i].errorbar(odds_ratios, y_pos, 
                            xerr=[np.array(odds_ratios)-np.array(ci_lowers),
                                np.array(ci_uppers)-np.array(odds_ratios)],
                            fmt='o', color='red', capsize=5)

            # Add disorder labels with larger font
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(disorders, fontsize=20)

            # Add p-values with larger font
            for j, (x, y, p) in enumerate(zip(odds_ratios, y_pos, pvalues)):
                axes[i].text(x, y-0.2, f'p={p:.2e}', ha='center', va='top', fontsize=12)

            # Add vertical line at x=1
            axes[i].axvline(x=1, color='gray', linestyle='--', alpha=0.5)

            # Customize plot with larger fonts
            axes[i].set_xlabel('Rate Ratio', fontsize=15)
            axes[i].set_title(f'{target_disorder} LGD Burden of Top {TopN} Genes from Other Disorders', fontsize=15)
            axes[i].margins(y=0.05)
            
            # Increase tick label font sizes
            axes[i].tick_params(axis='both', which='major', labelsize=12)

    elif target_disorder in ["MDD", "EDU", "RT", "VNR"]:
        for i, TopN in enumerate(top_n_values):
            disorders = []
            odds_ratios = []
            ci_lowers = []
            ci_uppers = []
            pvalues = []

            for Disorder, DF in comparison_data.items():
                if Disorder == "LOEUF":
                    disorders.append(Disorder + "Constraint")
                elif Disorder in ["ASD", "NDD"]:
                    disorders.append(Disorder + " De Novo")
                else:
                    disorders.append(Disorder + " Case Control")

                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                target_df['in_candidates'] = target_df['GeneSymbol'].isin(CandidateGenes)
                odds_ratio, pvalue, ci_lower, ci_upper = compute_regression(target_df)
                
                
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            for Disorder, DF in GWAS_data.items():
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                target_df['in_candidates'] = target_df['GeneSymbol'].isin(CandidateGenes)
                odds_ratio, pvalue, ci_lower, ci_upper = compute_regression(target_df)
                
                disorders.append(Disorder + " GWAS")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            # Plot odds ratios and confidence intervals
            y_pos = np.arange(len(disorders))
            axes[i].errorbar(odds_ratios, y_pos, 
                            xerr=[np.array(odds_ratios)-np.array(ci_lowers),
                                np.array(ci_uppers)-np.array(odds_ratios)],
                            fmt='o', color='red', capsize=5)

            # Add disorder labels with larger font
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(disorders, fontsize=20)

            # Add p-values with larger font
            for j, (x, y, p) in enumerate(zip(odds_ratios, y_pos, pvalues)):
                axes[i].text(x, y-0.2, f'p={p:.2e}', ha='center', va='top', fontsize=12)

            # Add vertical line at x=1
            axes[i].axvline(x=1, color='gray', linestyle='--', alpha=0.5)

            # Customize plot with larger fonts
            axes[i].set_xlabel('Odds Ratio', fontsize=15)
            axes[i].set_title(f'{target_disorder} Beta OR of Top {TopN} Genes from Other Disorders', fontsize=15)
            axes[i].margins(y=0.05)
            
            # Increase tick label font sizes
            axes[i].tick_params(axis='both', which='major', labelsize=12)

    else:
        for i, TopN in enumerate(top_n_values):
            disorders = []
            odds_ratios = []
            ci_lowers = []
            ci_uppers = []
            pvalues = []

            for Disorder, DF in comparison_data.items():
                if Disorder == "LOEUF":
                    disorders.append(Disorder)
                elif Disorder in ["ASD", "NDD"]:
                    disorders.append(Disorder + " De Novo")
                else:
                    disorders.append(Disorder + " Case Control")
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden(DF_filt)
                
                #disorders.append(Disorder + " Rare Coding")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            for Disorder, DF in GWAS_data.items():
                CandidateGenes = DF.head(TopN)["GeneSymbol"].values
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden(DF_filt)
                
                disorders.append(Disorder + " GWAS")
                odds_ratios.append(odds_ratio)
                ci_lowers.append(ci_lower) 
                ci_uppers.append(ci_upper)
                pvalues.append(pvalue)

            # Plot odds ratios and confidence intervals
            y_pos = np.arange(len(disorders))
            axes[i].errorbar(odds_ratios, y_pos, 
                            xerr=[np.array(odds_ratios)-np.array(ci_lowers),
                                np.array(ci_uppers)-np.array(odds_ratios)],
                            fmt='o', color='red', capsize=5)

            # Add disorder labels with larger font
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(disorders, fontsize=20)

            # Add p-values with larger font
            for j, (x, y, p) in enumerate(zip(odds_ratios, y_pos, pvalues)):
                axes[i].text(x, y-0.2, f'p={p:.2e}', ha='center', va='top', fontsize=12)

            # Add vertical line at x=1
            axes[i].axvline(x=1, color='gray', linestyle='--', alpha=0.5)

            # Customize plot with larger fonts
            axes[i].set_xlabel('Odds Ratio', fontsize=15)
            axes[i].set_title(f'{target_disorder} LGD Burden of Top {TopN} Genes from Other Disorders', fontsize=15)
            axes[i].margins(y=0.05)
            
            # Increase tick label font sizes
            axes[i].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()

def plot_disorder_burden_comparison_all(target_disorder, RareCoding_data, GWAS_data, top_n_values=[100, 500, 1000]):
    """
    Plot burden comparison of top genes from other disorders in target disorder.
    
    Args:
        target_disorder (str): Name of target disorder to analyze
        RareCoding_data (dict): Dictionary of rare coding data for each disorder
        GWAS_data (dict): Dictionary of GWAS data for each disorder 
        top_n_values (list): List of top N genes to analyze [default: [100,500,1000]]
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(top_n_values), figsize=(24,8), dpi=300)

    # Remove target disorder from comparison
    comparison_data = {k:v for k,v in RareCoding_data.items() if k != target_disorder}
    target_df = RareCoding_data[target_disorder]

    # Process each top N value
    for i, TopN in enumerate(top_n_values):
        disorders = []
        odds_ratios = []
        ci_lowers = []
        ci_uppers = []
        pvalues = []

        # Process rare coding data
        for Disorder, DF in comparison_data.items():
            CandidateGenes = DF.head(TopN)["GeneSymbol"].values
            
            # Handle different disorder types
            if target_disorder in ["ASD", "NDD"]:
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden_DNV(DF_filt)
                disorder_label = Disorder + " De Novo"
            
            elif target_disorder in ["MDD", "EDU", "RT", "VNR"]:
                target_df['in_candidates'] = target_df['GeneSymbol'].isin(CandidateGenes)
                odds_ratio, pvalue, ci_lower, ci_upper = compute_regression(target_df)
                if Disorder == "LOEUF":
                    disorder_label = Disorder + " Constraint"
                elif Disorder in ["ASD", "NDD"]:
                    disorder_label = Disorder + " De Novo"
                else:
                    disorder_label = Disorder + " Case Control"
            
            else:
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden(DF_filt)
                if Disorder == "LOEUF":
                    disorder_label = Disorder
                elif Disorder in ["ASD", "NDD"]:
                    disorder_label = Disorder + " De Novo"
                else:
                    disorder_label = Disorder + " Case Control"

            disorders.append(disorder_label)
            odds_ratios.append(odds_ratio)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            pvalues.append(pvalue)

        # Process GWAS data
        for Disorder, DF in GWAS_data.items():
            CandidateGenes = DF.head(TopN)["GeneSymbol"].values
            
            if target_disorder in ["ASD", "NDD"]:
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden_DNV(DF_filt)
            elif target_disorder in ["MDD", "EDU", "RT", "VNR"]:
                target_df['in_candidates'] = target_df['GeneSymbol'].isin(CandidateGenes)
                odds_ratio, pvalue, ci_lower, ci_upper = compute_regression(target_df)
            else:
                DF_filt = target_df[target_df["GeneSymbol"].isin(CandidateGenes)]
                odds_ratio, pvalue, ci_lower, ci_upper = compute_burden(DF_filt)

            disorders.append(Disorder + " GWAS")
            odds_ratios.append(odds_ratio)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
            pvalues.append(pvalue)

        # Plot results
        y_pos = np.arange(len(disorders))
        axes[i].errorbar(odds_ratios, y_pos,
                        xerr=[np.array(odds_ratios)-np.array(ci_lowers),
                            np.array(ci_uppers)-np.array(odds_ratios)],
                        fmt='o', color='red', capsize=5)

        # Customize plot
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(disorders, fontsize=20)
        
        for j, (x, y, p) in enumerate(zip(odds_ratios, y_pos, pvalues)):
            axes[i].text(x, y-0.2, f'p={p:.2e}', ha='center', va='top', fontsize=12)

        axes[i].axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        
        xlabel = 'Rate Ratio' if target_disorder in ["ASD", "NDD"] else 'Odds Ratio'
        title_prefix = 'Beta OR' if target_disorder in ["MDD", "EDU", "RT", "VNR"] else 'LGD Burden'
        
        axes[i].set_xlabel(xlabel, fontsize=15)
        axes[i].set_title(f'{target_disorder} {title_prefix} of Top {TopN} Genes from Other Disorders', fontsize=15)
        axes[i].margins(y=0.05)
        axes[i].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()


# Cell Type Bias related functions 


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

def plot_beta_distribution_by_supercluster(results_df, EFFECT="beta"):
    """
    Create a boxplot showing the distribution of beta values across different superclusters.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the results with columns:
        - Supercluster: The cell type supercluster
        - beta: The beta coefficient values
    
    Returns:
    --------
    None
        Displays the plot using matplotlib
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='Supercluster', y=EFFECT, 
                order=results_df.groupby('Supercluster')[EFFECT].median().sort_values(ascending=False).index)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Supercluster')
    plt.ylabel(EFFECT)
    plt.title('Distribution of {} by Supercluster'.format(EFFECT))
    plt.tight_layout()
    plt.show()

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
    df = pd.DataFrame(data={"CellType": CellTypes, "EFFECT": EFFECTS})
    df = df.sort_values("EFFECT", ascending=False)
    df = df.reset_index(drop=True)
    df["Rank"] = df.index + 1
    df = df.set_index("CellType")
    
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
    })
    
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

def perform_pca_analysis(data):
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
    
    # Print PC1 loadings
    print("\nPC1 loadings:")
    print(loadings['PC1'].sort_values(ascending=False))
    
    return scaled_data, pca, pca_result, loadings

def plot_pc_loadings(pca, spearman_df_sub, topLoadings=10):
    """
    Create a visualization of component loadings for all PCs.
    
    Args:
        pca: PCA object after fitting
        spearman_df_sub: Original dataframe used for PCA
        
    Returns:
        matplotlib figure
    """
    # Visualize component loadings for all PCs
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), dpi=240)
    axes = axes.ravel()

    # Define colors for positive and negative loadings
    colors = ['#FF6B6B', '#4ECDC4']  # Red for negative, teal for positive
    PCs = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11"]
    
    for i, pc in enumerate(PCs):
        if i >= len(pca.components_):
            fig.delaxes(axes[i])
            continue
            
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(pca.components_))],
            index=spearman_df_sub.columns
        )
        
        # Sort loadings
        sorted_loadings = loadings[pc].sort_values(ascending=True)
        sorted_loadings = sorted_loadings.tail(topLoadings)
        # Create horizontal bar plot with colors based on value
        bars = axes[i].barh(range(len(sorted_loadings)), sorted_loadings,
                           color=[colors[0] if x < 0 else colors[1] for x in sorted_loadings])
        
        # Customize plot appearance
        axes[i].set_yticks(range(len(sorted_loadings)))
        axes[i].set_yticklabels(sorted_loadings.index, fontsize=15)  # Increased from 10
        axes[i].set_title(f'{pc} Loadings', fontsize=18, pad=10)  # Increased from 12
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            label_pos = width + 0.01 if width >= 0 else width - 0.01
            ha = 'left' if width >= 0 else 'right'
            axes[i].text(label_pos, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', va='center', ha=ha, fontsize=13.5)  # Increased from 9
        
        # Set x-axis limits with some padding
        max_abs = max(abs(sorted_loadings.min()), abs(sorted_loadings.max()))
        axes[i].set_xlim(-max_abs*1.2, max_abs*1.2)
        
    plt.suptitle('Principal Component Loadings Analysis', fontsize=21, y=1.02)  # Increased from 14
    plt.tight_layout()
    plt.show()
    return 

def plot_scree_and_get_loadings(pca):
    """
    Creates a scree plot showing cumulative explained variance ratio and calculates PCA loadings.
    
    Args:
        pca: Fitted PCA object
        
    Returns:
        fig: matplotlib figure object
        loadings: DataFrame of PCA loadings
    """
    # Create scree plot
    fig = plt.figure(figsize=(10, 6), dpi=300)  # Increased DPI for publication quality
    plt.plot(range(0, len(pca.explained_variance_ratio_) + 1), 
             np.concatenate(([0], np.cumsum(pca.explained_variance_ratio_))), 
             'o-', color='#2E86C1', linewidth=2, markersize=8)  # Better color and line style

    plt.xlabel('Number of Components', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Explained\nVariance Ratio', fontsize=14, fontweight='bold')
    #plt.title('Scree Plot', fontsize=16, fontweight='bold', pad=15)

    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add text annotations for cumulative variance with better formatting
    for i, cum_var in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        plt.text(i+1.1, cum_var-0.01, f'+{pca.explained_variance_ratio_[i]:.3f}', 
                 verticalalignment='center',
                 fontsize=11,
                 color='#34495E')

    # Add spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Calculate variable contributions 
    # loadings = pd.DataFrame(
    #     pca.components_.T * np.sqrt(pca.explained_variance_),
    #     columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    #     index=spearman_df_sub.columns
    # )
    plt.show()
    return 

# Create figure
def plot_pc1_boxplot(pc_scores_df, Anno, ALL_CTs, PC, ylabel="PC1 Score"):
    """
    Create a publication-quality boxplot showing PC score distributions across cell types.
    
    Args:
        pc_scores_df: DataFrame containing PC scores
        Anno: DataFrame containing cell type annotations
        ALL_CTs: List of all cell types to plot
        PC: Name of PC column to plot
    """
    # Create high quality figure
    plt.figure(figsize=(12,6), dpi=300)

    # Create list to store data for boxplot and calculate means for sorting
    box_data = []
    tick_labels = []
    means = []

    # Define color palette
    palette = {'Neuron': 'red', 'Non-neuron': 'blue'}

    # Collect data and means for each cell type
    for CT in ALL_CTs:
        CT_idx = Anno[Anno["Supercluster"]==CT].index.values
        dat = pc_scores_df.loc[CT_idx, PC]
        box_data.append(dat)
        tick_labels.append(CT)
        means.append(dat.mean())

    # Sort everything by means
    sorted_indices = np.argsort(means)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]

    # Create boxplot with custom styling
    bp = plt.boxplot(box_data, tick_labels=tick_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                  markersize=4, alpha=0.5))
    
    # Color boxes based on neuronal vs non-neuronal using palette
    for i, (patch, label) in enumerate(zip(bp['boxes'], tick_labels)):
        if label in Neurons:
            patch.set_facecolor(palette['Neuron'])
        else:
            patch.set_facecolor(palette['Non-neuron'])
        patch.set_alpha(0.7)

    # Customize plot
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('{}'.format(ylabel), fontsize=14, fontweight='bold')   
    #plt.title('Distribution of {} Scores by Human Cell Type'.format(PC), 
    #         fontsize=16, fontweight='bold', pad=15)
    plt.title("Human Cell Type", fontsize=16)
    
    # Add grid and spines
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Create custom legend patches
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=palette['Neuron'], alpha=0.7, label='Neuron'),
                      plt.Rectangle((0,0),1,1, facecolor=palette['Non-neuron'], alpha=0.7, label='Non-neuron')]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=15)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def plot_pc1_boxplot_mouseCT(pc_scores_df, Anno, ALL_CTs, PC, ylabel="PC1 Score"):
    plt.figure(figsize=(12,6), dpi=300)

    # Create list to store data for boxplot and calculate means for sorting
    box_data = []
    tick_labels = []
    means = []

    # Define color palette
    palette = {'Neuron': 'red', 'Non-neuron': 'blue'}

    # Collect data and means for each cell type
    for CT in ALL_CTs:
        CT_idx = Anno[Anno["class_id_label"]==CT].index.values
        valid_CT_idx = [idx for idx in CT_idx if idx in pc_scores_df.index]
        dat = pc_scores_df.loc[valid_CT_idx, PC]
        box_data.append(dat)
        tick_labels.append(CT)
        means.append(dat.mean())

    # Sort everything by means
    sorted_indices = np.argsort(means)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]

    # Create boxplot with custom styling
    bp = plt.boxplot(box_data, tick_labels=tick_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                  markersize=4, alpha=0.5))

    # Color boxes based on neuronal vs non-neuronal using palette
    Non_NEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']
    for i, (patch, label) in enumerate(zip(bp['boxes'], tick_labels)):
        if label not in Non_NEUR:
            patch.set_facecolor(palette['Neuron'])
        else:
            patch.set_facecolor(palette['Non-neuron'])
        patch.set_alpha(0.7)

    # Customize plot
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('{}'.format(ylabel), fontsize=14, fontweight='bold')
    plt.title("Mouse Cell Type", fontsize=16)

    # Add grid and spines
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Create custom legend patches
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=palette['Neuron'], alpha=0.7, label='Neuron'),
                      plt.Rectangle((0,0),1,1, facecolor=palette['Non-neuron'], alpha=0.7, label='Non-neuron')]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=15)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def plot_pc1_boxplot_structure(pc_scores_df, Regions, REG2STR, PC):
    # Set style for publication quality
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(6,7), dpi=300)

    # Create list to store data for boxplot and calculate means for sorting
    box_data = []
    tick_labels = []
    means = []

    # Collect data and means for each cell type
    for Reg in Regions:
        print(Reg)
        STRs = REG2STR[Reg]
        valid_STRs = [str for str in STRs if str in pc_scores_df.index]
        dat = pc_scores_df.loc[valid_STRs, PC]
        box_data.append(dat)
        tick_labels.append(Reg)
        means.append(dat.mean())

    # Sort everything by means
    sorted_indices = np.argsort(means)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]

    # Create boxplot with custom style
    bp = ax.boxplot(box_data, vert=False, patch_artist=True, 
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(facecolor='lightblue', color='black', linewidth=1.5, alpha=0.6),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=5,
                                  linestyle='none', markeredgecolor='gray', alpha=0.5))

    # Add individual points with better jittering
    for i, data in enumerate(box_data):
        # Add jitter to y-coordinates with controlled randomness
        y = np.random.normal(i + 1, 0.06, size=len(data))
        ax.scatter(data, y, alpha=0.4, s=25, c='navy', zorder=2)

    # Customize plot with enhanced styling
    ax.set_yticklabels(tick_labels, rotation=0, ha='right', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{PC} Score', fontsize=14, fontweight='bold', labelpad=10)
    #ax.set_title(f'Distribution of {PC} Scores\nby Brain Region', 
    #             fontsize=16, fontweight='bold', pad=20)
    ax.set_title("Brain Structures (Mouse ISH)", fontsize=16)
    
    # Add subtle grid only for x-axis with enhanced styling
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken remaining spines and style them
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Adjust tick parameters for better visibility
    ax.tick_params(axis='both', width=2, length=6, labelsize=11)
    ax.tick_params(axis='x', labelsize=11)

    # Add subtle background color
    ax.set_facecolor('#f8f9fa')
    
    # Adjust layout to prevent label cutoff with more padding
    plt.tight_layout(pad=1.5)
    plt.show()
    
def plot_disorder_correlation_heatmap(df, title="", cluster=True):
    """
    Create a correlation heatmap showing relationships between psychiatric disorders.
    
    Args:
        df (pd.DataFrame): DataFrame containing disorder data to correlate
        title (str): Title for the plot
        cluster (bool): Whether to perform hierarchical clustering on correlation matrix
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    correlation_matrix_CT_Bias = df.corr()

    # Create publication quality figure 
    plt.figure(figsize=(7,6), dpi=200)

    if cluster:
        # Perform hierarchical clustering
        linkage = hierarchy.linkage(correlation_matrix_CT_Bias, method='ward')
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        idx = dendro['leaves']
        
        # Reorder correlation matrix
        correlation_matrix_CT_Bias = correlation_matrix_CT_Bias.iloc[idx, idx]

    # Create heatmap with customized appearance
    sns.heatmap(correlation_matrix_CT_Bias, 
                annot=True,
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                annot_kws={'size': 12},
                square=True)
                #cbar_kws={'label': 'Correlation Coefficient', 'labelsize': 18},

    # Customize font sizes and appearance  
    plt.title('{}'.format(title),
              fontsize=15,
              pad=20, 
              fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(rotation=0, fontsize=15)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
    return correlation_matrix_CT_Bias

def plot_effect_vs_pfactor_mouseSTR(mouse_str_DF1, mouse_str_DF2, name1, name2, label1="EFFECT", label2="EFFECT"):
    # Sort both dataframes by index to ensure alignment
    mouse_str_DF1 = mouse_str_DF1.sort_index()
    mouse_str_DF2 = mouse_str_DF2.sort_index()
    
    # Create publication quality figure
    plt.figure(figsize=(6.5, 6), dpi=360)
    plt.style.use('seaborn-v0_8')

    # Plot EFFECT vs P-factor using input dataframes with improved styling
    plt.scatter(mouse_str_DF1[label1], mouse_str_DF2[label2], 
               alpha=0.7, s=50, c='red', edgecolor='white', linewidth=0.5)
    
    # Improve axis labels with larger font
    plt.xlabel(f'{name1}', fontsize=20)
    plt.ylabel(f'{name2}', fontsize=20)
    #plt.title(f'{name1} vs {name2}', fontsize=15, pad=20, fontweight='bold')
    
    # Make tick labels larger
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Calculate correlation on sorted data
    corr = scipy.stats.pearsonr(mouse_str_DF1[label1], mouse_str_DF2[label2])
    
    # Add correlation text with improved formatting and position
    plt.text(0.70, 0.1, f'r = {corr[0]:.2f}\np = {corr[1]:.1e}',
            transform=plt.gca().transAxes,
            fontsize=20)
    
    # Adjust layout to prevent label cutoff
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return 
