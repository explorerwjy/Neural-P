import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse

def combine_bias_files(input_dir):
    """
    Combine bias files from multiple disorders into a single dataframe.
    
    Args:
        input_dir (str): Directory containing subdirectories of bias files
        
    Returns:
        pd.DataFrame: Combined dataframe with beta values from all disorders
    """
    # Dictionary to store dataframes for each disorder
    disorder_dfs = {}
    
    # Find all CSV files in directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    
    for csv_file in csv_files:
        # Load the dataframe
        print(csv_file)
        if "Meta" in csv_file:
            continue
        #df = pd.read_csv(os.path.join(input_dir, csv_file), index_col="ct_idx")
        try:
            df = pd.read_csv(os.path.join(input_dir, csv_file), index_col="ct_idx")
        except:
            df = pd.read_csv(os.path.join(input_dir, csv_file), index_col="Structure")
        
        # Extract disorder name from filename (assuming format like "ASD_bias.csv")
        disorder = csv_file.split('.')[1]
        print(disorder)
        
        # Store beta values in dictionary
        if 'beta' in df.columns:
            disorder_dfs[disorder] = df['beta']
        elif 'EFFECT' in df.columns:
            disorder_dfs[disorder] = df['EFFECT']
        elif 'Slope' in df.columns:
            disorder_dfs[disorder] = df['Slope']

    # Combine all dataframes
    combined_df = pd.DataFrame(disorder_dfs)
    #combined_df = combined_df.T
    return combined_df

def perform_pca_analysis(data):
    """
    Perform PCA analysis on input data and return results.
    
    Args:
        data: pandas DataFrame to analyze
        
    Returns:
        tuple: (scaled_data, pca, pca_result, loadings)
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    # Get component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=data.columns
    )
    
    return scaled_data, pca, pca_result, loadings

def plot_scree_and_get_loadings(pca):
    """
    Creates a scree plot showing cumulative explained variance ratio.
    
    Args:
        pca: Fitted PCA object
    """
    # Create scree plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(0, len(pca.explained_variance_ratio_) + 1), 
             np.concatenate(([0], np.cumsum(pca.explained_variance_ratio_))), 
             'o-', color='#2E86C1', linewidth=2, markersize=8)

    plt.xlabel('Number of Components', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Explained\nVariance Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add text annotations for cumulative variance
    for i, cum_var in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        plt.text(i+1.1, cum_var-0.01, f'+{pca.explained_variance_ratio_[i]:.3f}', 
                 verticalalignment='center',
                 fontsize=11,
                 color='#34495E')

    # Add spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

def analyze_bias_pca(input_dir, output_dir=None):
    """
    Combine bias files from multiple disorders, perform PCA analysis, and save results.
    
    Args:
        input_dir (str): Directory containing subdirectories of bias files
        output_dir (str, optional): Directory to save results. If None, uses input_dir.
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine bias files
    print("Combining bias files...")
    combined_df = combine_bias_files(input_dir)
    
    # Save combined dataframe
    combined_file = os.path.join(output_dir, "combined_bias_data.csv")
    combined_df.to_csv(combined_file)
    print(f"Saved combined bias data to {combined_file}")
    
    # Perform PCA analysis
    print("\nPerforming PCA analysis...")
    scaled_data, pca, pca_result, loadings = perform_pca_analysis(combined_df)
    
    # Create dataframe with PC scores
    pc_scores = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
        index=combined_df.index
    )
    
    # Save PC scores
    pc_scores_file = os.path.join(output_dir, "pc_scores.csv")
    pc_scores.to_csv(pc_scores_file)
    print(f"Saved PC scores to {pc_scores_file}")
    
    # Save loadings
    loadings_file = os.path.join(output_dir, "loadings.csv")
    loadings.to_csv(loadings_file)
    print(f"Saved loadings to {loadings_file}")
    
    # Save explained variance ratios
    explained_variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
    })
    explained_variance_file = os.path.join(output_dir, "explained_variance_ratio.csv")
    explained_variance_df.to_csv(explained_variance_file, index=False)
    print(f"Saved explained variance ratios to {explained_variance_file}")
    
    # Create and save scree plot
    plot_scree_and_get_loadings(pca)
    plt.savefig(os.path.join(output_dir, "scree_plot.png"))
    plt.close()
    
    # Print explained variance
    print("\nExplained variance ratios:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Combine bias files and perform PCA analysis')
    parser.add_argument('input_dir', help='Directory containing subdirectories of bias files')
    parser.add_argument('--output_dir', help='Directory to save results (default: same as input_dir)')
    
    args = parser.parse_args()
    analyze_bias_pca(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 

   # python src/analyze_bias_pca.py /path/to/input/directory --output_dir /path/to/output/directory