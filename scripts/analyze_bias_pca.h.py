import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
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
        try:
            df = pd.read_csv(os.path.join(input_dir, csv_file), index_col="ct_idx")
        except:
            df = pd.read_csv(os.path.join(input_dir, csv_file), index_col="Structure")
        
        # Extract disorder name from filename
        disorder = csv_file.split('.')[2]
        print(disorder)
        
        # Store beta values in dictionary
        if 'beta' in df.columns:
            disorder_dfs[disorder] = df['beta']

    # Combine all dataframes
    combined_df = pd.DataFrame(disorder_dfs)
    combined_df = combined_df.T
    return combined_df

def plot_dendrogram(Z, names, figsize=(15, 10), title="Cell Type Hierarchy Dendrogram"):
    """
    Create a dendrogram visualization of the hierarchical clustering.
    
    Args:
        Z: Linkage matrix
        names: Names of cell types
        figsize: Figure size
        title: Plot title
    """
    plt.figure(figsize=figsize, dpi=300)
    plt.title(title, fontsize=16, fontweight='bold')
    dendrogram(
        Z,
        labels=names,
        orientation='right',
        leaf_font_size=8,
        color_threshold=0.7*max(Z[:,2])
    )
    plt.tight_layout()
    return plt.gcf()

def perform_hierarchical_clustering(data, n_clusters=None, distance_threshold=None):
    """
    Perform hierarchical clustering on cell types based on their disorder profiles.
    
    Args:
        data: DataFrame with cell types as rows and disorders as columns
        n_clusters: Number of clusters to form
        distance_threshold: Distance threshold for forming clusters
        
    Returns:
        tuple: (linkage_matrix, cluster_labels, cluster_representatives)
    """
    # Transpose to get cell types as rows
    if data.shape[0] > data.shape[1]:  # If cell types are already rows
        data_for_clustering = data
    else:  # If disorders are rows
        data_for_clustering = data.T
    
    # Calculate the linkage matrix
    linkage_matrix = linkage(data_for_clustering, method='ward', metric='euclidean')
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        affinity='euclidean',
        linkage='ward'
    )
    
    cluster_labels = clustering.fit_predict(data_for_clustering)
    
    # Find representatives for each cluster (closest to cluster centroid)
    cluster_representatives = {}
    for cluster_id in range(max(cluster_labels) + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_data = data_for_clustering.iloc[cluster_indices]
        
        # Calculate centroid
        centroid = cluster_data.mean(axis=0)
        
        # Find closest cell type to centroid
        distances = ((cluster_data - centroid) ** 2).sum(axis=1).sort_values()
        representative_idx = distances.index[0]
        
        cluster_representatives[cluster_id] = representative_idx
    
    return linkage_matrix, cluster_labels, cluster_representatives

def perform_cluster_based_pca(data, cluster_labels, cluster_representatives, method='representative'):
    """
    Perform PCA analysis based on clustering results.
    
    Args:
        data: Original data with cell types as columns
        cluster_labels: Cluster assignment for each cell type
        cluster_representatives: Representative cell type for each cluster
        method: 'representative' or 'aggregate'
        
    Returns:
        tuple: (scaled_data, pca, pca_result, loadings)
    """
    if method == 'representative':
        # Use only representative cell types for PCA
        rep_indices = list(cluster_representatives.values())
        if data.shape[0] > data.shape[1]:  # If cell types are rows
            data_for_pca = data.iloc[rep_indices]
        else:  # If disorders are rows
            data_for_pca = data.iloc[:, rep_indices]
    
    elif method == 'aggregate':
        # Aggregate cell types within each cluster (average their values)
        cluster_aggregates = {}
        
        for cluster_id in range(max(cluster_labels) + 1):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if data.shape[0] > data.shape[1]:  # If cell types are rows
                cluster_data = data.iloc[cluster_indices]
                cluster_aggregates[f"Cluster_{cluster_id}"] = cluster_data.mean(axis=0)
            else:  # If disorders are rows
                cluster_data = data.iloc[:, cluster_indices]
                cluster_aggregates[f"Cluster_{cluster_id}"] = cluster_data.mean(axis=1)
        
        data_for_pca = pd.DataFrame(cluster_aggregates).T if data.shape[0] > data.shape[1] else pd.DataFrame(cluster_aggregates)
    
    # Perform standard PCA on the cluster-based data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_pca)

    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    # Get component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=data_for_pca.columns if data.shape[0] > data.shape[1] else data_for_pca.index
    )
    
    return scaled_data, pca, pca_result, loadings, data_for_pca

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

def plot_cluster_heatmap(data, cluster_labels, title="Cell Types Clustered by Disorder Associations"):
    """
    Plot a heatmap of cell types sorted by clusters.
    
    Args:
        data: Original data with cell types as rows and disorders as columns
        cluster_labels: Cluster assignment for each cell type
    """
    # Ensure data has cell types as rows
    if data.shape[0] < data.shape[1]:  # If disorders are rows
        data_for_plot = data.T
    else:
        data_for_plot = data.copy()
    
    # Add cluster info
    data_for_plot['Cluster'] = cluster_labels
    
    # Sort by cluster
    data_for_plot = data_for_plot.sort_values('Cluster')
    
    # Create cluster boundaries for visualization
    cluster_boundaries = []
    current_cluster = None
    for i, cluster in enumerate(data_for_plot['Cluster']):
        if current_cluster != cluster:
            cluster_boundaries.append(i)
            current_cluster = cluster
    
    # Remove cluster column before plotting
    plot_data = data_for_plot.drop('Cluster', axis=1)
    
    # Create heatmap
    plt.figure(figsize=(12, 10), dpi=300)
    ax = sns.heatmap(plot_data, cmap='viridis', center=0)
    
    # Add horizontal lines for cluster boundaries
    for boundary in cluster_boundaries[1:]:  # Skip the first boundary (0)
        ax.axhline(y=boundary, color='red', linewidth=1)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return plt.gcf()

def map_pc_to_original_data(original_data, pca, cluster_data, cluster_labels, method='representative'):
    """
    Map PC scores back to the original cell types.
    
    Args:
        original_data: Original data with cell types as columns
        pca: Fitted PCA object
        cluster_data: Data used for PCA
        cluster_labels: Cluster assignment for each cell type
        method: 'representative' or 'aggregate'
        
    Returns:
        pd.DataFrame: Original data with PC scores added
    """
    # Ensure we're working with cell types as rows
    if original_data.shape[0] < original_data.shape[1]:  # If disorders are rows
        data = original_data.T
    else:
        data = original_data.copy()
    
    # Create a mapping of clusters to PC scores
    if method == 'representative':
        # Get PC scores for representative cell types
        pc_scores = pd.DataFrame(pca.transform(cluster_data), index=cluster_data.index)
        
        # Map each cell type to its cluster's representative PC score
        result = data.copy()
        for i, cluster_id in enumerate(cluster_labels):
            rep_idx = cluster_data.index[cluster_id] if cluster_id < len(cluster_data.index) else None
            if rep_idx is not None:
                for j in range(pc_scores.shape[1]):
                    result.loc[data.index[i], f'PC{j+1}'] = pc_scores.loc[rep_idx, j]
    
    elif method == 'aggregate':
        # For the aggregate method, each cluster gets the same PC score
        pc_scores = pd.DataFrame(pca.transform(cluster_data), index=cluster_data.index)
        
        # Map each cell type to its cluster's PC score
        result = data.copy()
        for i, cluster_id in enumerate(cluster_labels):
            for j in range(pc_scores.shape[1]):
                result.loc[data.index[i], f'PC{j+1}'] = pc_scores.loc[f"Cluster_{cluster_id}", j]
    
    return result

def analyze_bias_pca_hierarchical(input_dir, output_dir=None, n_clusters=None, distance_threshold=0.7, method='representative'):
    """
    Combine bias files from multiple disorders, perform hierarchical clustering,
    then PCA analysis on clusters, and save results.
    
    Args:
        input_dir (str): Directory containing subdirectories of bias files
        output_dir (str, optional): Directory to save results. If None, uses input_dir.
        n_clusters (int, optional): Number of clusters to form (default: determine from distance_threshold)
        distance_threshold (float, optional): Distance threshold for forming clusters
        method (str): 'representative' or 'aggregate' for PCA on clusters
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
    
    # Ensure data has cell types as rows for clustering
    data_for_analysis = combined_df.T if combined_df.shape[0] < combined_df.shape[1] else combined_df
    
    # Perform hierarchical clustering
    print("\nPerforming hierarchical clustering...")
    linkage_matrix, cluster_labels, cluster_representatives = perform_hierarchical_clustering(
        data_for_analysis, n_clusters=n_clusters, distance_threshold=distance_threshold
    )
    
    # Save cluster labels
    cluster_df = pd.DataFrame({'Cluster': cluster_labels}, index=data_for_analysis.index)
    cluster_file = os.path.join(output_dir, "cluster_labels.csv")
    cluster_df.to_csv(cluster_file)
    print(f"Saved cluster labels to {cluster_file}")
    
    # Plot and save dendrogram
    plt.figure(figsize=(15, 10))
    dendrogram_plot = plot_dendrogram(linkage_matrix, names=data_for_analysis.index)
    plt.savefig(os.path.join(output_dir, "dendrogram.png"))
    plt.close()
    
    # Plot and save cluster heatmap
    heatmap = plot_cluster_heatmap(data_for_analysis, cluster_labels)
    plt.savefig(os.path.join(output_dir, "cluster_heatmap.png"))
    plt.close()
    
    # Perform PCA on clusters
    print("\nPerforming PCA on clusters...")
    if method == 'representative':
        rep_indices = list(cluster_representatives.values())
        cluster_data = data_for_analysis.iloc[rep_indices]
    else:  # 'aggregate'
        # Aggregate cell types within each cluster
        cluster_aggregates = {}
        for cluster_id in range(max(cluster_labels) + 1):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_data = data_for_analysis.iloc[cluster_indices]
            cluster_aggregates[f"Cluster_{cluster_id}"] = cluster_data.mean(axis=0)
        cluster_data = pd.DataFrame(cluster_aggregates).T
    
    # Save cluster data used for PCA
    cluster_data_file = os.path.join(output_dir, f"cluster_data_{method}.csv")
    cluster_data.to_csv(cluster_data_file)
    print(f"Saved cluster data to {cluster_data_file}")
    
    # Perform PCA on cluster data
    scaled_data, pca, pca_result, loadings, _ = perform_cluster_based_pca(
        combined_df, cluster_labels, cluster_representatives, method=method
    )
    
    # Create dataframe with PC scores for clusters
    pc_scores = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
        index=cluster_data.index
    )
    
    # Save PC scores for clusters
    pc_scores_file = os.path.join(output_dir, f"pc_scores_{method}.csv")
    pc_scores.to_csv(pc_scores_file)
    print(f"Saved PC scores to {pc_scores_file}")
    
    # Save loadings
    loadings_file = os.path.join(output_dir, f"loadings_{method}.csv")
    loadings.to_csv(loadings_file)
    print(f"Saved loadings to {loadings_file}")
    
    # Map PC scores back to original cell types
    mapped_data = map_pc_to_original_data(combined_df, pca, cluster_data, cluster_labels, method=method)
    mapped_file = os.path.join(output_dir, f"cell_type_pc_scores_{method}.csv")
    mapped_data.to_csv(mapped_file)
    print(f"Saved mapped PC scores for all cell types to {mapped_file}")
    
    # Create and save scree plot
    plot_scree_and_get_loadings(pca)
    plt.savefig(os.path.join(output_dir, f"scree_plot_{method}.png"))
    plt.close()
    
    # Print explained variance
    print("\nExplained variance ratios:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    
    # Additional analysis: Compare with standard PCA
    print("\nPerforming standard PCA for comparison...")
    standard_scaled_data, standard_pca, standard_pca_result, standard_loadings = perform_pca_analysis(combined_df)
    
    # Save standard PCA results
    standard_pc_scores = pd.DataFrame(
        standard_pca_result,
        columns=[f'PC{i+1}' for i in range(standard_pca_result.shape[1])],
        index=combined_df.index
    )
    
    standard_pc_scores_file = os.path.join(output_dir, "standard_pc_scores.csv")
    standard_pc_scores.to_csv(standard_pc_scores_file)
    
    standard_loadings_file = os.path.join(output_dir, "standard_loadings.csv")
    standard_loadings.to_csv(standard_loadings_file)
    
    # Compare explained variance
    print("\nComparison of explained variance:")
    print("Hierarchical clustering + PCA:")
    for i, var in enumerate(pca.explained_variance_ratio_[:3]):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    print("\nStandard PCA:")
    for i, var in enumerate(standard_pca.explained_variance_ratio_[:3]):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    # Correlation between hierarchical and standard PC1
    if method == 'representative':
        rep_pc1 = mapped_data['PC1'] if 'PC1' in mapped_data.columns else None
    else:
        # For aggregate method, need to map back differently
        rep_pc1 = None
        for i, cluster_id in enumerate(cluster_labels):
            if rep_pc1 is None:
                rep_pc1 = pd.Series(index=data_for_analysis.index)
            rep_pc1.iloc[i] = pc_scores.loc[f"Cluster_{cluster_id}", 0] if f"Cluster_{cluster_id}" in pc_scores.index else None
    
    # Only compare if we have both PC1s
    if rep_pc1 is not None and 'PC1' in standard_pc_scores.columns:
        # Need to align indices for correlation
        if combined_df.shape[0] < combined_df.shape[1]:  # If disorders are rows
            std_pc1 = standard_pc_scores['PC1']
            correlation = np.corrcoef(rep_pc1, std_pc1)[0, 1]
        else:
            # Need to reindex standard PC scores to match mapped data
            correlation = np.corrcoef(rep_pc1, standard_pc_scores['PC1'])[0, 1]
        
        print(f"\nCorrelation between hierarchical and standard PC1: {correlation:.3f}")

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

def main():
    parser = argparse.ArgumentParser(description='Combine bias files, perform hierarchical clustering and PCA analysis')
    parser.add_argument('input_dir', help='Directory containing subdirectories of bias files')
    parser.add_argument('--output_dir', help='Directory to save results (default: same as input_dir)')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters to form (default: automatic)')
    parser.add_argument('--distance_threshold', type=float, default=0.7, help='Distance threshold for forming clusters')
    parser.add_argument('--method', choices=['representative', 'aggregate'], default='representative', 
                        help='Method for PCA on clusters: "representative" uses representative cell type from each cluster, "aggregate" uses averaged values')
    
    args = parser.parse_args()
    analyze_bias_pca_hierarchical(
        args.input_dir, 
        args.output_dir, 
        n_clusters=args.n_clusters, 
        distance_threshold=args.distance_threshold,
        method=args.method
    )

if __name__ == "__main__":
    main()