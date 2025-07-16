#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

def load_bias_data(input_dir):
    """Load bias data from the specified directory."""
    # Load the bias matrix
    bias_file = os.path.join(input_dir, 'bias_matrix.csv')
    if not os.path.exists(bias_file):
        raise FileNotFoundError(f"Bias matrix not found at {bias_file}")
    
    return pd.read_csv(bias_file, index_col=0)

def perform_pca(data, n_components=3):
    """Perform PCA on the data."""
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    return pca, pca_result, pca.explained_variance_ratio_

def create_3d_plot(pca_result, explained_var, output_dir, title_prefix="Disorder"):
    """Create and save 3D PCA plot."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot
    scatter = ax.scatter(pca_result[:, 1],  # PC2 on x-axis
                        pca_result[:, 0],  # PC1 on y-axis
                        pca_result[:, 2],  # PC3 on z-axis
                        c=pca_result[:, 0],  # Color by PC1
                        cmap='viridis',
                        alpha=0.6)

    # Add colorbar
    cbar = plt.colorbar(scatter, label='PC1 (P-factor)', shrink=0.5)

    # Set labels with explained variance
    ax.set_xlabel(f'PC2 ({explained_var[1]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC1 ({explained_var[0]:.1%} var)', fontsize=12)
    ax.set_zlabel(f'PC3 ({explained_var[2]:.1%} var)', fontsize=12)

    plt.title(f'3D PCA Plot of {title_prefix} Cell Type Associations', fontsize=14)

    # Adjust view angle
    ax.view_init(elev=20, azim=45)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_3d_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_pca_results(pca_result, explained_var, output_dir):
    """Save PCA results to files."""
    # Save PCA coordinates
    pca_df = pd.DataFrame(pca_result, 
                         columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    pca_df.to_csv(os.path.join(output_dir, 'pca_coordinates.csv'))
    
    # Save explained variance
    var_df = pd.DataFrame({
        'PC': range(1, len(explained_var) + 1),
        'Explained_Variance': explained_var
    })
    var_df.to_csv(os.path.join(output_dir, 'explained_variance.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='Perform PCA analysis on disorder-specific bias data')
    parser.add_argument('input_dir', help='Directory containing the bias matrix')
    parser.add_argument('--output_dir', help='Directory to save results (default: input_dir/PCA)')
    parser.add_argument('--title_prefix', default='Disorder', help='Prefix for plot title')
    args = parser.parse_args()

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'PCA')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load data
        print("Loading bias data...")
        data = load_bias_data(args.input_dir)

        # Perform PCA
        print("Performing PCA...")
        pca, pca_result, explained_var = perform_pca(data)

        # Create visualization
        print("Creating 3D visualization...")
        create_3d_plot(pca_result, explained_var, args.output_dir, args.title_prefix)

        # Save results
        print("Saving PCA results...")
        save_pca_results(pca_result, explained_var, args.output_dir)

        print(f"Analysis complete. Results saved to {args.output_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main()) 