import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.cluster import hierarchy

from adjustText import adjust_text

plt.style.use('seaborn-v0_8-whitegrid')

Anno = pd.read_excel("/mnt/data0/HumanBrainCellType/annotation.xlsx", index_col="Cluster")
Anno.drop(Anno.tail(1).index,inplace=True) # drop last n rows
Anno.fillna('', inplace=True)
Anno.index = [int(x) for x in Anno.index.values]

for i, row in Anno.iterrows():
    Class, NT = row["Class auto-annotation"], row["Neurotransmitter auto-annotation"]
    if NT != "":
        Anno.loc[i, "Class auto-annotation"] = "NEUR"
Neur_idx = [int(x) for x in Anno[Anno["Class auto-annotation"]=="NEUR"].index]
NonNeur_idx = [int(x) for x in Anno[Anno["Class auto-annotation"]!="NEUR"].index]

Neurons = sorted(list(set(Anno[Anno["Class auto-annotation"]=="NEUR"]["Supercluster"])))
Not_Neurons = sorted(list(set(Anno[Anno["Class auto-annotation"]!="NEUR"]["Supercluster"])))
Not_Neurons = [x for x in Not_Neurons if x not in Neurons]
ALL_CTs = Neurons + Not_Neurons


def create_scatter_plot(x_values, y_values, xlabel, ylabel, disorders, dpi=80):
    """Create scatter plot with non-overlapping annotations and correlation stats, styled like plot_supercluster_specificity"""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=dpi, facecolor='none', edgecolor='k')
    ax.set_facecolor('none')
    # Use the same marker and color style as plot_supercluster_specificity
    scatter = ax.scatter(x_values, y_values, s=80, color="#1f77b4", edgecolor="k", zorder=3)

    # Add linear fit (log10 x)
    x_log = np.log10(x_values)
    z = np.polyfit(x_log, y_values, 1)
    p = np.poly1d(z)
    x_fit = np.logspace(min(x_log), max(x_log), 100)
    ax.plot(x_fit, p(np.log10(x_fit)), "r--", alpha=0.8, lw=2, zorder=2)

    # Axes labels and ticks
    ax.set_xlabel(xlabel, fontsize=18, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=10)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=16, length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', labelsize=14, length=3, width=1)
    ax.set_axisbelow(True)

    # Ensure bottom and left spines are visible and thickened
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add disorder name annotations (collect first)
    texts = []
    for i, txt in enumerate(disorders):
        texts.append(
            ax.text(x_values[i], y_values[i], txt, fontsize=13, fontweight='bold', color="#333333", zorder=4)
        )

    # Automatically adjust text to avoid overlap
    adjust_text(
        texts,
        only_move={'points': 'y', 'texts': 'y'},
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.7, alpha=0.7),
        force_text=0.5,
        force_points=0.5,
        ax=ax
    )

    #corr = scipy.stats.pearsonr(x_values, y_values)
    corr = stats.spearmanr(x_values, y_values)
    ax.text(
        0.02, 0.98,
        f'ρ = {corr[0]:.3f}\np = {corr[1]:.2e}',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round,pad=0.3'),
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='left'
    )

    # Adjust layout to prevent label cutoff and accommodate annotation
    plt.subplots_adjust(bottom=0.18, right=0.95)
    plt.show()


def plot_HumanCT_boxplot(DF, Anno, ALL_CTs, label, title, dpi=80):
    plt.figure(figsize=(12,6), dpi=dpi, transparent=True)

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


def plot_correlation_humanCT(values1, values2, name1, name2, title="", xlim=None, Neur_idx=None, NonNeur_idx=None, dpi=80, ax=None, addXY = False):
    # Calculate correlations and p-values
    corr_all_res = stats.spearmanr(values1, values2)
    corr_all = corr_all_res.correlation if hasattr(corr_all_res, 'correlation') else corr_all_res[0]
    p_all = corr_all_res.pvalue if hasattr(corr_all_res, 'pvalue') else corr_all_res[1]

    if Neur_idx is not None and NonNeur_idx is not None:
        corr_neur_res = stats.spearmanr(values1[Neur_idx], values2[Neur_idx])
        corr_neur = corr_neur_res.correlation if hasattr(corr_neur_res, 'correlation') else corr_neur_res[0]
        p_neur = corr_neur_res.pvalue if hasattr(corr_neur_res, 'pvalue') else corr_neur_res[1]

        corr_nonneur_res = stats.spearmanr(values1[NonNeur_idx], values2[NonNeur_idx])
        corr_nonneur = corr_nonneur_res.correlation if hasattr(corr_nonneur_res, 'correlation') else corr_nonneur_res[0]
        p_nonneur = corr_nonneur_res.pvalue if hasattr(corr_nonneur_res, 'pvalue') else corr_nonneur_res[1]

    if ax is None:
        plt.figure(figsize=(6,6), dpi = dpi)
        ax = plt.gca()
    
    if Neur_idx is not None and NonNeur_idx is not None:
        ax.scatter(values1[Neur_idx], values2[Neur_idx],
                color="red", alpha=0.6, s=80, label="Neuronal")
        ax.scatter(values1[NonNeur_idx], values2[NonNeur_idx],
                color="blue", alpha=0.6, s=80, label="Non-neuronal")

        # Add text annotations on lower right with p-values
        ax.text(0.95, 0.05,
                 f'All: ρ = {corr_all:.2f} (p={p_all:.0e})\n'
                 f'Neuronal: ρ = {corr_neur:.2f} (p={p_neur:.0e})\n'
                 f'Non-neuronal: ρ = {corr_nonneur:.2f} (p={p_nonneur:.0e})',
                 transform=ax.transAxes,
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 fontsize=12)
    else:
        ax.scatter(values1, values2, alpha=0.6, s=80)
        ax.text(0.95, 0.05,
                 f'ρ = {corr_all:.2f} (p={p_all:.1e})',
                 transform=ax.transAxes,
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 fontsize=12)

    # Add X=Y line if requested
    if addXY:
        # Determine plotting range
        if xlim is not None:
            min_val, max_val = xlim
        else:
            min_val = min(np.min(values1), np.min(values2))
            max_val = max(np.max(values1), np.max(values2))
        ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1.5, label='X=Y')

    ax.set_xlabel(name1, fontsize=22)
    ax.set_ylabel(name2, fontsize=22)
    if title == None:
        pass
    ax.set_title(f"{title}", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlim(xlim)
    if Neur_idx is not None and NonNeur_idx is not None:
        ax.legend(fontsize=15, loc="upper left")


def plot_correlation_scatter_mouseCT(df1, df2, name1, name2, effect_col1="EFFECT", effect_col2="EFFECT", class_col="class_id_label", ax=None, dpi=80):
    # Find indices that appear in both dataframes
    common_indices = df1.index.intersection(df2.index)

    # Filter both dataframes to only include common indices and sort them
    df1_filtered = df1.loc[common_indices].sort_index()
    df2_filtered = df2.loc[common_indices].sort_index()

    # Ensure the indices are aligned
    assert df1_filtered.index.equals(df2_filtered.index), "Indices are not properly aligned"

    # Extract values
    values1 = df1_filtered[effect_col1].values
    values2 = df2_filtered[effect_col2].values

    # Get class labels (using df1, but they should be the same in both)
    try:
        class_labels = df1_filtered[class_col].values
    except:
        class_labels = df2_filtered[class_col].values

    # Create boolean mask for non-neuronal cell types
    non_neuronal_types = ['30 Astro-Epen',
                         '31 OPC-Oligo',
                         '32 OEC',
                         '33 Vascular',
                         '34 Immune']
    non_neur_mask = np.isin(class_labels, non_neuronal_types)

    # Calculate correlations and p-values
    corr_all, p_all = stats.spearmanr(values1, values2)
    corr_neur, p_neur = stats.spearmanr(values1[~non_neur_mask], values2[~non_neur_mask])
    corr_nonneur, p_nonneur = stats.spearmanr(values1[non_neur_mask], values2[non_neur_mask])

    print(f"All cells correlation: {corr_all:.2f} (p={p_all:.0e})")
    print(f"Neuronal correlation: {corr_neur:.2f} (p={p_neur:.0e})")
    print(f"Non-neuronal correlation: {corr_nonneur:.2f} (p={p_nonneur:.0e})")

    if ax is None:
        plt.figure(figsize=(6.5,6), dpi=dpi)
        ax = plt.gca()

    ax.scatter(values1[~non_neur_mask], values2[~non_neur_mask],
            color="red", alpha=0.6, s=20, label="Neuronal")
    ax.scatter(values1[non_neur_mask], values2[non_neur_mask],
            color="blue", alpha=0.6, s=20, label="Non-neuronal")
    ax.set_xlabel(name1, fontsize=22)
    ax.set_ylabel(name2, fontsize=22)
    ax.legend(fontsize=12, loc="upper left")

    # Add correlation information
    ax.text(0.95, 0.05,
             f'All: ρ = {corr_all:.2f} (p={p_all:.0e})\n'
             f'Neuronal: ρ = {corr_neur:.2f} (p={p_neur:.0e})\n'
             f'Non-neuronal: ρ = {corr_nonneur:.2f} (p={p_nonneur:.0e})',
             transform=ax.transAxes,
             horizontalalignment='right',
             verticalalignment='bottom',
             fontsize=12)

    ax.tick_params(axis='both', which='major', labelsize=15)


def plot_correlation_scatter_mouseCT_axis(values1, values2, name1, name2, ax=None, dpi=80):
    """
    Plot correlation scatter for MouseCT data with neuronal/non-neuronal classification using provided axis

    Parameters:
    -----------
    values1, values2 : array-like
        Values to plot (should be aligned)
    name1, name2 : str
        Labels for x and y axes
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure
    dpi : int
        DPI for figure if creating new one
    """
    if ax is None:
        plt.figure(figsize=(6.5,6), dpi=dpi)
        ax = plt.gca()

    # Since we're working with pre-extracted values, we need to determine neuronal/non-neuronal split
    # For MouseCT PC consistency, we'll use a simpler approach and just plot all points
    # with correlation information

    # Calculate correlation
    corr_all = stats.spearmanr(values1, values2)[0]

    # Plot all points as single color for PC consistency analysis
    ax.scatter(values1, values2, color="red", alpha=0.6, s=20)
    ax.set_xlabel(name1, fontsize=22)
    ax.set_ylabel(name2, fontsize=22)

    # Add correlation information
    ax.text(0.95, 0.05,
             f'ρ = {corr_all:.3f}',
             transform=ax.transAxes,
             horizontalalignment='right',
             verticalalignment='bottom',
             fontsize=15)

    ax.tick_params(axis='both', which='major', labelsize=15)
    return ax


def plot_effect_vs_pfactor_mouseSTR_axis(values1, values2, name1, name2, ax=None, dpi=80):
    """
    Plot scatter plot for MouseSTR data using provided axis

    Parameters:
    -----------
    values1, values2 : array-like
        Values to plot (should be aligned)
    name1, name2 : str
        Labels for x and y axes
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure
    dpi : int
        DPI for figure if creating new one
    """
    if ax is None:
        plt.figure(figsize=(6.5, 6), dpi=dpi)
        ax = plt.gca()

    # Plot scatter
    ax.scatter(values1, values2,
                alpha=0.7, s=80, color='red', edgecolors='black', linewidth=0.5)

    # Labels and formatting
    ax.set_xlabel(f'{name1}', fontsize=20)
    ax.set_ylabel(f'{name2}', fontsize=20)

    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Calculate and display correlation
    corr = stats.spearmanr(values1, values2)

    ax.text(0.70, 0.1, f'r = {corr[0]:.2f}',
            transform=ax.transAxes,
            fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, linestyle='--', alpha=0.7)
    return ax


def plot_disorder_burden_comparison_gwas(target_disorder, RareCoding_data, GWAS_data, top_n_values=[100, 500, 1000], dpi=80):
    """
    Plot burden comparison between rare coding and GWAS data for a target disorder
    """

    # Create subplots
    fig, axes = plt.subplots(1, len(top_n_values), figsize=(24,8), dpi=dpi, transparent=True)

    for i, top_n in enumerate(top_n_values):
        ax = axes[i] if len(top_n_values) > 1 else axes

        # Get top genes from rare coding data
        target_rare = RareCoding_data[target_disorder].dropna()
        top_genes_rare = target_rare.nlargest(top_n).index

        # Initialize lists to store burdens
        gwas_burdens = []
        rare_burdens = []
        disorder_names = []

        # Calculate burdens for each disorder in GWAS data
        for disorder in GWAS_data.columns:
            if disorder == target_disorder:
                continue

            # Get GWAS data for this disorder, drop NaNs
            gwas_disorder = GWAS_data[disorder].dropna()

            # Find intersection of top rare coding genes with available GWAS genes
            common_genes = top_genes_rare.intersection(gwas_disorder.index)

            if len(common_genes) > 10:  # Only include if we have sufficient overlap
                # Calculate burden for common genes
                gwas_burden = gwas_disorder.loc[common_genes].sum()
                rare_burden = target_rare.loc[common_genes].sum()

                gwas_burdens.append(gwas_burden)
                rare_burdens.append(rare_burden)
                disorder_names.append(disorder)

        # Create scatter plot
        ax.scatter(rare_burdens, gwas_burdens, alpha=0.7, s=100)

        # Add disorder labels
        for j, disorder in enumerate(disorder_names):
            ax.annotate(disorder, (rare_burdens[j], gwas_burdens[j]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Calculate and display correlation
        if len(rare_burdens) > 3:
            correlation = np.corrcoef(rare_burdens, gwas_burdens)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_xlabel(f'{target_disorder} Rare Coding Burden\n(Top {top_n} genes)', fontsize=12)
        ax.set_ylabel('GWAS Burden', fontsize=12)
        ax.set_title(f'Burden Comparison (Top {top_n} genes)\nn_disorders = {len(disorder_names)}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add diagonal line for reference
        min_val = min(min(rare_burdens) if rare_burdens else 0,
                     min(gwas_burdens) if gwas_burdens else 0)
        max_val = max(max(rare_burdens) if rare_burdens else 1,
                     max(gwas_burdens) if gwas_burdens else 1)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.show()


def plot_disorder_burden_comparison_all(target_disorder, RareCoding_data, GWAS_data, top_n_values=[100, 500, 1000], dpi=80):
    """
    Plot burden comparison between rare coding and GWAS data for a target disorder
    """

    # Create subplots
    fig, axes = plt.subplots(1, len(top_n_values), figsize=(24,8), dpi=dpi)

    for i, top_n in enumerate(top_n_values):
        ax = axes[i] if len(top_n_values) > 1 else axes

        # Get top genes from rare coding data
        target_rare = RareCoding_data[target_disorder].dropna()
        top_genes_rare = target_rare.nlargest(top_n).index

        # Initialize lists to store burdens
        gwas_burdens = []
        rare_burdens = []
        disorder_names = []

        # Calculate burdens for each disorder in GWAS data
        for disorder in GWAS_data.columns:
            # Get GWAS data for this disorder, drop NaNs
            gwas_disorder = GWAS_data[disorder].dropna()

            # Find intersection of top rare coding genes with available GWAS genes
            common_genes = top_genes_rare.intersection(gwas_disorder.index)

            if len(common_genes) > 10:  # Only include if we have sufficient overlap
                # Calculate burden for common genes
                gwas_burden = gwas_disorder.loc[common_genes].sum()
                rare_burden = target_rare.loc[common_genes].sum()

                gwas_burdens.append(gwas_burden)
                rare_burdens.append(rare_burden)
                disorder_names.append(disorder)

        # Create scatter plot
        scatter = ax.scatter(rare_burdens, gwas_burdens, alpha=0.7, s=100)

        # Highlight target disorder if it's in the data
        if target_disorder in disorder_names:
            target_idx = disorder_names.index(target_disorder)
            ax.scatter(rare_burdens[target_idx], gwas_burdens[target_idx],
                      color='red', s=150, marker='*', label=f'{target_disorder} (target)')

        # Add disorder labels
        for j, disorder in enumerate(disorder_names):
            if disorder == target_disorder:
                continue  # Skip labeling the target disorder to avoid overlap
            ax.annotate(disorder, (rare_burdens[j], gwas_burdens[j]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Label the target disorder separately
        if target_disorder in disorder_names:
            target_idx = disorder_names.index(target_disorder)
            ax.annotate(f'{target_disorder} (target)',
                       (rare_burdens[target_idx], gwas_burdens[target_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='red')

        # Calculate and display correlation
        if len(rare_burdens) > 3:
            correlation = np.corrcoef(rare_burdens, gwas_burdens)[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_xlabel(f'{target_disorder} Rare Coding Burden\n(Top {top_n} genes)', fontsize=12)
        ax.set_ylabel('GWAS Burden', fontsize=12)
        ax.set_title(f'Burden Comparison (Top {top_n} genes)\nn_disorders = {len(disorder_names)}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add diagonal line for reference
        min_val = min(min(rare_burdens) if rare_burdens else 0,
                     min(gwas_burdens) if gwas_burdens else 0)
        max_val = max(max(rare_burdens) if rare_burdens else 1,
                     max(gwas_burdens) if gwas_burdens else 1)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)

        # Add legend if target disorder is highlighted
        if target_disorder in disorder_names:
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_beta_distribution_by_supercluster(results_df, EFFECT="beta", dpi=80):
    """
    Create a boxplot showing the distribution of beta values by supercluster.

    Args:
        results_df (pd.DataFrame): DataFrame containing results with 'Supercluster' and effect columns
        EFFECT (str): Column name for the effect size (default: "beta")

    Returns:
        None: Displays the plot using matplotlib
    """
    plt.figure(figsize=(12, 6), dpi=dpi)

    # Create boxplot
    results_df.boxplot(column=EFFECT, by='Supercluster', ax=plt.gca())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Supercluster')
    plt.ylabel(EFFECT)
    plt.title('Distribution of {} by Supercluster'.format(EFFECT))
    plt.tight_layout()
    plt.show()


def plot_pc_loadings(pca, spearman_df_sub, topLoadings=10, dpi=80):
    """
    Plot the top positive and negative loadings for the first few principal components

    Args:
        pca: fitted PCA object
        spearman_df_sub: DataFrame with disorder names as columns (for loading labels)
        topLoadings: number of top loadings to show (default: 10)

    Returns:
        matplotlib figure
    """

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), dpi=dpi)
    axes = axes.flatten()

    n_components = min(12, pca.n_components_)

    for i in range(n_components):
        ax = axes[i]

        # Get loadings for this PC
        loadings = pca.components_[i]
        feature_names = spearman_df_sub.columns

        # Create a series for easier sorting
        loading_series = pd.Series(loadings, index=feature_names)

        # Get top positive and negative loadings
        top_positive = loading_series.nlargest(topLoadings)
        top_negative = loading_series.nsmallest(topLoadings)

        # Combine and sort by absolute value for plotting
        combined = pd.concat([top_positive, top_negative]).drop_duplicates()
        combined_sorted = combined.reindex(combined.abs().sort_values(ascending=True).index)

        # Create horizontal bar plot
        colors = ['red' if x > 0 else 'blue' for x in combined_sorted.values]
        bars = ax.barh(range(len(combined_sorted)), combined_sorted.values, color=colors, alpha=0.7)

        # Customize the plot
        ax.set_yticks(range(len(combined_sorted)))
        ax.set_yticklabels(combined_sorted.index, fontsize=8)
        ax.set_xlabel(f'PC{i+1} Loadings', fontsize=10)
        ax.set_title(f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%} variance)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linewidth=0.8)

        # Set x-axis limits for better visualization
        max_abs = max(abs(combined_sorted.min()), abs(combined_sorted.max()))
        ax.set_xlim(-max_abs*1.1, max_abs*1.1)

    plt.suptitle('Principal Component Loadings Analysis', fontsize=21, y=1.02)  # Increased from 14
    plt.tight_layout()
    plt.show()


def plot_scree_and_get_loadings(pca, dpi=80):
    """
    Create a scree plot showing explained variance ratio for each principal component

    Args:
        pca: fitted PCA object

    Returns:
        fig: matplotlib figure object
        loadings_df: DataFrame containing the loadings for each PC
    """

    fig = plt.figure(figsize=(10, 6), dpi=dpi)
    plt.plot(range(0, len(pca.explained_variance_ratio_) + 1),
             [0] + list(np.cumsum(pca.explained_variance_ratio_)),
             'bo-', linewidth=2, markersize=8, markerfacecolor='lightblue',
             markeredgecolor='blue', markeredgewidth=2)
    plt.xlabel('Number of Components', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Explained\nVariance Ratio', fontsize=14, fontweight='bold')
    #plt.title('Scree Plot', fontsize=16, fontweight='bold', pad=15)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Improve tick formatting
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add annotations for individual contributions
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        if i < 10:  # Only annotate first 10 components to avoid clutter
            cum_var = np.cumsum(pca.explained_variance_ratio_)[i]
            plt.text(i+1.1, cum_var-0.01, f'+{pca.explained_variance_ratio_[i]:.3f}',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

    # Improve layout
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Set reasonable axis limits
    plt.xlim(-0.5, len(pca.explained_variance_ratio_) + 0.5)
    plt.ylim(-0.05, 1.05)

    plt.show()



def plot_Effect_boxplot_HumanCT(pc_scores_df, Anno, ALL_CTs, PC, ylabel="PC1 Score", dpi=80):
    """
    Create a boxplot of PC1 scores by cell type, sorted by median values
    """
    plt.figure(figsize=(12,6), dpi=dpi)

    # Create list to store data for boxplot and calculate medians for sorting
    box_data = []
    tick_labels = []
    medians = []
    cell_types = []  # Store for neuronal classification

    # Collect data and medians for each cell type
    for CT in ALL_CTs:
        CT_idx = Anno[Anno["Supercluster"]==CT].index.values
        if len(CT_idx) > 0:  # Only include if we have data
            dat = pc_scores_df.loc[CT_idx, PC]
            box_data.append(dat)
            tick_labels.append(CT)
            medians.append(dat.median())
            cell_types.append(CT)

    # Sort everything by medians
    sorted_indices = np.argsort(medians)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]
    sorted_cell_types = [cell_types[i] for i in sorted_indices]

    # Create boxplot with custom colors
    bp = plt.boxplot(box_data, tick_labels=tick_labels, patch_artist=True,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2, color='black'))

    # Color boxes based on cell type
    palette = {'Neuron': '#FF6B6B', 'Non-neuron': '#4ECDC4'}
    for patch, cell_type in zip(bp['boxes'], sorted_cell_types):
        if cell_type in Neurons:
            patch.set_facecolor(palette['Neuron'])
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(palette['Non-neuron'])
            patch.set_alpha(0.7)

    # Customize plot
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('{}'.format(ylabel), fontsize=14, fontweight='bold')
    #plt.title('Distribution of {} Scores by Human Cell Type'.format(PC),
    #         fontsize=16, fontweight='bold', pad=20)
    #plt.title("Human Cell Type", fontsize=16)

    # Add subtle grid
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax = plt.gca()
    ax.set_axisbelow(True)

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=palette['Neuron'], alpha=0.7, label='Neuron'),
                      plt.Rectangle((0,0),1,1, facecolor=palette['Non-neuron'], alpha=0.7, label='Non-neuron')]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=15)

    plt.tight_layout()
    plt.show()

def plot_pc1_boxplot_mouseCT(pc_scores_df, Anno, ALL_CTs, PC, ylabel="PC1 Score", dpi=80):
    plt.figure(figsize=(12,5), dpi=dpi)

    # Create list to store data for boxplot and calculate medians for sorting
    box_data = []
    tick_labels = []
    medians = []
    cell_types = []  # Store for neuronal classification

    # Collect data and medians for each cell type
    for CT in ALL_CTs:
        CT_idx = Anno[Anno["class_id_label"]==CT].index.values
        CT_idx = [x for x in CT_idx if x in pc_scores_df.index]
        if len(CT_idx) > 0:  # Only include if we have data
            dat = pc_scores_df.loc[CT_idx, PC]
            box_data.append(dat)
            tick_labels.append(CT)
            medians.append(dat.median())
            cell_types.append(CT)

    # Sort everything by medians
    sorted_indices = np.argsort(medians)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]
    sorted_cell_types = [cell_types[i] for i in sorted_indices]

    # Define neuronal vs non-neuronal classification for mouse
    non_neuronal_types = ['30 Astro-Epen',
                         '31 OPC-Oligo',
                         '32 OEC',
                         '33 Vascular',
                         '34 Immune']

    # Create boxplot with custom colors
    bp = plt.boxplot(box_data, tick_labels=tick_labels, patch_artist=True,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2, color='black'))

    # Color boxes based on cell type
    palette = {'Neuron': '#FF6B6B', 'Non-neuron': '#4ECDC4'}
    for patch, cell_type in zip(bp['boxes'], sorted_cell_types):
        if cell_type in non_neuronal_types:
            patch.set_facecolor(palette['Non-neuron'])
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(palette['Neuron'])
            patch.set_alpha(0.7)

    # Customize plot
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('{}'.format(ylabel), fontsize=14, fontweight='bold')
    #plt.title("Mouse Cell Type", fontsize=16)

    # Add subtle grid
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax = plt.gca()
    ax.set_axisbelow(True)

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=palette['Neuron'], alpha=0.7, label='Neuron'),
                      plt.Rectangle((0,0),1,1, facecolor=palette['Non-neuron'], alpha=0.7, label='Non-neuron')]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=15)


    plt.tight_layout()
    plt.show()

def plot_pc1_boxplot_MouseSTR(pc_scores_df, Regions, REG2STR, PC, annotate_top10=False, dpi=80, xlabel="Neural P-factor Score"):

    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(7,8), dpi=dpi)

    # Create list to store data for boxplot and calculate medians for sorting
    box_data = []
    tick_labels = []
    medians = []
    region_to_structures = []

    # Collect data and medians for each region
    for region in Regions:
        # Get structures for this region
        structures = REG2STR[region]

        # Get valid structures that exist in the PC scores
        valid_structures = [struct for struct in structures if struct in pc_scores_df.index]

        if len(valid_structures) > 0:  # Only include if we have data
            dat = pc_scores_df.loc[valid_structures, PC]
            box_data.append(dat)
            # Replace underscores with spaces for display
            tick_labels.append(region.replace("_", " "))
            medians.append(dat.median())
            region_to_structures.append(valid_structures)

    # Sort everything by medians
    sorted_indices = np.argsort(medians)
    box_data = [box_data[i] for i in sorted_indices]
    tick_labels = [tick_labels[i] for i in sorted_indices]
    sorted_region_structures = [region_to_structures[i] for i in sorted_indices]

    # Create horizontal boxplot with enhanced styling
    bp = ax.boxplot(box_data, tick_labels=tick_labels, patch_artist=True, vert=False,
                    boxprops=dict(linewidth=1.5, facecolor='lightblue', alpha=0.7),
                    whiskerprops=dict(linewidth=1.5, color='darkblue'),
                    capprops=dict(linewidth=1.5, color='darkblue'),
                    medianprops=dict(linewidth=2.5, color='darkred'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))

    # Add individual data points and collect positions for annotation
    texts = []
    point_positions = {}  # structure -> (x, y) position
    
    for i, (data, structures) in enumerate(zip(box_data, sorted_region_structures)):
        y_pos = i + 1
        # Add some jitter to y-position for better visibility
        y_jitter = np.random.normal(0, 0.05, len(data))
        
        ax.scatter(data, [y_pos] * len(data) + y_jitter, 
                  alpha=0.6, s=15, color='black', zorder=3)
        
        # Store positions for each structure (only if annotation is enabled)
        if annotate_top10:
            for j, (value, struct) in enumerate(zip(data, structures)):
                point_positions[struct] = (value, y_pos + y_jitter[j])
    
    # Add annotations for top 10 structures if enabled
    if annotate_top10:
        # Get top 10 structures by PC1 score
        pc_scores_sorted = pc_scores_df[PC].sort_values(ascending=False)
        top_10_structures = pc_scores_sorted.head(10).index.tolist()
        
        # Add annotations for top 10 structures that are in our plot
        for struct in top_10_structures:
            if struct in point_positions:
                x_pos, y_pos = point_positions[struct]
                
                # Clean structure name for display
                display_name = struct.replace("_", " ")
                
                texts.append(
                    ax.text(x_pos, y_pos, display_name, fontsize=10, fontweight='bold', 
                           color="#333333", zorder=4)
                )
        
        # Use adjust_text to prevent overlapping annotations
        if texts:  # Only if we have annotations to adjust
            from adjustText import adjust_text
            adjust_text(
                texts,
                only_move={'points': 'xy', 'texts': 'xy'},
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.7, alpha=0.7),
                force_text=0.5,
                force_points=0.5,
                ax=ax
            )

    # Customize plot
    ax.set_yticklabels(tick_labels, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    if xlabel != "":
        ax.set_xlabel(f'{xlabel}', fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel(f'{PC} Score', fontsize=16, fontweight='bold')
    #ax.set_xlabel(f'{PC} Score', fontsize=16, fontweight='bold')
    #ax.set_title("Brain Region", fontsize=18, fontweight='bold', pad=20)

    # Enhanced grid
    ax.grid(True, linestyle='--', alpha=0.4, axis='x')
    ax.set_axisbelow(True)

    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add subtle background color
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout(pad=1.5)
    plt.show()

def plot_disorder_correlation_heatmap(df, title="", cluster=True, dpi=80):
    """
    Create a correlation heatmap for disorder data

    Args:
        df: DataFrame with disorders as columns
        title: Title for the plot
        cluster: Whether to perform hierarchical clustering
    """
    # Calculate correlation matrix
    corr_matrix = df.corr(method='spearman')

    # Perform hierarchical clustering if requested
    if cluster:
        linkage_matrix = hierarchy.linkage(corr_matrix, method='ward')
        dendro = hierarchy.dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
        cluster_order = dendro['leaves']
        corr_matrix = corr_matrix.iloc[cluster_order, cluster_order]

    plt.figure(figsize=(7,6), dpi=dpi)

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Mask upper triangle

    # Disable grid by setting current axes to have no grid
    ax = plt.gca()
    ax.grid(False)

    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
                annot_kws={'fontsize': 10})

    plt.title('{}'.format(title),
              fontsize=16,
              fontweight='bold',
              pad=20)

    # Customize ticks
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(rotation=0, fontsize=15)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    return corr_matrix


def plot_effect_vs_pfactor_mouseSTR(mouse_str_DF1, mouse_str_DF2, name1, name2, label1="EFFECT", label2="EFFECT", ax=None, dpi=80):
    """
    Plot EFFECT values between two mouse structure datasets
    """
    # Find common indices
    common_idx = mouse_str_DF1.index.intersection(mouse_str_DF2.index)

    if ax is None:
        plt.figure(figsize=(6.5, 6), dpi=dpi)
        plt.style.use('seaborn-v0_8-whitegrid')
        ax = plt.gca()

    # Plot scatter
    ax.scatter(mouse_str_DF1.loc[common_idx, label1], mouse_str_DF2.loc[common_idx, label2],
                alpha=0.7, s=80, color='red', edgecolors='black', linewidth=0.5)

    # Labels and formatting
    ax.set_xlabel(f'{name1}', fontsize=20)
    ax.set_ylabel(f'{name2}', fontsize=20)
    #ax.set_title(f'{name1} vs {name2}', fontsize=15, pad=20, fontweight='bold')

    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Calculate and display correlation
    values1 = mouse_str_DF1.loc[common_idx, label1]
    values2 = mouse_str_DF2.loc[common_idx, label2]
    corr = stats.spearmanr(values1, values2)

    # Print correlation and p-value
    print(f"Spearman correlation: r = {corr.correlation:.2f}, p-value = {corr.pvalue:.0e}")

    ax.text(0.70, 0.1, f'r = {corr[0]:.2f}\np = {corr[1]:.0e}',
            transform=ax.transAxes,
            fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, linestyle='--', alpha=0.7)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_correlation_comparison_magmaGene_vs_Neural(disorders, ct_correlation_mat, gwas_correlation_mat, dpi=480):
    """
    Plot comparison between cell type association correlation and GWAS gene association correlation.
    
    Parameters:
    -----------
    disorders : list
        List of disorder names to compare
    ct_correlation_mat : pd.DataFrame
        Cell type bias correlation matrix
    gwas_correlation_mat : pd.DataFrame
        GWAS gene association correlation matrix
    dpi : int
        Resolution for the plot
    """
    # Set figure style and size 
    plt.style.use('seaborn-v0_8-whitegrid')  # Use 'seaborn' instead of deprecated 'seaborn-whitegrid'
    plt.figure(figsize=(8, 5), dpi=dpi)

    # Calculate Spearman correlation
    x_values = []
    y_values = []
    annotations = []
    for i, disorder1 in enumerate(disorders):
        for disorder2 in disorders[i+1:]:
            corr = ct_correlation_mat.loc[disorder1, disorder2]
            x_values.append(corr)
            y_values.append(gwas_correlation_mat.loc[disorder1, disorder2])
            if corr > 0.6:
                annotations.append((corr, gwas_correlation_mat.loc[disorder1, disorder2], f"{disorder1}-{disorder2}"))

    spearman_corr, p_val = stats.spearmanr(x_values, y_values)

    # Get all unique disorder pairs and plot
    for i, disorder1 in enumerate(disorders):
        for disorder2 in disorders[i+1:]:
            plt.scatter(ct_correlation_mat.loc[disorder1, disorder2],
                       gwas_correlation_mat.loc[disorder1, disorder2],
                       alpha=0.7,
                       s=80,
                       edgecolor='black',
                       linewidth=0.5)

    # Add annotations for pairs with correlation > 0.8 using adjustText
    texts = []
    for x, y, label in annotations:
        texts.append(plt.text(x, y, label, fontsize=8))

    # Use adjustText to prevent overlapping labels
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

    # Add Spearman correlation text to plot
    plt.text(0.05, 0.85, f'Spearman r = {spearman_corr:.3f}\np_val = {p_val:.1e}',
             transform=plt.gca().transAxes,
             fontsize=12,
             fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Customize plot appearance
    plt.xlabel("Cell Type Association Correlation", fontsize=15, fontweight='bold')
    plt.ylabel("GWAS Gene Association Correlation", fontsize=15, fontweight='bold')
    #plt.title("Cell Type Bias vs Gene level association \nAcross Psychiatric Disorders",
    #         fontsize=14, fontweight='bold', pad=15)

    # Add grid and adjust spines
    plt.grid(True, linestyle='--', alpha=0.7)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        
    plt.tight_layout()
    plt.show()

# same, if give a ax plot on ax stay same if no ax given
def plot_cumulative_variance(pc_variance_df, ax=None, dpi=300):
    """
    Plot cumulative explained variance for different datasets at publication quality.

    Args:
        pc_variance_df: DataFrame with explained variance ratios for different datasets
        ax: matplotlib.axes.Axes, optional
        dpi: Resolution for the plot (default: 300 for publication quality)
    """
    import matplotlib as mpl

    # Nature-style settings
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "axes.titlesize": 22,
        "axes.linewidth": 2,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "legend.fontsize": 16,
        "legend.frameon": False,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    # Set up figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    else:
        fig = ax.figure

    # Calculate cumulative explained variance
    cumulative_variance = pd.DataFrame()
    for col in pc_variance_df.columns:
        zeros = pd.Series([0], index=['PC0'])
        col_series = pc_variance_df[col].reset_index(drop=True)
        col_series.index = [f'PC{i+1}' for i in range(len(col_series))]
        cumsum = pd.concat([zeros, col_series.cumsum()])
        cumulative_variance[col] = cumsum

    # Plot settings
    colors = {
        'HumanCT': '#009E73',   # Nature green
        'MouseCT': '#D55E00',   # Nature orange
        'MouseSTR': '#0072B2',  # Nature blue
        'GeneZstat': '#666666'  # Neutral gray
    }
    markers = {
        'HumanCT': 'o',
        'MouseCT': 's',
        'MouseSTR': 'D',
        'GeneZstat': '^'
    }
    labels = {
        'HumanCT': 'Human Cell Type',
        'MouseCT': 'Mouse Cell Type',
        'MouseSTR': 'Mouse Structure',
        'GeneZstat': 'Gene Level'
    }

    x_range = range(len(cumulative_variance))
    x_labels = ['0'] + [f'{i+1}' for i in range(len(cumulative_variance)-1)]

    # Plot each dataset with distinct marker and color
    for key in ['GeneZstat', 'HumanCT', 'MouseCT', 'MouseSTR']:
        if key in cumulative_variance.columns:
            ax.plot(
                x_range,
                cumulative_variance[key],
                marker=markers[key],
                markersize=8,
                linewidth=2.5,
                color=colors[key],
                label=labels[key],
                alpha=0.95
            )

    # Axis labels and ticks
    ax.set_xlabel('Principal Component', fontsize=20, fontweight='bold', labelpad=10)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=20, fontweight='bold', labelpad=10)
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{y:.1f}" for y in np.linspace(0, 1, 6)], fontsize=16)

    # Remove top and right spines, thicken left/bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Add grid behind
    ax.grid(True, axis='y', linestyle='--', linewidth=1, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Add legend with no frame, outside plot
    ax.legend(
        loc='lower right',
        fontsize=16,
        frameon=False,
        handletextpad=0.5,
        borderaxespad=0.5,
        labelspacing=0.4
    )

    # Tight layout for publication
    fig.tight_layout(pad=1.5)

    # Optionally show if no ax provided
    if ax is None:
        plt.show()

def plot_pc_loadings_consistency(loadings_all, pc_component='PC1', Method="top_gene_enrich", Exclude="Empty", dpi=80):
    """
    Plot disorder loadings for a specific PC component across datasets.
    
    Parameters:
    pc_component (str): The principal component to plot (e.g., 'PC1', 'PC2')
    dpi (int): Resolution of the plot (default: 80)
    """
    # Create a figure for the specified PC
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi, facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Colors for each dataset
    colors = {'HumanCT': '#2ecc71', 'MouseCT': '#e74c3c', 'MouseSTR': '#3498db'}

    # Create legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=dataset)
                     for dataset, color in colors.items()]

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'HumanCT': loadings_all["PGC"]["HumanCT"][Method][Exclude][pc_component],
        'MouseCT': loadings_all["PGC"]["MouseCT"][Method][Exclude][pc_component],
        'MouseSTR': loadings_all["PGC"]["MouseSTR"][Method][Exclude][pc_component]
    })

    # Sort by absolute values of HumanCT loadings
    plot_data = plot_data.reindex(plot_data['HumanCT'].abs().sort_values(ascending=False).index)

    # Plot each dataset
    for dataset, color in colors.items():
        ax.scatter(range(len(plot_data)), plot_data[dataset],
                  alpha=0.9, color=color, s=200, marker='o', edgecolor='black', linewidth=1.5)
        
        # # Add trend line
        # z = np.polyfit(range(len(plot_data)), plot_data[dataset], 1)
        # p = np.poly1d(z)
        # ax.plot(range(len(plot_data)), p(range(len(plot_data))), 
        #         '--', color=color, alpha=0.9, linewidth=2)

    # Calculate correlations
    corr_hct_mct = stats.spearmanr(plot_data['HumanCT'], plot_data['MouseCT'])[0]
    corr_hct_mstr = stats.spearmanr(plot_data['HumanCT'], plot_data['MouseSTR'])[0]
    corr_mct_mstr = stats.spearmanr(plot_data['MouseCT'], plot_data['MouseSTR'])[0]

    # Add correlation text
    corr_text = f'SpearmanCorrelations:\nHCT-MCT: {corr_hct_mct:.2f}\nHCT-MSTR: {corr_hct_mstr:.2f}\nMCT-MSTR: {corr_mct_mstr:.2f}'
    ax.text(0.95, 0.95, corr_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
        fontsize=15)

    # Customize plot
    #ax.set_title(pc_component, fontsize=18, pad=30)
    ax.set_ylabel(f'Disorder Loadings ({pc_component})', fontsize=18, fontweight='bold')

    # Add grid with higher visibility
    ax.grid(True, linestyle='--', alpha=1.0, color='gray')

    # Set x-axis ticks
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=25, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Add legend
    #plt.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=14)
    plt.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.05, 0.2), fontsize=14)

    plt.tight_layout()
    plt.show()