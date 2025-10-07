#from ASD_Circuits import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy import stats
from scipy.stats import hypergeom
from scipy.stats import fisher_exact
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection, multipletests
import math
import pickle as pk
import csv

# Load some common used variables
#HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()


def LoadGeneINFO():
    #HGNC = pd.read_csv("../dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    HGNC = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    ENSID2Entrez = dict(zip(HGNC["ensembl_gene_id"].values, HGNC["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC["symbol"].values, HGNC["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC["entrez_id"].values, HGNC["symbol"].values))
    #allen_mouse_genes = loadgenelist("/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol 


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

def List2Fil(List, filename):
    fout = open(filename, 'wt')
    for x in List:
        fout.write(str(x)+"\n")

def LoadList(filename):
    fin = open(filename, 'rt')
    return [x.strip() for x in fin.readlines()]

def Dict2Fil(dict_, fil_):
    with open(fil_, 'wt') as f:
        writer = csv.writer(f)
        for k,v in dict_.items():
            writer.writerow([k, v])

def Fil2Dict(fil_):
    df = pd.read_csv(fil_, header=None)
    return dict(zip(df[0].values, df[1].values))

def Aggregate_Gene_Weights_NDD(MutFil, usepLI=False, Bmis=False, out=None):
    gene2MutN = {}
    for i, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
        except:
            print(g, "Error converting Entrez ID")

        nLGD = row["frameshift_variant"] + row["splice_acceptor_variant"] + row["splice_donor_variant"] + row["stop_gained"] + row["stop_lost"] 
        nMis = row["missense_variant"] 

        gene2MutN[g] = nLGD * 0.347 + nMis * 0.194
    if out != None:
        writer = csv.writer(open(out, 'wt'))
        for k,v in sorted(gene2MutN.items(), key=lambda x:x[1], reverse=True):
           writer.writerow([k,v]) 
    return gene2MutN

def quantileNormalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : np.sort(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df
    
def ZscoreConverting_V2(values, mean=np.nan, std=np.nan, low_exp = 0, min_z=-5): 
    """
    Convert values to z-scores with special handling for zeros:
    - Build distribution using only non-zero values
    - Set minimum z-score to min_z (default -5)
    - Set all zero expressions to min_z (default -5)
    
    Args:
        values: Array-like input values
        mean: Optional pre-computed mean
        std: Optional pre-computed standard deviation
        min_z: Minimum z-score (default -5)
    Returns:
        numpy array of z-scores
    """
    # Convert to numpy array and identify non-zero values
    values = np.array(values)
    non_zero_mask = values >= low_exp
    non_zero_values = values[non_zero_mask]
    
    # If no non-zero values, return array of -5
    if len(non_zero_values) == 0:
        return np.full_like(values, min_z)
    
    # Calculate mean and std from non-zero values if not provided
    if mean != mean:  # Check for nan
        mean = np.mean(non_zero_values)
    if std != std:    # Check for nan
        std = np.std(non_zero_values)
        # Handle case where std is 0
        if std == 0:
            std = 1
    
    # Calculate z-scores
    zscores = np.full_like(values, min_z)  # Initialize with -5
    non_zero_zscores = (non_zero_values - mean) / std
    
    # Apply minimum threshold 
    non_zero_zscores = np.maximum(non_zero_zscores, min_z)
    
    # Put non-zero z-scores back in original positions
    zscores[non_zero_mask] = non_zero_zscores
    
    return zscores

def Z1Conversion(ExpMat, outname="test.z1.mat"):
    Z_mat = []
    for g, row in ExpMat.iterrows():
        tmp = ZscoreConverting(row.values)
        Z_mat.append(tmp)
    Z_mat = np.array(Z_mat)
    CT_Z1_DF = pd.DataFrame(data=Z_mat, index=ExpMat.index.values, 
                            columns=ExpMat.columns.values)
    CT_Z1_DF.to_csv(outname)
    return CT_Z1_DF

def Z1Conversion_V2(ExpMat, outname="test.z1.mat", low_exp = 0, min_z=-5):
    """
    Convert expression matrix to z-scores with zero handling
    
    Args:
        ExpMat: pandas DataFrame with genes as rows and cell types as columns
        outname: output file name for saving results
        
    Returns:
        pandas DataFrame with z-scores
    """
    Z_mat = []
    for g, row in ExpMat.iterrows():
        tmp = ZscoreConverting_V2(row.values, low_exp=low_exp, min_z=min_z)
        Z_mat.append(tmp)
    
    Z_mat = np.array(Z_mat)
    CT_Z1_DF = pd.DataFrame(data=Z_mat, 
                           index=ExpMat.index.values,
                           columns=ExpMat.columns.values)
    
    CT_Z1_DF.to_csv(outname)
    return CT_Z1_DF

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
        # Weighted average for this cell type
        weighted_avg = np.average(expr_values, weights=weights)
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
        # Weighted average for this cell type
        weighted_avg = np.average(expr_values, weights=weights)
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

def plot_correlation(values1, values2, name1, name2, title="", xlim=None, dpi=360):
    # Calculate correlations
    corr_all = stats.spearmanr(values1, values2)[0]
    tmp_Neur_idx = [i for i in Neur_idx if i in values1]
    tmp_NonNeur_idx = [i for i in NonNeur_idx if i in values1]
    corr_neur = stats.spearmanr(values1[tmp_Neur_idx], values2[tmp_Neur_idx])[0]
    corr_nonneur = stats.spearmanr(values1[tmp_NonNeur_idx], values2[tmp_NonNeur_idx])[0]
    

    plt.figure(figsize=(6,6), dpi = dpi)
    plt.scatter(values1[tmp_Neur_idx], values2[tmp_Neur_idx],
            color="red", alpha=0.6, s=80, label="Neuronal")
    plt.scatter(values1[tmp_NonNeur_idx], values2[tmp_NonNeur_idx], 
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

def GetSingeCellBiasCorr(Bias1, Bias2, name1="1", name2="2", efflabel="EFFECT", CTs=None):
    res = Bias1.join(Bias2, how = 'inner', lsuffix="_{}".format(name1), rsuffix="_{}".format(name2))
    res["Diff"] = res["{}_{}".format(efflabel, name1)] - res["{}_{}".format(efflabel, name2)]
    if CTs is not None:
        res = res.loc[CTs, :]
    X = res["{}_{}".format(efflabel, name1)].values
    Y = res["{}_{}".format(efflabel, name2)].values
    r, p = stats.spearmanr(X, Y)
    return r,p


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

def Enrichment(myset, totalset, Ntotal = 20000, method="hypergeom"):
    N = len(myset)
    n = len(myset.intersection(totalset))
    M = Ntotal #20996 #18454 #20000
    m = len(totalset)
    #print(n, N, m, M)
    Rate = (n*M)/(N*m)
    if method == "hypergeom":
        Pvalue = 1 - hypergeom.cdf(n-1, M, m, N)
    elif method == "binom":
        # your code here
        Pvalue = binom_test(n, N, p=m/M)
    elif method == "fisher_exact":
        # your code here
        Odds, Pvalue = fisher_exact([[M-m, m], [N-n, n]], alternative="greater")
    else:
        raise "Unimplemented Error"
    return Rate, Pvalue

def LoadHumanCTAnno():
    Anno = pd.read_excel("/home/jw3514/Work/data/HumanBrainCellType/annotation.xlsx", index_col="Cluster")
    Anno.drop(Anno.tail(1).index,inplace=True) # drop last n rows
    Anno.fillna('', inplace=True)
    Anno.index = [int(x) for x in Anno.index.values]
    return Anno

def AnnoteSubcluster(df, Anno):
    for i, row in df.iterrows():
        idx = int(i.split("-")[0])
        df.loc[i, "Class"] = Anno.loc[idx, "Class"]
        df.loc[i, "Supercluster"] = Anno.loc[idx, "Supercluster"]
        df.loc[i, "Neurotransmitter"] = Anno.loc[idx, "Neurotransmitter"]
        df.loc[i, "Neuropeptide"] = Anno.loc[idx, "Neuropeptide"] #Top three dissections
        df.loc[i, "Top regions"] = Anno.loc[idx, "Top ROIGroupFine"]
        df.loc[i, "Top structures"] = Anno.loc[idx, "Top ROI"]
        df.loc[i, "Number of cells"] = Anno.loc[idx, "Number of cells"] #Top three dissections
    return df

#################################################
# Human CT Bias Comparison Functions
#################################################
def CompareCT(Bias1, Bias2, name1="1", name2="2", effectlabel = "EFFECT", SuperClusters=ALL_CTs, xmin=0, xmax=0, savefig="", show_plot=False):
    res = Bias1.join(Bias2, how = 'inner', lsuffix="_{}".format(name1), rsuffix="_{}".format(name2))
    res["Diff"] = res[effectlabel + "_{}".format(name1)] - res[effectlabel + "_{}".format(name2)]
    #res.to_csv("./test/{}_{}_vs_{}.csv".format(name0, name1, name2))
    res = res[res["Supercluster_{}".format(name1)].isin(SuperClusters)]

    print(stats.spearmanr(res[effectlabel + "_{}".format(name1)].values, res[effectlabel + "_{}".format(name2)].values))
    #print(pearsonr(res["EFFECT_{}".format(name1)].values, res["EFFECT_{}".format(name2)].values))

    if not show_plot and savefig == "":
        return res, None

    idx = 0
    N_col = 4
    N_rows = math.ceil(len(SuperClusters)/N_col)
    print(len(SuperClusters))
    if len(SuperClusters) > 22:
        fig = plt.figure(figsize=(12, 20), constrained_layout=False, dpi=480)
    else:
        fig = plt.figure(figsize=(12, 16), constrained_layout=False, dpi=480)
    spec = fig.add_gridspec(ncols=N_col, nrows=N_rows)
    if xmin ==0 and xmax==0:
        xmin = min(min(res[effectlabel + "_{}".format(name1)].values), min(res[effectlabel + "_{}".format(name2)].values)) * 1.1
        xmax = max(max(res[effectlabel + "_{}".format(name1)].values), max(res[effectlabel + "_{}".format(name2)].values)) * 1.1
    for a in range(N_rows):
        for b in range(N_col):
            ax0 = fig.add_subplot(spec[a, b])
            tmp = res[res["Supercluster_{}".format(name1)]==SuperClusters[idx]]
            ax0.scatter(tmp[effectlabel + "_{}".format(name1)].values, tmp[effectlabel + "_{}".format(name2)].values, s=10, )
            ax0.set_title(SuperClusters[idx], fontsize=12)
            ax0.plot([xmin,xmax],[xmin,xmax], color="grey", ls="dashed")
            ax0.plot([0,0],[xmin,xmax], color="grey", ls="dotted")
            ax0.plot([xmin,xmax],[0,0], color="grey", ls="dotted")
            ax0.set_xlim((xmin, xmax))
            ax0.set_xlim((xmin, xmax))
            ax0.tick_params(axis='both', which='major', labelsize=12)
            #ax0.set_ylim((0, 5e-3))
            if idx >= len(SuperClusters)-1:
                break
            idx += 1
    fig.text(0.5, -0.01, '{} Bias'.format(name1), ha='center', fontsize=15)
    fig.text(-0.02, 0.5, '{} Bias'.format(name2), va='center', rotation='vertical', fontsize=15)
    #fig.suptitle(name0)
    fig.tight_layout()
    if savefig != "":
        plt.savefig(savefig, bbox_inches="tight")
    if show_plot:
        plt.show()
    return res, fig


def format_pval_scientific(p):
    """Format p-value as 10^{x} notation for plot text."""
    print("????",p)
    if p == 0 or np.isnan(p):
        return "0"
    exp = int(math.floor(np.log10(p)))
    base = p / (10 ** exp)
    if p >= 0.99:
        return r"$P = 1$"
    elif p > 0.05:
        return r"$P = {:.2f}$".format(p)
    elif p < 1e-10:
        return r"$ P < 10^{-10}$"
    elif p < 0.05:
        return r"$P = {:.0f} \times 10^{{{}}}$".format(base, exp)
    else:
        return r"$P = {:.0f}$".format(p)

def PlotBiasContrast(DF, label1, label2, name1, name2, title="", neur_only=False):

    fig, ax = plt.subplots(dpi=300, figsize=(6, 5),  facecolor='none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Defensive: handle empty DF
    if DF.empty or label1 not in DF.columns or label2 not in DF.columns:
        print("Warning: DataFrame is empty or missing required columns.")
        plt.close(fig)
        return

    X_all = DF[label1].values
    Y_all = DF[label2].values

    # Defensive: handle empty arrays for correlation
    if len(X_all) < 2 or len(Y_all) < 2:
        r_all, p_all = np.nan, np.nan
    else:
        try:
            r_all, p_all = stats.spearmanr(X_all, Y_all)
        except Exception as e:
            print(f"Warning: Could not compute Spearman correlation for all data: {e}")
            r_all, p_all = np.nan, np.nan

    xmin = np.min(X_all) if len(X_all) > 0 else 0
    xmax = np.max(X_all) if len(X_all) > 0 else 1
    ymin = np.min(Y_all) if len(Y_all) > 0 else 0
    ymax = np.max(Y_all) if len(Y_all) > 0 else 1

    # Set axis limits to be equal and cover full data range
    plot_min = min(xmin, ymin)
    plot_max = max(xmax, ymax)

    tmp_Neur_idx = [i for i in Neur_idx if i in DF.index]
    tmp_NonNeur_idx = [i for i in NonNeur_idx if i in DF.index]

    # Plot neurons first
    DF_neur = DF.loc[tmp_Neur_idx, :] if len(tmp_Neur_idx) > 0 else DF.iloc[[]]
    X = DF_neur[label1].values if label1 in DF_neur.columns else np.array([])
    Y = DF_neur[label2].values if label2 in DF_neur.columns else np.array([])

    # Defensive: handle empty arrays for correlation
    if len(X) < 2 or len(Y) < 2:
        r_neur, p_neur = np.nan, np.nan
    else:
        try:
            r_neur, p_neur = stats.spearmanr(X, Y)
        except Exception as e:
            print(f"Warning: Could not compute Spearman correlation for neurons: {e}")
            r_neur, p_neur = np.nan, np.nan

    if len(X) > 0 and len(Y) > 0:
        ax.scatter(X, Y, s=40, lw=2, facecolor="none", edgecolor='darkblue', alpha=0.7, label="Neuron")
        # Add linear fit line for neurons
        if len(X) >= 2 and len(Y) >= 2:
            try:
                z_neur = np.polyfit(X, Y, 1)
                p_neur_line = np.poly1d(z_neur)
                ax.plot(np.array([plot_min, plot_max]), p_neur_line(np.array([plot_min, plot_max])),
                        color="darkblue", linestyle="--", linewidth=1.5)
            except Exception as e:
                print(f"Warning: Could not fit line for neurons: {e}")
        # Show r and p-value (not the poly1d object), format p-value as 10^{x}
        ax.text(0.05, 0.95, f"$spearman R = {r_neur:.2f}$\n{format_pval_scientific(p_neur)}", transform=ax.transAxes, fontsize=15, ha='left', va='top')
    else:
        ax.text(0.05, 0.95, "No neuron data", transform=ax.transAxes, fontsize=15, ha='left', va='top')

    if not neur_only:
        # Plot non-neurons
        DF_nonneur = DF.loc[tmp_NonNeur_idx, :] if len(tmp_NonNeur_idx) > 0 else DF.iloc[[]]
        X = DF_nonneur[label1].values if label1 in DF_nonneur.columns else np.array([])
        Y = DF_nonneur[label2].values if label2 in DF_nonneur.columns else np.array([])

        if len(X) > 0 and len(Y) > 0:
            ax.scatter(X, Y, s=40, lw=2, facecolor="none", edgecolor='darkred', alpha=0.7, label="Non-Neuron")
            # Add linear fit line for non-neurons
            if len(X) >= 2 and len(Y) >= 2:
                try:
                    z_nonneur = np.polyfit(X, Y, 1)
                    p_nonneur = np.poly1d(z_nonneur)
                    ax.plot(np.array([plot_min, plot_max]), p_nonneur(np.array([plot_min, plot_max])),
                            color="darkred", linestyle="--", linewidth=1.5)
                except Exception as e:
                    print(f"Warning: Could not fit line for non-neurons: {e}")
            ax.legend(fontsize=12, frameon=True, edgecolor='black', facecolor='white', framealpha=1.0, loc="lower right")
        else:
            # No non-neuron data
            pass

    # Add correlation statistics text
    # Position text in empty quadrants to avoid overlapping with data points
    if len(X_all) > 0 and len(Y_all) > 0:
        data_quadrants = np.sign(np.vstack([X_all, Y_all]))
        points_in_q1 = np.sum((data_quadrants[0] > 0) & (data_quadrants[1] > 0))
        points_in_q3 = np.sum((data_quadrants[0] < 0) & (data_quadrants[1] < 0))
    else:
        points_in_q1 = points_in_q3 = 0

    # Choose text position based on which quadrant has fewer points
    if points_in_q1 <= points_in_q3:
        text_x = plot_max * 0.5  # Upper right quadrant
        text_y = plot_min * 0.5
        va = 'top'
    else:
        text_x = plot_min * 0.5  # Lower left quadrant
        text_y = plot_max * 0.5
        va = 'bottom'

    # Print correlation statistics, format p-values as 10^{x}
    if neur_only:
        print(f"Neuron Types:\nSpearmanR = {r_neur if not np.isnan(r_neur) else 'NA'}\nP = {format_pval_scientific(p_neur) if not np.isnan(p_neur) else 'NA'}")
    else:
        print(f"All Cell Types:\nSpearmanR = {r_all if not np.isnan(r_all) else 'NA'}\nP = {format_pval_scientific(p_all) if not np.isnan(p_all) else 'NA'}\n\n"
              f"Neuron Types:\nSpearmanR = {r_neur if not np.isnan(r_neur) else 'NA'}\nP = {format_pval_scientific(p_neur) if not np.isnan(p_neur) else 'NA'}")

    # Style the plot
    ax.set_xlabel(name1, fontsize=18)
    ax.set_ylabel(name2, fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Make left and bottom spines darker
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()


def combineBias(Bias1, Bias2, name1="1", name2 = "2"):
    res = Bias1.join(Bias2, how = 'inner', lsuffix="_{}".format(name1), rsuffix="_{}".format(name2))
    res["Diff"] = res["EFFECT_{}".format(name1)] - res["EFFECT_{}".format(name2)]
    return res

def compare_biases(bias1, bias2, name1="1", name2="2", efflabel="EFFECT", neurons=None):
    """
    Compare two bias datasets and calculate statistics for each cell type.
    
    Args:
        bias1: First bias DataFrame
        bias2: Second bias DataFrame 
        name1: Name/label for first bias dataset (default: "1")
        name2: Name/label for second bias dataset (default: "2")
        neurons: List of neuron types to analyze (default: None)
        
    Returns:
        DataFrame with comparison statistics for each cell type
    """
    BiasDF_cb = combineBias(bias1, bias2, name1=name1, name2=name2)

    # Create list to store results
    results_list = []

    #for CT in neurons:
    for CT in ALL_CTs:
        # Filter data for current CT using proper column name
        CT_mask = BiasDF_cb[f"Supercluster_{name1}"] == CT
        CT_data = BiasDF_cb[CT_mask]
        
        # Get bias values, checking for empty results
        CT_Bias1 = CT_data[f"{efflabel}_{name1}"].values
        CT_Bias2 = CT_data[f"{efflabel}_{name2}"].values
        stm_1 = np.std(CT_Bias1) / np.sqrt(len(CT_Bias1))
        stm_2 = np.std(CT_Bias2) / np.sqrt(len(CT_Bias2))
        
        if len(CT_Bias1) == 0 or len(CT_Bias2) == 0:
            print(f"Warning: No data found for {CT}")
            continue
            
        # Calculate statistics
        bias = np.mean(CT_Bias1) - np.mean(CT_Bias2)
        
        try:
            t_man, p_man = stats.mannwhitneyu(CT_Bias1, CT_Bias2)
            t_wil, p_wil = stats.wilcoxon(CT_Bias1, CT_Bias2)
        except Exception as e:
            print(f"Error calculating statistics for {CT}: {str(e)}")
            continue
        
        # Add results to list with proper string formatting
        results_list.append({
            'Supercluster': CT,
            f'Bias_{name1}': np.mean(CT_Bias1),
            f'Bias_{name2}': np.mean(CT_Bias2), 
            f'STM_{name1}': stm_1,
            f'STM_{name2}': stm_2,
            'Bias_Diff': bias,
            'Mann_Whitney_P': p_man,
            'Wilcoxon_P': p_wil
        })

    # Convert list to DataFrame and sort
    results = pd.DataFrame(results_list)
    if len(results) > 0:
        results = results.sort_values(by="Bias_Diff")
    else:
        print("Warning: No results generated")
    
    # Perform FDR correction on p-values
    results['Mann_Whitney_FDR'] = multipletests(results['Mann_Whitney_P'], method='fdr_bh')[1]
    results['Wilcoxon_FDR'] = multipletests(results['Wilcoxon_P'], method='fdr_bh')[1]
    #results['Bonferroni_P'] = (results['Mann_Whitney_P'] * results.shape[0]).clip(upper=1.0)
    results['Bonferroni_P'] = (results['Wilcoxon_P'] * results.shape[0]).clip(upper=1.0)
    #results['Bonferroni_P'] = min(results['Mann_Whitney_P'] * n_tests, 1.0)

    results = results.set_index("Supercluster")
        
    return results   

def plot_bias_comparison(results, name1, name2, p_test="Wilcoxon_P", legend_anchor=(1, 0)):
    # Create publication-quality figure
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300, facecolor='none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Filter results
    results_to_plot = results[(results[f'Bias_{name1}'] > 0) | (results[f'Bias_{name2}'] > 0)]
    x = np.arange(len(results_to_plot))
    width = 0.35

    # Create bars with refined colors and error bars
    bar1 = ax.bar(x - width/2, results_to_plot[f'Bias_{name1}'], width, 
                  label=name1, color='#3182bd', alpha=0.8,
                  yerr=results_to_plot[f'STM_{name1}'], 
                  capsize=4, error_kw={'ecolor': '#08519c', 'capthick': 1.5})

    bar2 = ax.bar(x + width/2, results_to_plot[f'Bias_{name2}'], width,
                  label=name2, color='#e6550d', alpha=0.8,
                  yerr=results_to_plot[f'STM_{name2}'],
                  capsize=4, error_kw={'ecolor': '#a63603', 'capthick': 1.5})

    # Customize plot appearance
    ax.set_ylabel('Mutation Bias', fontsize=30, )
    ax.set_xticks(x)
    ax.set_xticklabels(results_to_plot.index.values, rotation=45, ha='right', 
                       fontsize=21, fontweight='medium')

    # Add refined legend
    ax.legend(fontsize=21, frameon=False,
             loc='lower right', bbox_to_anchor=legend_anchor)

    # Add zero line and grid
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=21, width=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')

    # Add significance markers with corrected vertical positioning
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    y_range = y_max - y_min

    for i, (idx, row) in enumerate(results_to_plot.iterrows()):
        p_val = row[p_test]

        if p_val < 0.0001:
            marker = '****'
        elif p_val < 0.001:
            marker = '***'
        elif p_val < 0.01:
            marker = '**'
        elif p_val < 0.05:
            marker = '*'
        else:
            marker = ''
            
        # Calculate maximum height of bars and error bars at this position
        bar1_height = row[f'Bias_{name1}'] + row[f'STM_{name1}']
        bar2_height = row[f'Bias_{name2}'] + row[f'STM_{name2}']
        max_height = max(bar1_height, bar2_height)
        
        # Add marker slightly above the highest point
        y_offset = y_range * 0.05  # 5% of the y-axis range
        ax.text(i, max_height + y_offset, marker, 
                ha='center', va='bottom', fontsize=22)

    # Adjust y-axis limits to accommodate markers
    ax.set_ylim(y_min*1.5, y_max + y_range * 0.1)  # Add 10% padding at top

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def PlotBiasContrast_Diff(X, Y, label1, label2, ContrastDF, title="",pval="Mann_Whitney_FDR", loc=1, text_x=None, text_y=None):
    fig, ax = plt.subplots(dpi=150, figsize=(4, 4), facecolor='none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.style.use('seaborn-v0_8-whitegrid')
    scatter = ax.scatter(X, Y, s=40, color='dodgerblue', alpha=0.6, edgecolor='k')

    # Find the absolute min and max across both X and Y
    data_min = min(min(X), min(Y)) 
    data_max = max(max(X), max(Y)) 
    
    # Add some padding
    padding = 0.01
    plot_min = data_min - padding
    plot_max = data_max + padding
    
    # Set equal limits for both axes
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    
    BiasDiff = np.mean(X) - np.mean(Y)
    #t_wil, p_wil = mannwhitneyu(X, Y)
    P_adj = ContrastDF.loc[title, pval]

    ax.plot([plot_min, plot_max], [plot_min, plot_max], color="grey", ls="dashed")
    
    if text_x is None:
        text_x = plot_min
    if text_y is None:
        text_y = plot_max

    # Location of text
    if loc == 1:
        text_x = plot_min + 0.01
        text_y = plot_max - 0.03
    elif loc == 2:
        text_x = plot_max * 0.6
        text_y = (plot_max - plot_min) * 0.5
    else:
        text_x = loc[0]
        text_y = loc[1]

    ax.text(text_x, text_y, 
            s=f"ΔBias={BiasDiff:.3f}\n{format_pval_scientific(P_adj)}",
            fontsize=13, ha='left', va='top')
    
    ax.set_title(title, fontsize=18, weight='normal', pad=10)
    ax.set_xlabel(label1, fontsize=16, weight='normal')
    ax.set_ylabel(label2, fontsize=16, weight='normal')
    ax.set_facecolor('none')
    # Increase font size of X and Y ticks
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def CompareSingleCT(Bias1, Bias2, CT, ContrastDF, name1="1", name2="2", efflabel="EFFECT", title="", pval="Mann_Whitney_FDR", loc=1, text_x=None, text_y=None):
    res = Bias1.join(Bias2, how='inner', lsuffix=f"_{name1}", rsuffix=f"_{name2}")
    res["Diff"] = res[f"{efflabel}_{name1}"] - res[f"{efflabel}_{name2}"]
    res = res[res[f"Supercluster_{name1}"]==CT]
    X = res[f"{efflabel}_{name1}"].values
    Y = res[f"{efflabel}_{name2}"].values
    PlotBiasContrast_Diff(X, Y, name1, name2, ContrastDF, title=CT, pval=pval, loc=loc, text_x=text_x, text_y=text_y)


def plot_mutation_bias_comparison(CT, datasets, anno_df, TestPairs = []):
    """
    Create a comparative plot of mutation bias across multiple datasets for a given cell type.
    
    Parameters:
    -----------
    CT : str
        Cell type name to plot
    datasets : dict
        Dictionary of dataset name -> bias dataframe pairs
    anno_df : pandas DataFrame
        Annotation dataframe with cell type information
        
    Returns:
    --------
    None, displays plot
    """
    # Get cell type indices
    CT_idx = anno_df[anno_df["Supercluster"]==CT].index.values
    
    # Get data for each dataset
    data = {name: df.loc[CT_idx, "EFFECT"] for name, df in datasets.items()}

    # Calculate means and sort datasets
    means = {k: v.mean() for k,v in data.items()}
    sorted_datasets = sorted(data.keys(), key=lambda x: means[x])
    sorted_data = {k: data[k] for k in sorted_datasets}

    # Set style for Nature publication
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300, facecolor='none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 25  # Increased from 8
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5

    # Plot individual points and connect same cell types
    positions = range(1, len(sorted_data)+1)
    n_cells = len(CT_idx)

    # Create arrays to store coordinates
    x_coords = np.zeros((n_cells, len(sorted_data)))
    y_coords = np.zeros((n_cells, len(sorted_data)))

    # Plot points and store coordinates
    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_data)))
    for i, (pos, (label, values)) in enumerate(zip(positions, sorted_data.items())):
        x = np.random.normal(pos, 0.04, size=len(values))
        ax.scatter(x, values, color=colors[i], edgecolor="black", s=20, alpha=1, label=label)
        x_coords[:, i] = x
        y_coords[:, i] = values

    # Add boxplots with more prominent boxes
    bp = ax.boxplot(
        [v for v in sorted_data.values()],
        positions=positions,
        showfliers=False,
        patch_artist=True,
        widths=0.4,  # Make boxes wider
        boxprops=dict(facecolor='white', alpha=0, edgecolor='black', linewidth=1),
        medianprops=dict(color='grey', linewidth=1),
        whiskerprops=dict(color='grey', linewidth=1),
        capprops=dict(color='grey', linewidth=1)
    )

    # Further customize boxplots for prominence
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor='white', alpha=0.6)
        box.set(edgecolor='black', linewidth=1)
    plt.setp(bp['medians'], color='grey', linewidth=1)
    plt.setp(bp['whiskers'], color='grey', linewidth=1)
    plt.setp(bp['caps'], color='grey', linewidth=1)

    # Calculate and plot p-values for each test pair, with Bonferroni correction
    n_tests = 31 * 5
    global_y_max = max([max(v) for v in sorted_data.values()])
    y_offset = (global_y_max) * 0.05 if global_y_max != 0 else 0.05  # Offset for annotation

    annotation_heights = []
    min_sep = y_offset * 3  # Minimum vertical separation between annotations

    for idx, (DisorderA, DisorderB) in enumerate(TestPairs):
        DataA = datasets[DisorderA]
        DataB = datasets[DisorderB]
        stat, pval = scipy.stats.mannwhitneyu(DataA.loc[CT_idx, "EFFECT"], 
                                              DataB.loc[CT_idx, "EFFECT"])
        pval_corr = min(pval * n_tests, 1.0)
        print("???",pval_corr)
        print(f"{DisorderA} vs {DisorderB}: {format_pval_scientific(pval_corr)}")

        disorder_labels = list(sorted_data.keys())
        x1 = disorder_labels.index(DisorderA) + 1
        x2 = disorder_labels.index(DisorderB) + 1
        x_center = (x1 + x2) / 2

        y_base = max(max(data[DisorderA]), max(data[DisorderB]))
        if annotation_heights:
            y = max(max(annotation_heights) + min_sep, y_base + y_offset)
        else:
            y = y_base + y_offset

        # Store this annotation's height for future overlap checks
        annotation_heights.append(y + y_offset/2)

        # Draw a line and annotate p-value
        ax.plot([x1, x1, x2, x2], [y, y + y_offset/2, y + y_offset/2, y], lw=0.8, c='k', ls='--', alpha=0.7)
        ax.text(x_center, y + y_offset/2 + 0.01*global_y_max, f"{format_pval_scientific(pval_corr)}", ha='center', va='bottom', fontsize=10, backgroundcolor='none')

    # Optionally, adjust y-limits to fit all annotations
    if annotation_heights:
        ax.set_ylim(top=max(annotation_heights) + y_offset*2)

    # Customize plot
    #ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(range(1,len(sorted_data)+1))
    ax.set_xticklabels(sorted_data.keys(), rotation=45, ha='center', weight='normal', fontsize=15)  # Added bold
    ax.set_ylabel('Mutation Bias', labelpad=5, weight='normal', fontsize=15)  # Added bold
    ax.set_title(f'{CT}', pad=10, weight='normal', fontsize=15)  # Added bold

    # Adjust layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_alpha(0.9)
    ax.spines['bottom'].set_color('black') 
    ax.spines['bottom'].set_alpha(0.9)
    
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.show()

def plot_mutation_bias_comparison_V2(CT, datasets, anno_df, PvalDF, TestPairs = []):
    """
    Create a comparative plot of mutation bias across multiple datasets for a given cell type.
    
    Parameters:
    -----------
    CT : str
        Cell type name to plot
    datasets : dict
        Dictionary of dataset name -> bias dataframe pairs
    anno_df : pandas DataFrame
        Annotation dataframe with cell type information
        
    Returns:
    --------
    None, displays plot
    """
    # Get cell type indices
    CT_idx = anno_df[anno_df["Supercluster"]==CT].index.values
    
    # Get data for each dataset
    data = {name: df.loc[CT_idx, "EFFECT"] for name, df in datasets.items()}

    # Calculate means and sort datasets
    means = {k: v.mean() for k,v in data.items()}
    sorted_datasets = sorted(data.keys(), key=lambda x: means[x])
    sorted_data = {k: data[k] for k in sorted_datasets}

    # Set style for Nature publication (match previous function)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300, facecolor='none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Plot individual points and connect same cell types
    positions = range(1, len(sorted_data)+1)
    n_cells = len(CT_idx)

    # Create arrays to store coordinates
    x_coords = np.zeros((n_cells, len(sorted_data)))
    y_coords = np.zeros((n_cells, len(sorted_data)))

    # Plot points and store coordinates
    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_data)))
    for i, (pos, (label, values)) in enumerate(zip(positions, sorted_data.items())):
        x = np.random.normal(pos, 0.04, size=len(values))
        ax.scatter(x, values, color=colors[i], edgecolor="black", s=20, alpha=1, label=label)
        x_coords[:, i] = x
        y_coords[:, i] = values

    # Add boxplots
    bp = ax.boxplot(
        [v for v in sorted_data.values()],
        positions=positions,
        showfliers=False,
        patch_artist=True,
        widths=0.4,  # Make boxes wider
        boxprops=dict(facecolor='white', alpha=0, edgecolor='black', linewidth=1),
        medianprops=dict(color='grey', linewidth=1),
        whiskerprops=dict(color='grey', linewidth=1),
        capprops=dict(color='grey', linewidth=1)
    )

    # Further customize boxplots for prominence
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor='white', alpha=0.6)
        box.set(edgecolor='black', linewidth=1)
    plt.setp(bp['medians'], color='grey', linewidth=1)
    plt.setp(bp['whiskers'], color='grey', linewidth=1)
    plt.setp(bp['caps'], color='grey', linewidth=1)

    # Calculate and plot p-values for each test pair, with Bonferroni correction
    n_tests = 31 * 5
    global_y_max = max([max(v) for v in sorted_data.values()])
    y_offset = (global_y_max) * 0.05 if global_y_max != 0 else 0.05  # Offset for annotation

    # Improved annotation placement to avoid overlap and make better use of space
    annotation_heights = []
    min_sep = y_offset * 3  # Minimum vertical separation between annotations

    for idx, (DisorderA, DisorderB) in enumerate(TestPairs):
        DataA = datasets[DisorderA]
        DataB = datasets[DisorderB]
        stat, pval = scipy.stats.mannwhitneyu(DataA.loc[CT_idx, "EFFECT"], 
                                              DataB.loc[CT_idx, "EFFECT"])
        pval_corr = PvalDF[PvalDF["Pair"] == f"{DisorderA} - {DisorderB}"]
        pval_corr = pval_corr[pval_corr["SuperCluster"] == CT]["MWU_FDR"].values[0]
        print("???",pval_corr)
        print(f"{DisorderA} vs {DisorderB}: {format_pval_scientific(pval_corr)}")

        # Find x positions for the two disorders
        disorder_labels = list(sorted_data.keys())
        x1 = disorder_labels.index(DisorderA) + 1
        x2 = disorder_labels.index(DisorderB) + 1
        x_center = (x1 + x2) / 2

        # For this annotation, y_base is just above the max of the two compared groups
        y_base = max(max(data[DisorderA]), max(data[DisorderB]))
        if annotation_heights:
            y = max(max(annotation_heights) + min_sep, y_base + y_offset)
        else:
            y = y_base + y_offset

        # Store this annotation's height for future overlap checks
        annotation_heights.append(y + y_offset/2)

        # Draw a line and annotate p-value
        ax.plot([x1, x1, x2, x2], [y, y + y_offset/2, y + y_offset/2, y], lw=0.8, c='k', ls='--', alpha=0.7)
        ax.text(x_center, y + y_offset/2 + 0.01*global_y_max, f"{format_pval_scientific(pval_corr)}", ha='center', va='bottom', fontsize=10, backgroundcolor='none')

    # Optionally, adjust y-limits to fit all annotations
    if annotation_heights:
        ax.set_ylim(top=max(annotation_heights) + y_offset*2)

    # Customize plot (match previous function)
    ax.set_xticks(range(1,len(sorted_data)+1))
    ax.set_xticklabels(sorted_data.keys(), rotation=45, ha='center', weight='normal', fontsize=15)
    ax.set_ylabel('Mutation Bias', labelpad=5, weight='normal', fontsize=15)
    ax.set_title(f'{CT}', pad=10, weight='normal', fontsize=15)

    # Adjust layout and axes (match previous function)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_alpha(0.9)
    ax.spines['bottom'].set_color('black') 
    ax.spines['bottom'].set_alpha(0.9)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.show()

#################################################
# Allen SC data related Functions
#################################################
ABC_ALL_Class = ['01 IT-ET Glut', '02 NP-CT-L6b Glut', '03 OB-CR Glut',
       '04 DG-IMN Glut', '05 OB-IMN GABA', '06 CTX-CGE GABA',
       '07 CTX-MGE GABA', '08 CNU-MGE GABA', '09 CNU-LGE GABA',
       '11 CNU-HYa GABA', '10 LSX GABA', '12 HY GABA', '13 CNU-HYa Glut',
       '14 HY Glut', '15 HY Gnrh1 Glut', '16 HY MM Glut', '17 MH-LH Glut',
       '18 TH Glut', '19 MB Glut', '20 MB GABA', '21 MB Dopa',
       '22 MB-HB Sero', '23 P Glut', '24 MY Glut', '25 Pineal Glut',
       '26 P GABA', '27 MY GABA', '28 CB GABA', '29 CB Glut',
       '30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular',
       '34 Immune']
ABC_nonNEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']
def CompareCT_ABC(Bias1, Bias2, name1="1", name2="2", name0 = "",SuperClusters=ABC_ALL_Class, xmin=0, xmax=0, savefig=""):
    res = Bias1.join(Bias2, how = 'inner', lsuffix="_{}".format(name1), rsuffix="_{}".format(name2))
    res["Diff"] = res["EFFECT_{}".format(name1)] - res["EFFECT_{}".format(name2)]
    #res.to_csv("./test/{}_{}_vs_{}.csv".format(name0, name1, name2))
    res = res[res["class_id_label_{}".format(name1)].isin(SuperClusters)]
    
    columns_to_drop_na = ["EFFECT_{}".format(name1), "EFFECT_{}".format(name2)]
    res = res.dropna(subset=columns_to_drop_na)

    print(pearsonr(res["EFFECT_{}".format(name1)].values, res["EFFECT_{}".format(name2)].values))
    print(spearmanr(res["EFFECT_{}".format(name1)].values, res["EFFECT_{}".format(name2)].values))

    idx = 0
    N_col = 4
    N_rows = math.ceil(len(SuperClusters)/N_col)
    #print(len(SuperClusters))
    if len(SuperClusters) > 22:
        fig = plt.figure(figsize=(12, 20), constrained_layout=False)
    else:
        fig = plt.figure(figsize=(12, 16), constrained_layout=False)
    spec = fig.add_gridspec(ncols=N_col, nrows=N_rows)
    if xmin ==0 and xmax==0:
        xmin = min(min(res["EFFECT_{}".format(name1)].values), min(res["EFFECT_{}".format(name2)].values)) * 1.1
        xmax = max(max(res["EFFECT_{}".format(name1)].values), max(res["EFFECT_{}".format(name2)].values)) * 1.1
    for a in range(N_rows):
        for b in range(N_col):
            ax0 = fig.add_subplot(spec[a, b])
            tmp = res[res["class_id_label_{}".format(name1)]==SuperClusters[idx]]
            ax0.scatter(tmp["EFFECT_{}".format(name1)].values, tmp["EFFECT_{}".format(name2)].values, s=10, )
            ax0.set_title(SuperClusters[idx])
            ax0.plot([xmin,xmax],[xmin,xmax], color="grey", ls="dashed")
            ax0.plot([0,0],[xmin,xmax], color="grey", ls="dotted")
            ax0.plot([xmin,xmax],[0,0], color="grey", ls="dotted")
            ax0.set_xlim((xmin, xmax))
            ax0.set_ylim((xmin, xmax))
            if idx >= len(SuperClusters)-1:
                break
            idx += 1
    plt.tight_layout()
    fig.text(0.5, -0.01, '{} Bias'.format(name1), ha='center', fontsize=20)
    fig.text(-0.02, 0.5, '{} Bias'.format(name2), va='center', rotation='vertical', fontsize=20)
    if savefig != "":
        plt.savefig(savefig, bbox_inches="tight")


#################################################
# Go terms related Functions
#################################################
Go2Uniprot = pk.load(open("/home/jw3514/Work/CellType_Psy/dat3/Goterms/Go2Uniprot.pk", 'rb'))
Uniprot2Entrez = pk.load(open("/home/jw3514/Work/CellType_Psy/dat3/Goterms/Uniprot2Entrez.pk", 'rb'))

def GetALLGo(go, GoID):
    Root = go[GoID]
    all_go = Root.get_all_children()
    all_go.add(GoID)
    return all_go
def GetGeneOfGo2(go, GoID, Go2Uniprot=Go2Uniprot):
    goset = GetALLGo(go, GoID)
    Total_Genes = set([])
    for i, tmpgo in enumerate(goset):
        #print(i, tmpgo)
        if tmpgo in Go2Uniprot:
            geneset = set([Uniprot2Entrez.get(x, 0) for x in Go2Uniprot[tmpgo]])
            Total_Genes = Total_Genes.union(geneset)
    return Total_Genes

def CT_Specific_GoTerm_Intersect(CellType, BG_Genes, Z2Bias, CT_Goterm, go, Anno=Anno, topN=100):
    CT_Idx = Anno[Anno["Supercluster"]==CellType].index.values
    tmpmat = Z2Bias.loc[BG_Genes, CT_Idx]
    for g, row in tmpmat.iterrows():
        tmpmat.loc[g, "Mean"] = np.mean(row)
    tmpmat = tmpmat.sort_values("Mean", ascending=False)
    #tmp_genes = tmpmat[tmpmat["Mean"]>1.0].index.values
    #tmp_genes = tmpmat[(tmpmat["Mean"]>=0.5)&(tmpmat['Mean']<=1.0)].index.values
    tmp_genes = tmpmat[(tmpmat["Mean"]>=0.3)&(tmpmat['Mean']<=0.5)].index.values
    print(CellType, len(tmp_genes))
    print([Entrez2Symbol[x] for x in tmp_genes])
    RelatedGos = {}
    CT_Goterm = CT_Goterm.sort_values(CellType, ascending=False)
    for i, row in CT_Goterm.head(topN).iterrows():
        GoID = i
        GoName = row["GoName"]
        Rho = row[CellType]
        GoGenes = GetGeneOfGo2(go, GoID)

        #print(GoGenes)
        InterGenes = GoGenes.intersection(set(tmp_genes))
        InterSymbol = [Entrez2Symbol[x] for x in InterGenes]
        if len(InterGenes) > 0:
            print(GoID, GoName, Rho, InterSymbol)

def GetFDRCut(DF, FDR1=0.1, FDR2=0.05):
    tmp1 = DF[DF["q-value"]<=FDR1]
    tmp2 = DF[DF["q-value"]<=FDR2]
    try:
        return min(tmp1["-logP"].values), min(tmp2["-logP"].values)
    except:
        return 0, 0

def SuperClusterBias_BoxPlot(DF, title, NeuroOnly=False, sortby="mean", EffectCol="EFFECT", fdr_cut=0.05):
    """
    Create a boxplot of mutation bias by supercluster with individual data points.
    
    Args:
        DF: DataFrame containing EFFECT and Supercluster columns
        title: Title for the plot
        NeuroOnly: If True, only show neuronal cell types
        sortby: Sort by "mean" or "median" values
    """
    dat_Z2 = []
    mean_Z2 = []
    
    # Select cell types based on NeuroOnly parameter
    cell_types = Neurons if NeuroOnly else ALL_CTs
    
    # Collect data for each cell type
    for _CT in cell_types:
        tmp = DF[DF["Supercluster"] == _CT]
        dat_Z2.append(tmp[EffectCol].values)
        
        if sortby == "median":
            mean_Z2.append(np.median(tmp[EffectCol].values))
        elif sortby == "mean":
            mean_Z2.append(np.mean(tmp[EffectCol].values))
    
    mean_Z2 = np.array(mean_Z2)
    
    # Sort data by the mean/median values
    sort_idx = np.argsort(mean_Z2)
    show_dat_Z2 = [dat_Z2[x] for x in sort_idx]
    show_CTs = [cell_types[x] for x in sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(dpi=120, figsize=(8, 8), facecolor='none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set seaborn context for publication-quality plots
    sns.set_context("talk", font_scale=1.0)
    
    # Create boxplot without fill
    bp = ax.boxplot(show_dat_Z2, labels=show_CTs, vert=False, patch_artist=False,
                    boxprops=dict(color='blue', linewidth=1.5),
                    medianprops=dict(linestyle='-', linewidth=2.5, color='firebrick'),
                    meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'),
                    showmeans=True, widths=0.6)
    
    # Add individual data points with jitter
    colors = sns.color_palette("Set2", len(show_dat_Z2))
    for i, (data, color) in enumerate(zip(show_dat_Z2, colors)):
        # Add individual points with jitter
        y_pos = i + 1  # boxplot positions start at 1
        y_jitter = np.random.normal(y_pos, 0.04, size=len(data))
        ax.scatter(data, y_jitter, alpha=0.6, s=20, color=color, edgecolors='white', linewidth=0.5)
    
    # Customize plot
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel(f"{title} Mutation Bias", fontsize=14, fontweight='bold')
    #ax.set_ylabel("Supercluster", fontsize=14, fontweight='bold')
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=12)
    
    # Rotate y-axis labels if needed for better readability
    plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
    if EffectCol == "-logP":
        FDR_cut1, FDR_cut2 = GetFDRCut(DF, FDR1 = 0.1, FDR2 = 0.05)
        # add FDR_cut line to the plot
        #if FDR_cut1 != 0:
        #    ax.axvline(x=FDR_cut1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        if FDR_cut2 != 0:
            ax.axvline(x=FDR_cut2, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel(f"{title} -log10(P-value)", fontsize=14, fontweight='normal')
    else:
        ax.set_xlabel(f"{title} Mutation Bias", fontsize=14, fontweight='normal')
    
    plt.tight_layout()
    plt.show()

# MERFISH Related Functions
def FixSubiculum(DF):
    X = DF.loc["Subiculum_dorsal_part"]
    Y = DF.loc["Subiculum_ventral_part"]
    Z = [(X[0]+Y[0])/2, "Hippocampus", 214]
    DF.loc["Subiculum"] = Z
    DF = DF.drop(["Subiculum_dorsal_part", "Subiculum_ventral_part"])
    return DF


def GetExpQ(pvalues):
    sorted_pvalues = np.sort(pvalues)
    expected = np.linspace(0, 1, len(sorted_pvalues), endpoint=False)[1:]
    expected_quantiles = -np.log10(expected)
    observed_quantiles = -np.log10(sorted_pvalues[1:])
    return expected_quantiles, observed_quantiles


def add_class(BiasDF, ClusterAnn):
    for cluster, row in BiasDF.iterrows():
        BiasDF.loc[cluster, "class_id_label"] = ClusterAnn.loc[cluster, "class_id_label"]
        BiasDF.loc[cluster, "CCF_broad.freq"] = ClusterAnn.loc[cluster, "CCF_broad.freq"]
        BiasDF.loc[cluster, "CCF_acronym.freq"] = ClusterAnn.loc[cluster, "CCF_acronym.freq"]
        BiasDF.loc[cluster, "v3.size"] = ClusterAnn.loc[cluster, "v3.size"]
        BiasDF.loc[cluster, "v2.size"] = ClusterAnn.loc[cluster, "v2.size"]
    return BiasDF


####################################################
# Specificity Score Validation Functions
####################################################
def MergeBiasDF(Bias1, Bias2, name1="1", name2="2"):
    res = Bias1.join(Bias2, how = 'inner', lsuffix="_{}".format(name1), rsuffix="_{}".format(name2))
    res["Diff"] = res["EFFECT_{}".format(name1)] - res["EFFECT_{}".format(name2)]
    return res 
    #res.to_csv("./test/{}_{}_vs_{}.csv".format(name0, name1, name2))

def PlotBiasContrast_v2(MergeDF, name1, name2, dataset="Human", title=""):
    if dataset == "HumanCT":
        NEUR = MergeDF[MergeDF["Supercluster_{}".format(name1)].isin(Neurons)]
        NonNEUR = MergeDF[~MergeDF["Supercluster_{}".format(name1)].isin(Neurons)]
        X_NEUR, Y_NEUR = NEUR["EFFECT_{}".format(name1)], NEUR["EFFECT_{}".format(name2)]
        X_NonNEUR, Y_NonNEUR = NonNEUR["EFFECT_{}".format(name1)], NonNEUR["EFFECT_{}".format(name2)]
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5))

        ax.scatter(X_NEUR, Y_NEUR, s=40, color="blue", edgecolor='black', alpha=0.7)
        ax.scatter(X_NonNEUR, Y_NonNEUR, s=40, color="red", edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Spearman correlation
        r_Neur, p_Neur = spearmanr(X_NEUR, Y_NEUR)
        r_nonNeur, p_nonNeur = spearmanr(X_NonNEUR, Y_NonNEUR)
        
        xmin = np.min(X_NEUR)
        xmax = np.max(X_NEUR)
        ymax = np.max(Y_NEUR)
        # Adjust text position to avoid overlap
        text_x = xmin * 0.9 if r_Neur < 0 else xmin * 0.7
        text_y = ymax * 0.7
        
        ax.text(text_x, text_y, s=f"R_NEUR = {r_Neur:.2f}\nR_NonNEUR = {r_nonNeur:.2f}",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        
        ax.set_xlabel(name1, fontsize=12, fontweight='bold')
        ax.set_ylabel(name2, fontsize=12, fontweight='bold')
        
        # Style adjustments
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return
    elif dataset == "MouseCT":
        NEUR = MergeDF[~MergeDF["class_id_label_{}".format(name1)].isin(ABC_nonNEUR)]
        NonNEUR = MergeDF[MergeDF["class_id_label_{}".format(name1)].isin(ABC_nonNEUR)]
        X_NEUR, Y_NEUR = NEUR["EFFECT_{}".format(name1)], NEUR["EFFECT_{}".format(name2)]
        X_NonNEUR, Y_NonNEUR = NonNEUR["EFFECT_{}".format(name1)], NonNEUR["EFFECT_{}".format(name2)]
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5))

        ax.scatter(X_NEUR, Y_NEUR, s=40, color="blue", edgecolor='black', alpha=0.7)
        ax.scatter(X_NonNEUR, Y_NonNEUR, s=40, color="red", edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Spearman correlation
        r_Neur, p_Neur = spearmanr(X_NEUR, Y_NEUR)
        r_nonNeur, p_nonNeur = spearmanr(X_NonNEUR, Y_NonNEUR)
        
        xmin = np.min(X_NEUR)
        xmax = np.max(X_NEUR)
        ymax = np.max(Y_NEUR)
        # Adjust text position to avoid overlap
        text_x = xmin * 0.9 if r_Neur < 0 else xmin * 0.7
        text_y = ymax * 0.7
        
        ax.text(text_x, text_y, s=f"R_NEUR = {r_Neur:.2f}\nR_NonNEUR = {r_nonNeur:.2f}",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        
        ax.set_xlabel(name1, fontsize=12, fontweight='bold')
        ax.set_ylabel(name2, fontsize=12, fontweight='bold')
        
        # Style adjustments
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return

# Phenotype vs IQ Functions

def linear_fit(biases, IQs, alpha=0.05, fixed_intercept=None):
    if fixed_intercept is None:
        model = sm.OLS(IQs, sm.add_constant(biases))
        results = model.fit()
        intercept = results.params[0]
    else:
        # Fit model without intercept and add fixed intercept
        model = sm.OLS(IQs - fixed_intercept, biases)
        results = model.fit()
        intercept = fixed_intercept
    
    beta = results.params[0] if fixed_intercept is not None else results.params[1]
    # get CI of beta
    ci = results.conf_int(alpha=alpha)
    ci_low = ci[0][0] if fixed_intercept is not None else ci[1][0]
    ci_high = ci[0][1] if fixed_intercept is not None else ci[1][1]
    r_value = results.rsquared
    p_value = results.pvalues[0] if fixed_intercept is not None else results.pvalues[1]
    std_err = results.bse[0] if fixed_intercept is not None else results.bse[1]
    pho, p = spearmanr(biases, IQs)
    
    return intercept, beta, ci_low, ci_high, r_value, p_value, std_err, pho, p


def Plot_Bias_vs_IQ(STR, Mut_n_IQ_conf, BiasMat):
    biases, IQs = BiasVsPheno(Mut_n_IQ_conf, BiasMat , STR, 'XX')
    pho, p = spearmanr(biases, IQs)
    # Create the scatter plot
    plt.figure(dpi=150, figsize=(5, 4))

    # Plot data points
    plt.scatter(biases, IQs, s=50, color="#2c7bb6", edgecolor="black", alpha=0.8, zorder=10)

    # Fit and plot the trend line
    b, a = np.polyfit(biases, IQs, deg=1)
    xseq = np.linspace(min(biases), max(biases), num=100)
    plt.plot(xseq, a + b * xseq, color="#d7191c", lw=2.5, linestyle='--', zorder=5)
    _SuperCluster = Anno.loc[STR, "Supercluster"]
    # Add title with improved formatting
    plt.title(f'{_SuperCluster} - {STR} \nSpearman ρ = {pho:.2f}, p = {p:.2e}', fontsize=14, fontweight='bold')

    # Labeling axes
    plt.xlabel("Cell Type Bias", fontsize=12, fontweight='bold')

    plt.ylabel("Full Scale IQ", fontsize=12, fontweight='bold')

    # Grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.5)

    # Adjust tick parameters
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

def BiasVsPheno(MutPhenoDF, BiasMat, STR, label):
    biases = []
    IQs = []
    for i, row in MutPhenoDF.iterrows():
        if label == 'label':
            gene = row["HGNC"]
        else:
            gene = int(row["Entrez"])
        if gene in BiasMat.index.values:
            bias = BiasMat.loc[gene, STR]
            if bias == bias:
                IQ = row["IQ"]
                biases.append(bias)
                IQs.append(IQ)
    return np.array(biases), np.array(IQs)

def ADJ_P(DF, p_col="p_value"):
    #_, q = fdrcorrection(DF["p"].values)
    #_, q, alphacSidak, alphacBonf= multipletests(DF["p"].values, method='fdr_by')
    DF = DF.sort_values(by=p_col, ascending=True)
    _, q, alphacSidak, alphacBonf= multipletests(DF[p_col].values, method='fdr_bh')
    DF[f'{p_col}_adj'] = q
    return DF

def Plot_Bias_vs_IQ_MoustCT(STR, Mut_n_IQ_conf, HCT_Z2_MAT_HCT, ax=None):
    biases, IQs = BiasVsPheno(Mut_n_IQ_conf, HCT_Z2_MAT_HCT , STR, 'XX')
    pho, p = spearmanr(biases, IQs)
    
    if ax is None:
        fig, ax = plt.subplots(dpi=150, figsize=(5, 4))

    # Plot data points
    ax.scatter(biases, IQs, s=50, color="#2c7bb6", edgecolor="black", alpha=0.8, zorder=10)

    # Fit and plot the trend line
    b, a = np.polyfit(biases, IQs, deg=1)
    xseq = np.linspace(min(biases), max(biases), num=100)
    ax.plot(xseq, a + b * xseq, color="#d7191c", lw=2.5, linestyle='--', zorder=5)

  

    # Compute average bias for IQ > 70 and IQ < 70
    high_iq_bias = np.mean([bias for bias, iq in zip(biases, IQs) if iq > 70])
    low_iq_bias = np.mean([bias for bias, iq in zip(biases, IQs) if iq <= 70])
    bias_diff = high_iq_bias - low_iq_bias

    # Add title with improved formatting and average bias information
    # ax.set_title(f'{STR}\nSpearman ρ = {pho:.2f}, p = {p:.2e}\n'
    #              f'Avg Bias (IQ>70): {high_iq_bias:.2f}, (IQ≤70): {low_iq_bias:.2f}, Diff: {bias_diff:.2f}', 
    #              fontsize=12, fontweight='bold')

    ax.set_title(f'{STR}\nSpearman ρ = {pho:.2f}, p = {p:.2e} PBS = {b:.2f}', 
                 fontsize=12, fontweight='bold')
  # Add line at IQ 70
    #ax.axhline(y=70, color='green', linestyle=':', linewidth=2)
    # plot HIQ, LIQ and bias diff as x-axis on y=70
    #ax.plot([high_iq_bias, low_iq_bias], [70, 70], color='black', linestyle='-', linewidth=5)
    # Labeling axes
    ax.set_xlabel("Cell Type Bias", fontsize=12, fontweight='bold')
    ax.set_ylabel("Full Scale IQ", fontsize=12, fontweight='bold')

    # Grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Tight layout for better spacing
    plt.tight_layout()

    return ax

def Plot_Bias_vs_IQ_MoustCT_forPaper(STR, Mut_n_IQ_conf, HCT_Z2_MAT_HCT, fixed_intercept=None, ax=None):
    biases, IQs = BiasVsPheno(Mut_n_IQ_conf, HCT_Z2_MAT_HCT , STR, 'XX')
    intercept, beta, ci_low, ci_high, r_value, p_value, std_err, pho, p = linear_fit(biases, IQs, alpha=0.05, fixed_intercept=fixed_intercept)
    #pho, p = spearmanr(biases, IQs)
    
    if ax is None:
        fig, ax = plt.subplots(dpi=150, figsize=(5, 4))

    # Plot data points
    ax.scatter(biases, IQs, s=50, color="#2c7bb6", edgecolor="black", alpha=0.8, zorder=10)

    # Fit and plot the trend line
    #b, a = np.polyfit(biases, IQs, deg=1)
    xseq = np.linspace(min(biases), max(biases), num=100)
    ax.plot(xseq, intercept + beta * xseq, color="#d7191c", lw=2.5, linestyle='--', zorder=5)

    # Compute average bias for IQ > 70 and IQ < 70
    high_iq_bias = np.mean([bias for bias, iq in zip(biases, IQs) if iq > 70])
    low_iq_bias = np.mean([bias for bias, iq in zip(biases, IQs) if iq <= 70])
    bias_diff = high_iq_bias - low_iq_bias


    #ax.set_title(f'{STR}\nSpearman ρ = {pho:.2f}, p = {p:.2e} \nPBS = {beta:.2f} PBS_p_value = {p_value:.2e}', 
    #            fontsize=12, fontweight='bold')
    
    #ax.set_title(f'{STR}\nPBS = {beta:.2f} p_value = {p_value:.1e}', 
    #             fontsize=12, fontweight='bold')
    ax.set_title(f'{STR}\nPBS = {beta:.2f} p_value = {p_value:.0e}', 
                 fontsize=12, fontweight='bold')
  # Add line at IQ 70
    #ax.axhline(y=70, color='green', linestyle=':', linewidth=2)
    # plot HIQ, LIQ and bias diff as x-axis on y=70
    #ax.plot([high_iq_bias, low_iq_bias], [70, 70], color='black', linestyle='-', linewidth=5)
    # Labeling axes
    ax.set_xlabel("Cell Type Bias", fontsize=12, fontweight='bold')
    ax.set_ylabel("Full Scale IQ", fontsize=12, fontweight='bold')

    # Grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Tight layout for better spacing
    plt.tight_layout()

    return ax

def Plot_Bias_vs_IQ_HumanCT(STR, Mut_n_IQ_conf, HCT_Z2_MAT_HCT, ax=None, Pval=None, textPos=(0.4, 0.85)):
    biases, IQs = BiasVsPheno(Mut_n_IQ_conf, HCT_Z2_MAT_HCT , STR, 'XX')
    pho, p = spearmanr(biases, IQs)
    if Pval is None:
        Pval = p
    if ax is None:
        fig, ax = plt.subplots(dpi=150, figsize=(8, 4))
    ax.scatter(biases, IQs, s=50, color="#2c7bb6", edgecolor="black", alpha=0.8, zorder=10)

    b, a = np.polyfit(biases, IQs, deg=1)
    xseq = np.linspace(min(biases), max(biases), num=100)
    ax.plot(xseq, a + b * xseq, color="#d7191c", lw=2.5, linestyle='--', zorder=5)
    # Compute average bias for IQ > 70 and IQ < 70
    high_iq_bias = np.mean([bias for bias, iq in zip(biases, IQs) if iq > 70])
    low_iq_bias = np.mean([bias for bias, iq in zip(biases, IQs) if iq <= 70])
    bias_diff = high_iq_bias - low_iq_bias

    _SuperCluster = Anno.loc[STR, "Supercluster"]

    ax.set_title(f'{_SuperCluster} - {STR}', 
            fontsize=25, fontweight='normal')
    ax.text(textPos[0], textPos[1], s=f'$PBS = {b:.2f}$\n{format_pval_scientific(Pval)}',
            fontsize=22.0, ha='left', va='top', transform=ax.transAxes)

    ax.set_xlabel("Cell Type Bias", fontsize=25, fontweight='normal')
    ax.set_ylabel("Full Scale IQ", fontsize=25, fontweight='normal')

    # Grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=15)

    return ax



# Whole Genome Regression 
def plot_cluster_correlation(cluster, SCZMutDF, specificity_scores, eff_label = "LGD_OR", plot=False):
    """
    Plot correlation between cluster bias and LGD odds ratio.
    
    Args:
        cluster (int): Cluster number to analyze
        SCZMutDF (pd.DataFrame): DataFrame containing mutation data
        specificity_scores (pd.DataFrame): Matrix of bias/specificity scores
    """
    entrez_list = SCZMutDF.index.values
    Zscore_list = SCZMutDF[eff_label].values
    Bias_list = specificity_scores.loc[entrez_list, cluster]
    valid_mask = ~np.isnan(Zscore_list) & ~np.isnan(Bias_list)

    # Calculate correlations
    spearman_corr, spearman_p = stats.spearmanr(Zscore_list[valid_mask], Bias_list[valid_mask])
    pearson_corr, pearson_p = stats.pearsonr(Zscore_list[valid_mask], Bias_list[valid_mask])
    #print(f"Spearman correlation: {spearman_corr}")
    #print(f"Pearson correlation: {pearson_corr}")

    # Clean data
    Zscore_list_clean = Zscore_list[valid_mask]
    Bias_list_clean = Bias_list[valid_mask]

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(Bias_list_clean, Zscore_list_clean, alternative="greater")
    
    if plot:
        # Create scatter plot
        plt.scatter(Bias_list_clean, Zscore_list_clean, s=1)

        # Add regression line
        x_range = np.array([min(Bias_list_clean), max(Bias_list_clean)])
        plt.plot(x_range, slope * x_range + intercept, 'r', 
                label=f'β = {slope:.2f}\nR² = {r_value**2:.3f}\np = {p_value:.2e}')

        plt.xlabel("Bias")
        plt.ylabel("Z-score") 
        plt.legend()
        plt.show()
    return spearman_corr, spearman_p, pearson_corr, pearson_p, slope,std_err, r_value, p_value, 
# Example usage:

def cell_type_bias_Linear_fit(specificity_scores, Spark_Meta_test, Anno, eff_label="ZSTAT"):

    intercept, beta, ci_low, ci_high, r_value, p_value, std_err, pho, p = linear_fit(specificity_scores, Spark_Meta_test, Anno, eff_label=eff_label)
    return beta, r_value, std_err, p_value

def calculate_cluster_correlations(specificity_scores, Spark_Meta_test, Anno, eff_label="AutismMerged_LoF"):
    # Create lists to store results
    clusters = []
    spearman_correlations = []
    spearman_pvalues = []
    pearson_correlations = []
    pearson_pvalues = []
    superclusters = []
    slope_values = []
    std_err_values = []
    r_value_values = []
    p_value_values = []


    for cluster in specificity_scores.columns.values:
        #spearman_corr, spearman_p, pearson_corr, pearson_p = plot_cluster_correlation(cluster, Spark_Meta_test, specificity_scores, eff_label=eff_label)
        spearman_corr, spearman_p, pearson_corr, pearson_p, slope, std_err, r_value, p_value = plot_cluster_correlation(cluster, Spark_Meta_test, specificity_scores, eff_label=eff_label)
        # Store results
        clusters.append(cluster)
        spearman_correlations.append(spearman_corr)
        spearman_pvalues.append(spearman_p)
        #pearson_correlations.append(pearson_corr)
        #pearson_pvalues.append(pearson_p)
        slope_values.append(slope)
        std_err_values.append(std_err)
        r_value_values.append(r_value)
        p_value_values.append(p_value)
        superclusters.append(Anno.loc[cluster, "Supercluster"])

    # Create DataFrame with results
    corr_df_ASD = pd.DataFrame({
        'Cluster': clusters,
        'SuperCluster': superclusters,
        'Spearman_Correlation': spearman_correlations,
        'Spearman_P_value': spearman_pvalues,
        'Slope': slope_values,
        'Std_err': std_err_values,
        'R_value': r_value_values,
        'P_value': p_value_values
    })
    corr_df_ASD = corr_df_ASD.sort_values(by="Spearman_P_value", ascending=True)
    return corr_df_ASD

def plot_supercluster_correlations(corr_df, title=""):
    """
    Create a box plot showing distribution of Spearman correlations by SuperCluster.
    
    Args:
        corr_df: DataFrame containing 'SuperCluster' and 'Spearman_Correlation' columns
    """
    # Calculate mean correlation per SuperCluster for sorting
    mean_corr = corr_df.groupby('SuperCluster')['Spearman_Correlation'].mean().sort_values(ascending=False)
    order = mean_corr.index

    # Create a box plot grouped by SuperCluster
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=corr_df, y='SuperCluster', x='Spearman_Correlation', order=order)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Bias Permutation Test Realted Fcuntions
# Too Slow, Do not use 
def getBiasesBySTR(str_id, dfs):
    """Get bias values for a given STR ID across multiple dataframes."""
    biases = [df.loc[str_id, "EFFECT"] for df in dfs]
    return np.array(biases)

def GetPermutationP(null_dist, observed_val, greater_than=True):
    """Calculate permutation-based p-value and z-score.
    
    Args:
        null_dist: List/array of null distribution values
        observed_val: Observed value to test
        greater_than: If True, calculates P(null >= observed), else P(null <= observed)
    
    Returns:
        z_score: Z-score of observed value relative to null distribution
        p_value: Permutation-based p-value
        obs_adj: Mean-adjusted observed value
    """
    # Remove NaN values
    null_dist = np.array([x for x in null_dist if not np.isnan(x)])
    
    # Calculate statistics
    z_score = (observed_val - np.mean(null_dist)) / np.std(null_dist)
    count = sum(observed_val <= x for x in null_dist) if greater_than else sum(observed_val >= x for x in null_dist)
    p_value = (count + 1) / (len(null_dist) + 1)
    obs_adj = observed_val - np.mean(null_dist)
    
    return z_score, p_value, obs_adj

def AddPvalue(df, control_dfs):
    """Add p-values, z-scores and adjusted effects to dataframe.
    
    Args:
        df: DataFrame with EFFECT column
        control_dfs: List of control DataFrames for null distribution
    """
    # Calculate statistics for each row
    all_biases = {ct: getBiasesBySTR(ct, control_dfs) for ct in df.index}
    z_scores, p_values, obs_adjs = zip(*[
        GetPermutationP(all_biases[ct], row["EFFECT"]) 
        for ct, row in df.iterrows()
    ])
    
    # Add columns
    df["P-value"] = p_values
    df["Z-score"] = z_scores  
    df["EFFECT_adj"] = obs_adjs

    # Calculate FDR-corrected q-values
    from statsmodels.stats.multitest import multipletests
    _, q_values = multipletests(df["P-value"].values, alpha=0.1, method="fdr_i")[0:2]
    df["q-value"] = q_values
    df["-logP"] = -np.log10(df["P-value"])
    return df


# Strategy 1: Vectorized operations with NumPy
def getBiasesBySTR_vectorized(str_ids, dfs):
    """Vectorized version to get bias values for multiple STR IDs at once."""
    # Stack all dataframes into a 3D array: (n_permutations, n_cell_types, features)
    effects_matrix = np.stack([df.loc[str_ids, "EFFECT"].values for df in dfs], axis=0)
    return effects_matrix  # Shape: (n_permutations, n_cell_types)

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

def AddPvalue_optimized(df, control_dfs, method='vectorized', **kwargs):
    """Optimized version of AddPvalue with multiple acceleration strategies."""
    
    # Convert to numpy arrays for faster processing
    observed_vals = df["EFFECT"].values
    cell_type_ids = df.index.tolist()
    
    print(f"Processing {len(cell_type_ids)} cell types with {len(control_dfs)} permutations...")
    
    if method == 'vectorized':
        # Strategy 1: Pure vectorization
        null_matrix = getBiasesBySTR_vectorized(cell_type_ids, control_dfs)
        z_scores, p_values, obs_adjs = GetPermutationP_vectorized(null_matrix, observed_vals)
        
    elif method == 'numba':
        # Strategy 2: Numba JIT compilation
        null_matrix = getBiasesBySTR_vectorized(cell_type_ids, control_dfs)
        z_scores, p_values, obs_adjs = calculate_pvalues_numba(null_matrix, observed_vals)
        
    elif method == 'parallel':
        # Strategy 3: Parallel processing
        null_matrix = getBiasesBySTR_vectorized(cell_type_ids, control_dfs)
        z_scores, p_values, obs_adjs = calculate_pvalues_parallel(
            null_matrix, observed_vals, **kwargs)
        
    elif method == 'streaming':
        # Strategy 4: Memory-efficient streaming (for very large datasets)
        return AddPvalue_streaming(df, control_dfs, **kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add results to dataframe
    df = df.copy()
    df["P-value"] = p_values
    df["Z-score"] = z_scores  
    df["EFFECT_adj"] = obs_adjs

    # Calculate FDR-corrected q-values
    from statsmodels.stats.multitest import multipletests
    _, q_values = multipletests(df["P-value"].values, alpha=0.1, method="fdr_i")[0:2]
    df["q-value"] = q_values
    df["-logP"] = -np.log10(df["P-value"])
    
    return df

def test_all_superclusters_vectorized(all_cts, anno, obs_bias_df, rand_dfs):
    """Vectorized test enrichment for all superclusters."""
    # Map supercluster to indices
    supercluster_map = {
        sc: anno[anno["Supercluster"] == sc].index.values
        for sc in all_cts
    }
    # Prepare observed and null matrices
    obs_effects = []
    null_effects = []
    for sc, idxs in supercluster_map.items():
        obs_vals = obs_bias_df.loc[idxs, "EFFECT"].values
        obs_effects.append(obs_vals.mean())
        # Stack nulls for this supercluster
        null_vals = np.stack([df.loc[idxs, "EFFECT"].values for df in rand_dfs])
        null_effects.append(null_vals.mean(axis=1))
    obs_effects = np.array(obs_effects)
    null_effects = np.stack(null_effects)  # shape: (n_superclusters, n_perms)
    null_means = null_effects.mean(axis=1)
    null_stds = null_effects.std(axis=1, ddof=1)
    z_scores = (obs_effects - null_means) / null_stds
    # Two-sided p-value
    p_values = ((np.abs(null_effects - null_means[:, None]) >= np.abs(obs_effects - null_means)[:, None]).sum(axis=1) + 1) / (null_effects.shape[1] + 1)
    res_df = pd.DataFrame({
        'Supercluster': all_cts,
        'Observed_Effect': obs_effects,
        'Z_score': z_scores,
        'P_value': p_values
    })
    res_df['FDR'] = multipletests(res_df['P_value'], method='fdr_bh')[1]
    return res_df.sort_values(by='FDR', ascending=True)

def PlotQQ(df_list, name_list):
    """Create Q-Q plot comparing observed vs expected p-values."""
    plt.figure(dpi=120, figsize=(6, 6))
    sns.set(style="whitegrid", context="talk")
    
    for df, name in zip(df_list, name_list):
        q_exp, q_obs = GetExpQ(df["P-value"].values)
        plt.scatter(q_exp, q_obs, alpha=0.7, s=50, label=name)
    
    # Add reference line and formatting
    max_val = 4.5
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    plt.xlabel('Expected -log10(p-value)', fontsize=25, weight='bold')
    plt.ylabel('Observed -log10(p-value)', fontsize=25, weight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend(fontsize=15, loc="lower right")
    plt.show()

def LoadNullDF(ctrl_dir):
    """Load control DataFrames from directory."""
    rand_dfs = []
    for i in range(10000):
        #try:
        df = pd.read_csv(f"{ctrl_dir}/cont.bias.{i}.csv.gz", index_col=0)
        rand_dfs.append(df)
        #except:
        #    continue
    print(f"Loaded {len(rand_dfs)} control DataFrames")
    return rand_dfs

def test_all_superclusters(all_cts, anno, obs_bias_df, rand_dfs):
    """Test enrichment for all superclusters."""
    results = []
    for supercluster in all_cts:
        supercluster_idx = anno[anno["Supercluster"]==supercluster].index
        obs, null, z_score, p_val = test_supercluster_enrichment(
            supercluster_idx, obs_bias_df, rand_dfs
        )
        results.append({
            'Supercluster': supercluster,
            'Observed_Effect': obs,
            'Z_score': z_score,
            'P_value': p_val
        })
    
    res_df = pd.DataFrame(results)
    res_df['FDR'] = multipletests(res_df['P_value'], method='fdr_bh')[1]
    return res_df.sort_values(by='FDR', ascending=True)

def test_supercluster_enrichment(supercluster_idx, obs_df, rand_dfs, print_flag=False):
    """Test enrichment for a single supercluster."""
    obs = obs_df.loc[supercluster_idx, "EFFECT"].mean()
    null = [df.loc[supercluster_idx, "EFFECT"].mean() for df in rand_dfs]
    null = np.array(null)

    z_score, p_val, obs_adj = GetPermutationP(null, obs)
    
    if print_flag:
        print(f"Observed effect for Cell Type group: {obs}")
        print(f"P-value: {p_val}")
        
    return obs, null, z_score, p_val

########################################################
# PBS related functions
########################################################


def linear_fit(biases, IQs, alpha=0.05):
    model = sm.OLS(IQs, sm.add_constant(biases))
    results = model.fit()
    
    intercept = results.params[0]
    beta = results.params[1]
    # get CI of beta
    ci = results.conf_int(alpha=alpha)

    ci_low = ci[1][0]
    ci_high = ci[1][1]     
    r_value = results.rsquared
    p_value = results.pvalues[1]
    std_err = results.bse[1]
    rho, p_rho = spearmanr(biases, IQs)
    r, p_r = pearsonr(biases, IQs)
    
    return intercept, beta, ci_low, ci_high, r_value, p_value, std_err, rho, p_rho, r, p_r

def Make_HumanCT_DF(Mut_n_IQ_conf, HCT_Z2_MAT_HCT, output_file, alpha=0.05):
    names, supercluster, spearmanr, spearmanp, pearsonr, pearsonp, beta_values, beta_ci_low, beta_ci_high, intercept_values, r_value_values, p_value_values, std_err_values = [],[],[],[],[],[],[],[],[],[],[],[],[]
    for Idx in HCT_Z2_MAT_HCT.columns.values:
        biases, IQs = BiasVsPheno(Mut_n_IQ_conf, HCT_Z2_MAT_HCT , Idx, 'xx')
        intercept, beta, ci_low, ci_high, r_value, p_value, std_err, rho, p_rho, r, p_r = linear_fit(biases, IQs, alpha=0.05)
        
        names.append("{}".format(Idx))
        supercluster.append(Anno.loc[Idx, "Supercluster"])
        spearmanr.append(rho)
        spearmanp.append(p_rho)
        pearsonr.append(r)
        pearsonp.append(p_r)
        beta_values.append(beta)
        beta_ci_low.append(ci_low)
        beta_ci_high.append(ci_high)
        intercept_values.append(intercept)
        r_value_values.append(r_value)
        p_value_values.append(p_value)
        std_err_values.append(std_err)

    str_res_df = pd.DataFrame(data={"CT":names, "Supercluster":supercluster, "SpearmanR":spearmanr, "SpearmanP":spearmanp, 
                                            "PearsonR":pearsonr, "PearsonP":pearsonp, "beta":beta_values, "CI_low":beta_ci_low, "CI_high":beta_ci_high, "intercept":intercept_values, "r_value":r_value_values, 
                                            "p_value":p_value_values, "std_err":std_err_values})
    str_res_df = str_res_df.sort_values("SpearmanR")
    #str_res_df = ADJ_P(str_res_df)
    str_res_df.to_csv(output_file)
    return str_res_df

def GetFDRCut_PBS(DF, FDR=0.05, FDR_label="p_beta_perm_FDR", Pval_label="p_beta_perm_Log"):
    tmp = DF[DF[FDR_label]<=FDR]
    try:
        return min(tmp[Pval_label].values)
    except:
        return 0

def SuperClusterBias_BoxPlot_CorrIQ(DF1, flip_axis=True, plot_metric="beta", figsize=(8, 10), xlabel="", sortby="mean", FDR_label="p_beta_perm_FDR", Pval_label="p_beta_perm_Log"):
    # Prepare data
    dat_Z2 = []
    mean_Z2 = []
    for _CT in Neurons:
        tmp = DF1[DF1["Supercluster"] == _CT]
        dat_Z2.append(tmp[plot_metric].values)
        if sortby == "mean":
            mean_Z2.append(np.mean(tmp[plot_metric].values))
        elif sortby == "median":
            mean_Z2.append(np.median(tmp[plot_metric].values))
    mean_Z2 = np.array(mean_Z2)
    # Sort data###
    if flip_axis:
        sort_idx = np.argsort(mean_Z2)[::-1]
    else:
        sort_idx = np.argsort(mean_Z2)
    show_dat_Z2 = [dat_Z2[x] for x in sort_idx]
    show_CTs = [ALL_CTs[x] for x in sort_idx]

    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.4)
    
    # Create figure
    fig, ax = plt.subplots(dpi=600, figsize=figsize)

    # Define colors and styles
    box_color = '#2E5A88'  # Darker blue
    point_color = '#1f77b4'  # Seaborn blue
    
    # Customize boxplot appearance
    boxprops = dict(linestyle='-', linewidth=1.5, color=box_color, alpha=0.8)
    medianprops = dict(linestyle='-', linewidth=2.5, color='#D62728')  # Red median line
    whiskerprops = dict(color=box_color, linewidth=1.5, alpha=0.8)
    capprops = dict(color=box_color, linewidth=1.5)
    flierprops = dict(marker='', color=box_color, alpha=0)  # Hide outliers

    # Draw boxplot
    bp = ax.boxplot(show_dat_Z2, labels=show_CTs, vert=False, patch_artist=True,
                    boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
    
    # Fill boxes with lighter color
    for patch in bp['boxes']:
        patch.set_facecolor('#A8C8E4')  # Light blue
        patch.set_alpha(0.3)

    # Add individual points with jitter
    for i in range(len(show_dat_Z2)):
        y = np.random.normal(i + 1, 0.08, size=len(show_dat_Z2[i]))
        ax.scatter(show_dat_Z2[i], y, s=20, color=point_color, alpha=0.4, 
                  edgecolor='none', zorder=2)

    # Customize grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='gray')
    ax.grid(False, axis='y')

    # Labels and title
    if plot_metric == "beta":
        ax.set_xlabel("PBS", fontsize=20, fontweight='normal')
    elif plot_metric == "SpearmanR":
        ax.set_xlabel("Spearman Correlation", fontsize=20, fontweight='normal')
        #ax.set_ylabel("Superclusters", fontsize=20, fontweight='normal')
    elif plot_metric == "p_beta_perm_Log" or plot_metric == "-log10(p)":
        #ax.set_xlabel("PBS -log10(P)", fontsize=20, fontweight='normal')
        ax.set_xlabel(xlabel, fontsize=20, fontweight='normal')
        FDR_cut = GetFDRCut_PBS(DF1, FDR=0.05, FDR_label=FDR_label, Pval_label=Pval_label)
        ax.axvline(x=FDR_cut, color='red', linestyle='--', linewidth=1, alpha=0.8)
        #ax.text(FDR_cut, 0.95, "FDR=0.05", fontsize=12, fontweight='normal', ha='right', va='top', color='red', alpha=0.5)
        ax.text(FDR_cut, 0.95, "FDR=0.05", fontsize=12, fontweight='normal', va='top', color='red', alpha=0.8)
    else:
        ax.set_xlabel(xlabel, fontsize=20, fontweight='normal')

    # Customize axis appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='y', length=0)  # Remove y-axis ticks
    
    # Flip axis if needed
    if flip_axis:
        ax.invert_xaxis()
    
    if plot_metric == "beta":
        #ax.set_xlim(7.5, -10)
        pass



    # Add zero line
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.show()

def mapsize(x, min_size, max_size):
    return 5 + 50 * (x - min_size) / (max_size - min_size)

def plot_mutation_bias(df, beta_column, size_column='n_cells'):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(11, 11), dpi=600)

    if beta_column not in df.columns:
        print(f"'{beta_column}' column not found. Available columns:", df.columns)
        return

    # Prepare data
    median_bias = df.groupby("Subclass_idx")[beta_column].median().sort_values()
    sorted_superclusters = median_bias.index
    x_categorical = pd.Categorical(df['Subclass_idx'], categories=sorted_superclusters, ordered=True)
    x_numeric = x_categorical.codes

    # Enhanced boxplot
    sns.boxplot(x=x_numeric, y=beta_column, data=df,
                order=range(len(sorted_superclusters)), 
                color='white', fliersize=0, ax=ax1, zorder=0,
                boxprops={'alpha': 0.7, 'linewidth': 1.5},
                whiskerprops={'linewidth': 1.5},
                capprops={'linewidth': 1.5},
                medianprops={'color': '#D62728', 'linewidth': 2})

    # Add points with size proportional to cell count
    min_size, max_size = df[size_column].min(), df[size_column].max()
    for i, idx in enumerate(sorted_superclusters):
        subset = df[df["Subclass_idx"] == idx]
        color = '#2171B5' if 'Gaba' in subset['Subclass'].iloc[0] else '#CB181D'  # Darker colors
        jittered_x = np.random.normal(i, 0.1, size=len(subset))
        ax1.scatter(jittered_x, subset[beta_column],
                   s=mapsize(subset[size_column], min_size, max_size),
                   color=color, alpha=0.6, edgecolors='none', zorder=10)

    # Enhance plot appearance
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylabel(f'Mutation Bias - IQ {beta_column}', fontsize=14, fontweight='normal')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add weighted mean line
    weighted_mean_beta = (df[beta_column] * df[size_column]).sum() / df[size_column].sum()
    ax1.axhline(weighted_mean_beta, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(0.90, weighted_mean_beta * 1.1, 'Cell Mean',
             transform=ax1.get_yaxis_transform(), color='gray', 
             fontsize=12, fontweight='bold', ha='left', va='center')

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Inhibitory',
                   markerfacecolor='#2171B5', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Excitatory',
                   markerfacecolor='#CB181D', markersize=10)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, 
               framealpha=0.9, edgecolor='none')

    ax1.invert_yaxis()

    plt.tight_layout()
    plt.show()

def plot_null_distributions(HumanCT_res_df_GeneL, Perm_DFs, CT=276, plot=False):
    """
    Plot null distributions of Rho and Beta values compared to observed values for a given cell type.
    
    Parameters:
    -----------
    HumanCT_res_df_GeneL : pandas DataFrame
        DataFrame containing observed results
    Perm_DFs : list
        List of DataFrames containing permutation results
    CT : int
        Cell type ID to analyze (default: 276)
    """
    Obs_Rho = HumanCT_res_df_GeneL.loc[CT, "SpearmanR"]
    Obs_Beta = HumanCT_res_df_GeneL.loc[CT, "beta"]
    
    Null_Rho = []
    Null_Beta = []
    for df in Perm_DFs:
        Null_Rho.append(df.loc[CT, "SpearmanR"])
        Null_Beta.append(df.loc[CT, "beta"])
    Null_Rho = np.array(Null_Rho)
    Null_Beta = np.array(Null_Beta)
    # Calculate p-values
    p_rho = (Null_Rho <= Obs_Rho).mean()
    p_beta = (Null_Beta <= Obs_Beta).mean()
    
    if plot:
        # Create plots comparing observed values to null distributions
        plt.figure(figsize=(10,5))

        # Plot Rho distribution and observed value
        plt.subplot(121)
        plt.hist(Null_Rho, bins=20, alpha=0.5)
        plt.axvline(x=Obs_Rho, color='red', linestyle='--', label='Observed')
        plt.xlabel('Rho')
        plt.ylabel('Count')
        plt.text(0.60, 0.60, f'p_rho = {p_rho:.2e}', transform=plt.gca().transAxes, fontsize=12, ha='right', va='top')
        plt.title('Rho: Null Distribution vs Observed')
        plt.legend()

        # Plot Beta distribution and observed value  
        plt.subplot(122)
        plt.hist(Null_Beta, bins=20, alpha=0.5)
        plt.axvline(x=Obs_Beta, color='red', linestyle='--', label='Observed')
        plt.xlabel('Beta')
        plt.ylabel('Count')
        plt.title('Beta: Null Distribution vs Observed')
        plt.legend()
    # annotate p-values

        plt.text(0.60, 0.60, f'p_beta = {p_beta:.2e}', transform=plt.gca().transAxes, fontsize=12, ha='right', va='top')
        plt.tight_layout()
        plt.show()
    
    return p_rho, p_beta

def plot_null_suptercluster_distributions(ClusterIdx, HumanCT_res_df_GeneL, Perm_DFs, plot=False):

    Obs_Rho = HumanCT_res_df_GeneL.loc[ClusterIdx, "SpearmanR"].mean()
    Obs_Beta = HumanCT_res_df_GeneL.loc[ClusterIdx, "beta"].mean()
    

    Null_Rho = []
    Null_Beta = []
    for df in Perm_DFs:
        Null_Rho.append(df.loc[ClusterIdx, "SpearmanR"].mean())
        Null_Beta.append(df.loc[ClusterIdx, "beta"].mean())
    Null_Rho = np.array(Null_Rho)
    Null_Beta = np.array(Null_Beta)
    # Calculate p-values
    p_rho = (Null_Rho <= Obs_Rho).mean()
    p_beta = (Null_Beta <= Obs_Beta).mean()

    
    if plot:
        # Create plots comparing observed values to null distributions
        plt.figure(figsize=(10,5))

        # Plot Rho distribution and observed value
        plt.subplot(121)
        plt.hist(Null_Rho, bins=20, alpha=0.5)
        plt.axvline(x=Obs_Rho, color='red', linestyle='--', label='Observed')
        plt.xlabel('Rho')
        plt.ylabel('Count')
        plt.text(0.60, 0.60, f'p_rho = {p_rho:.2e}', transform=plt.gca().transAxes, fontsize=12, ha='right', va='top')
        plt.title('Rho: Null Distribution vs Observed')
        plt.legend()

        # Plot Beta distribution and observed value  
        plt.subplot(122)
        plt.hist(Null_Beta, bins=20, alpha=0.5)
        plt.axvline(x=Obs_Beta, color='red', linestyle='--', label='Observed')
        plt.xlabel('Beta')
        plt.ylabel('Count')
        plt.title('Beta: Null Distribution vs Observed')
        plt.legend()
    # annotate p-values

        plt.text(0.60, 0.60, f'p_beta = {p_beta:.2e}', transform=plt.gca().transAxes, fontsize=12, ha='right', va='top')
        plt.tight_layout()
        plt.show()
    
    return p_rho, p_beta