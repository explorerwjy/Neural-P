import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
import scipy.stats as stats
import pysam
import sys
import gzip
import subprocess
import igraph as ig
import itertools
import os 


def GetPermutationP(null, obs):
    count = 0
    for i,v in enumerate(null):
        if obs >= v:
            count += 1
    return 1-float(count)/(len(null)+1)

def PlotPermutationP(Null, Obs):
    n, bins, patches = plt.hist(Null, bins=20, histtype = "barstacked", align = 'mid', facecolor='black', alpha=0.8)
    p = GetPermutationP(Null, Obs)
    plt.vlines(x=Obs, ymin=0, ymax=max(n))
    plt.text(x=Obs, y=max(n), s="p=%.3f"%p)
    plt.show()
    return

#=========================================================================
# Metabolic Network
#=========================================================================
def GetFName(x1, x2, label, DIR):
    return DIR+"%s-%d-%s.csv"%(label,x1,str(x2))

def PermuteRNX(RNX = 1, ConditionDir="../dat/k-ecoli457/data/aerobic_glucose/"):
    F1 = GetFName(RNX,1,"Metabolite",ConditionDir)
    F2 = GetFName(RNX,0.5,"Metabolite",ConditionDir)
    F3 = GetFName(RNX,2,"Metabolite",ConditionDir)

    df1 = pd.read_csv(F1, header=None)
    df2 = pd.read_csv(F2, header=None)
    df3 = pd.read_csv(F3, header=None)
    Len1 = df1.shape[1]
    Len2 = df2.shape[1]
    Len3 = df3.shape[1]
    M1, M2, M3 = df1[Len1-1].values, df2[Len2-1].values, df3[Len3-1].values 

    F1 = GetFName(RNX,1,"Vnet",ConditionDir)
    F2 = GetFName(RNX,0.5,"Vnet",ConditionDir)
    F3 = GetFName(RNX,2,"Vnet",ConditionDir)

    df1 = pd.read_csv(F1, header=None)
    df2 = pd.read_csv(F2, header=None)
    df3 = pd.read_csv(F3, header=None)
    Len1 = df1.shape[1]
    Len2 = df2.shape[1]
    Len3 = df3.shape[1]
    V1, V2, V3 = df1[Len1-1].values, df2[Len2-1].values, df3[Len3-1].values 
    return V1,V2,V3,M1,M2,M3

def TestOne(RNX, cond = 1, pheno = "Metabolite", ConditionDir="../dat/k-ecoli457/data/aerobic_glucose/"):
    #df1 = pd.read_csv(F1, header=None); Len1 = df1.shape[1]
    step = 1
    if cond == 1:
        F1 = GetFName(RNX,1, pheno, ConditionDir)
        df1 = pd.read_csv(F1, header=None);
        Len1 = df1.shape[1]; 
        for i in range(100):
            plt.plot(np.array(range(Len1))*step, df1.loc[i, :])
    elif cond == 2:
        F2 = GetFName(RNX,0.5, pheno, ConditionDir)
        df2 = pd.read_csv(F2, header=None);
        Len2 = df2.shape[1]; 
        for i in range(100):
            plt.plot(np.array(range(Len2))*step, df2.loc[i, :], linewidth=0.8)
    elif cond == 3:
        F3 = GetFName(RNX,2, pheno, ConditionDir)
        df3 = pd.read_csv(F3, header=None);
        Len3 = df3.shape[1]
        for i in range(100):
            plt.plot(np.array(range(Len3))*step, df3.loc[i, :])
    #plt.xlim((1,4000))
    plt.title("K-ecoli457 without perturbation")
    #plt.ylim((0, 2))
    plt.grid(True)
    plt.xlabel("T")
    plt.ylabel("Metabolite concentrations")
    plt.show()

def isSteady(t, v, t_steady=0.5, err=0.005):
    idx = len(t)-1; last_t = t[idx]; last_v = v[idx]
    #for i in range(int(t_steady*len(t))):
    for i in range(100):
        curr_v = v[idx-10]; curr_t = t[idx-10]
        if (curr_v - last_v)/last_v > err:
        #if (curr_v - last_v) > err:
            return False
        idx = idx-1; last_v = curr_v
    return True

def TestOne2(RNX, cond = 1, pheno = "Metabolite", ConditionDir="../dat/k-ecoli457/data/aerobic_glucose/", filtUnsteady = False, ylim=None):
    print(GetFName(RNX, cond, "Metabolite", ConditionDir))
    df1 = pd.read_csv(GetFName(RNX, cond, "Metabolite", ConditionDir), header=None)
    df2 = pd.read_csv(GetFName(RNX, cond, "Vnet", ConditionDir), header=None)
    df2 = df2.transpose()
    Ntime, Nrxn = df1.shape 
    counter = 0
    fig, ax = plt.subplots(1,2, figsize = (8, 2), dpi=80)
    for i in range(Nrxn):
        if filtUnsteady:
            if isSteady(np.array(range(Ntime)), df1.loc[:, i]):
                ax[0].plot(np.array(range(Ntime)), df1.loc[:, i])
                ax[1].plot(np.array(range(Ntime)), df2.loc[:, i])
                counter += 1
        else:
            ax[0].plot(np.array(range(Ntime)), df1.loc[:, i])
            ax[1].plot(np.array(range(Ntime)), df2.loc[:, i])
            counter += 1
    #plt.xlim((1,4000))
    plt.ylim(ylim)
    ax[0].set_title("Metabolite rxn:%d, permute:%.1f, Nmeta:%d"%(RNX, cond, counter))
    ax[1].set_title("Vnet rxn:%d, permute:%.1f, Nmeta:%d"%(RNX, cond, counter))
    #plt.title("K-ecoli138 without perturbation")
    ax[0].grid(True)
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()

def ScanRXN(Nrxn = 138, Conds = [0.9, 1.1], pheno = "Metabolite", ConditionDir="../dat/k-ecoli138/dat/doPermute/"):
    Good_RXN = []; Bad_RXN = []
    for x in range(Nrxn):
        idx = x+1
        FLAG_GOOD = True
        for cond in Conds:
            df = pd.read_csv(GetFName(idx, cond, pheno, ConditionDir), header=None)
            Ntime, Nrxn = df.shape
            if Ntime < 100:
                FLAG_GOOD = False
        if FLAG_GOOD:
            Good_RXN.append(idx)
        else:
            Bad_RXN.append(idx)
    return Good_RXN, Bad_RXN

def PlotVariables(RNX, Nprofile = 100, ConditionDir="../dat/k-ecoli457/data/aerobic_glucose/"):
    FVnet = [GetFName(RNX, x, "Vnet", ConditionDir) for x in [1, 0.5, 2]]
    DFVnet = [pd.read_csv(X, header=None) for X in FVnet] 
    #Len1 = df1.shape[1]; Len2 = df2.shape[1]; Len3 = df3.shape[1]
    FMeta = [GetFName(RNX, x, "Metabolite", ConditionDir) for x in [1, 0.5, 2]]
    DFMeta = [pd.read_csv(X, header=None) for X in FMeta]
    fig, axs = plt.subplots(3,2, dpi=160)
    for i,j in [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]:
        if j == 1:
            for k in range(Nprofile):
                #axs[i, j].plot(range(DFVnet[i].shape[1]), DFVnet[i].loc[k, :])
                axs[i, j].plot(range(DFVnet[i].shape[1]), DFVnet[i].loc[k, :]/np.mean(DFVnet[i].loc[k, :]))
            axs[i, j].set_title('Vnet')
            #axs[i, j].legend()
        else:
            for k in range(Nprofile):
                axs[i, j].plot(range(DFMeta[i].shape[1]), DFMeta[i].loc[k, :])
            axs[i, j].set_title('Metabolite')
            #axs[i, j].legend()
    plt.tight_layout()
    plt.show()

def Change(RNX, ConditionDir="../dat/k-ecoli457/data/aerobic_glucose/", xlim=(-1,1)):
    FVnet = [GetFName(RNX, x, "Vnet", ConditionDir) for x in [1, 0.5, 2]]
    DFVnet = [pd.read_csv(X, header=None) for X in FVnet]
    FMeta = [GetFName(RNX, x, "Metabolite", ConditionDir) for x in [1, 0.5, 2]]
    DFMeta = [pd.read_csv(X, header=None) for X in FMeta]
    # Upregulate
    V1,V2,V3,M1,M2,M3 = PermuteRNX(RNX) 
    UPs, DNs = [], []
    for m1, m2, m3 in zip(M1, M2, M3):
        down = (m2 - m1) / m1
        up = (m3 - m1) / m1
        UPs.append(up)
        DNs.append(down)
    plt.title(r"Distribution of metabolite concentration change ($\frac{[A] - [A']}{[A]}$)")
    plt.hist(UPs, bins=50, label="up", alpha=0.5, density=True)
    plt.hist(DNs, bins=50, label="down", alpha=0.5, density=True)
    plt.grid(True)
    plt.legend()
    plt.xlim(xlim)
    plt.show()

# Permute 0, 0.5, 0.9, 1.1, 2
def Change2(RNX, up=10, down=0, ConditionDir="../dat/k-ecoli138/dat/doPermute/", show = True, xlim=None):
    VnetRef = pd.read_csv(ConditionDir+"Vnet-1-1.csv", header=None)
    MetaRef = pd.read_csv(ConditionDir+"Metabolite-1-1.csv", header=None)
    FVnet = [GetFName(RNX, x, "Vnet", ConditionDir) for x in [down, up]]
    DFVnet = [pd.read_csv(X, header=None) for X in FVnet]
    FMeta = [GetFName(RNX, x, "Metabolite", ConditionDir) for x in [down, up]]
    DFMeta = [pd.read_csv(X, header=None) for X in FMeta]
    # Upregulate
    #V1,V2,V3,M1,M2,M3 = PermuteRNX(RNX) 
    #print(VnetRef.shape)
    Nrxn, Ntime = MetaRef.shape
    MREF = MetaRef.values[-1, :]
    M1, M2 = [DF.values[-1, :] for DF in DFMeta]
    UPs, DNs = [], []
    for mref, m1, m2 in zip(MREF, M1, M2):
        #down = (m2 - mref) / mref
        #up = (m3 - mref) / mref
        down = math.log10(m1)
        up = math.log10(m2)
        UPs.append(up)
        DNs.append(down)
    c1, c2 = 0.0, 0.0
    cut_up = math.log10(1.1)
    cut_dn = math.log10(0.9)
    for down in DNs:
        if down < cut_dn or down > cut_up:
            c1 += 1
    for up in UPs:
        if up < cut_dn or up > cut_up:
            c2 += 1
    if show:
        #plt.title(r"Distribution of metabolite concentration change ($\frac{[A] - [A']}{[A]}$)")
        plt.title(r"Distribution of metabolite concentration change ($\log_{10}(\frac{[A']}{[A]})$)")
        plt.hist(UPs, bins=20, label="up", alpha=0.5, density=True)
        plt.hist(DNs, bins=20, label="down", alpha=0.5, density=True)
        plt.grid(True)
        #plt.axvline(x=cut_up)
        #plt.axvline(x=cut_dn)
        plt.legend()
        plt.xlim(xlim)
        plt.show()

    return c1/93, c2/93

def compare(RNX, cond1, cond2, ConditionDir="../dat/k-ecoli138/dat/doPermute/", log10=True):
    if cond1 == 1:
        M1 = pd.read_csv(GetFName(1, 1, "Metabolite", ConditionDir), header=None).values[-1, :]
    else:
        M1 = pd.read_csv(GetFName(RNX, cond1, "Metabolite", ConditionDir), header=None).values[-1, :]
    if cond2 == 1:
        M2 = pd.read_csv(GetFName(1, 1, "Metabolite", ConditionDir), header=None).values[-1, :]
    else:
        M2 = pd.read_csv(GetFName(RNX, cond2, "Metabolite", ConditionDir), header=None).values[-1, :]
    if log10:
        M1, M2 = np.log10(M1), np.log10(M2)
    plt.scatter(M1, M2, s=5, color="black")
    plt.grid(True)
    plt.show()


def PlotCorrelationScatter(idx,V1,V2,V3,M1,M2,M3):
    fig, axs = plt.subplots(2, 2, dpi = 120)
    axs[0, 0].scatter(V1, V3, s=5, label="Pearson=%.4f"%(pearsonr(V1, V3)[0]))
    axs[0, 0].set_title('2 vs 1 Vnet')
    axs[0, 0].legend()
    axs[0, 1].scatter(V1, V2, s=5, label="Pearson=%.4f"%(pearsonr(V1, V2)[0]))
    axs[0, 1].set_title('0.5 vs 1 Vnet')
    axs[0, 1].legend()
    axs[1, 0].scatter(M1, M3, s=5, label="Pearson=%.4f"%(pearsonr(M1, M3)[0]))
    axs[1, 0].set_title('2 vs 1 Concentration')
    axs[1, 0].legend()
    axs[1, 1].scatter(M1, M2, s=5, label="Pearson=%.4f"%(pearsonr(M1, M2)[0]))
    axs[1, 1].set_title('0.5 vs 1 Concentration')
    axs[1, 1].legend()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

#=========================================================================
# GWAS with Biological Network 
#=========================================================================
class SNP:
    def __init__(self, Chr, Pos, Effect, SE, P):
        self.Chr = Chr
        self.Pos = int(Pos)
        #self.INFO = INFO
        self.Effect = float(Effect)
        self.SE = float(SE)
        self.P = float(P)

class GENE:
    def __init__(self, Chr, Start, End, GeneName, GeneID):
        self.Chr = Chr
        self.Start = Start
        self.End = End
        self.GeneName = GeneName
        self.GeneID = GeneID
        self.SNPs = []

def reformat(inpfil, outputfil):
    fin = open(inpfil, 'rt')
    writer = csv.writer(open(outputfil, 'wt'), delimiter="\t")
    for l in fin:
        llist = l.split()
        writer.writerow(llist)

def LoadIDMapping():
    df = pd.read_csv("/home/jw3514/Work/Resources/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    # symbol2ensg = dict(zip(df["symbol"].values, df["ensembl_gene_id"].values))
    Symbol2ENSG = {}
    Uniprot2ENSG = {}
    ENSP2ENSG = {}
    Entrez2ENSG = {}
    for i, row in df.iterrows():
        ENSG = row["ensembl_gene_id"]
        Symbol = row["symbol"]
        Entrez = row["entrez_id"]
        Symbol2ENSG[Symbol] = ENSG
        Entrez2ENSG[Entrez] = ENSG
        try:
            for Uniprot in row["uniprot_ids"].strip('"').split("|"):
                Uniprot2ENSG[Uniprot] = ENSG
        except:
            continue
    return Symbol2ENSG, Uniprot2ENSG, ENSP2ENSG, Entrez2ENSG

def runBash(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr

def FiltSNP(infil, outfile, CHR_field = "CHR", POS_field = "BP", INFO_field="INFO", INFO=0.9, pvalue_field = "pvalue", pvalue_cut = 0.05):
    reader = csv.reader(gzip.open(infil, 'rt'), delimiter="\t")
    writer = csv.writer(open(outfile, 'wt'), delimiter="\t")
    head = next(reader)
    idx_info = head.index(INFO_field)
    idx_p = head.index(pvalue_field)
    #writer.writerow(head)
    idx_CHROM = head.index(CHR_field)
    idx_POS = head.index(POS_field)
    writer.writerow(["#Chr", "Start", "End"] + head)
    SORTED = True
    LastChr = None
    for row in reader:
        CHROM = row[idx_CHROM]
        POS = row[idx_POS]
        info = float(row[idx_info])
        p = float(row[idx_p])
        if info < INFO or p > pvalue_cut:
            continue
        else:
            if LastChr == CHROM and int(LastPos) > int(Pos):
                SORTED = False
            #writer.writerow(row)
            #writer.writerow(["#Chr", "Start", "End"] + head)
            row = [CHROM, POS, POS] + row
            writer.writerow(row)
            LastChr = CHROM
            LastPos = Pos
    if SORTED:
        bgzip = "bgzip %s"%outfile
        tabix = "tabix -p bed %s.gz"%outfile
        stdout, stderr = runBash(bgzip)
        print("bgzip out: %s; err: %s"%(stdout, stderr))
        stdout, stderr = runBash(tabix)
        print("tabix out: %s; err: %s"%(stdout, stderr))
    else:
        sort = "sort -k 1,1 -k2,2n %s > sorted.%s"%(outfile, outfile)
        stdout, stderr = runBash(sort)
        bgzip = "bgzip sorted.%s"%outfile
        tabix = "tabix -p bed sorted.%s.gz"%outfile
        stdout, stderr = runBash(bgzip)
        print("bgzip out: %s; err: %s"%(stdout, stderr))
        stdout, stderr = runBash(tabix)
        print("tabix out: %s; err: %s"%(stdout, stderr))
    return

# No Need to do this
def SNPTSV2VCF(InpFil, OutFil, CHROM="", POS="", ID="", REF="", ALT="", QUAL="", FILTER="", EFFECT="", SE="", P=""):
    OutHand = open(OutFil, 'wt')
    OutHand.write("##fileformat=VCFv4.2\n")
    reader = csv.reader(open(InpFil, 'rt'), delimiter="\t")
    head = next(reader)
    idx_CHROM = head.index(CHROM)
    idx_POS = head.index(POS)
    idx_ID = head.index(ID)
    idx_REF = head.index(REF)
    idx_ALT = head.index(ALT)
    idx_QUAL = head.index(QUAL)
    idx_FILTER = head.index(QUAL)
    idx_EFFECT = head.index(EFFECT)
    idx_SE = head.index(SE)
    idx_P = head.index(P)
    for row in reader:
        CHROM = row[idx_CHROM]

    return

def SNPTSV2BED(InpFil, OutFil, CHROM, POS, ID="", REF="", ALT="", QUAL="", FILTER="", EFFECT="", SE="", P=""):
    OutHand = open(OutFil, 'wt')
    reader = csv.reader(open(InpFil, 'rt'), delimiter="\t")
    writer = csv.writer(open(OutFil, 'wt'), delimiter="\t")
    head = next(reader)
    writer.writerow(["#Chr", "Start", "End"] + head)
    idx_CHROM = head.index(CHROM)
    idx_POS = head.index(POS)
    for row in reader:
        CHROM = row[idx_CHROM]
        POS = row[idx_POS]
        row = [CHROM, POS, POS] + row
        writer.writerow(row)
    return

def gtf_info_parser(info):
    res = {}
    for term in info.split(";"):
        if term == "" or term=="\n":
            continue
        #print("|{}|".format(term))
        key, v = term.split()
        v = v.strip('"')
        res[key] = v
    return res

def GetGENEGTF(InpFil, OutFil):
    Fin = open(InpFil, 'rt')
    Fout = open(OutFil, 'wt')
    for l in Fin:
        if l.startswith("#"):
            Fout.write(l)
        else:
            llist = l.split("\t")
            if llist[2] != "gene":
                continue
            info = gtf_info_parser(llist[8])
            if info["gene_type"] != "protein_coding":
                continue
            Fout.write(l)

def ReadGeneGTF(InpFil, withCHR=False):
    res1 = {}
    res2 = {}
    Fin = open(InpFil, 'rt')
    for l in Fin:
        if l.startswith("#"):
            continue
        else:
            llist = l.strip().split("\t")
            info = gtf_info_parser(llist[8])
            CHR = llist[0]
            if not withCHR:
                CHR = CHR.lstrip("chr")
            start = int(llist[3])
            end = int(llist[4])
            gene_name = info["gene_name"]
            gene_id = info["gene_id"].split(".")[0]
            gene = GENE(CHR, start, end, gene_name, gene_id)
            res1[gene_id] = gene
            res2[gene_name] = gene
    return res1, res2

def SNP2Gene(Genes, SNPFil, Padding = 50000, method="nearest", CHROM="", POS="", EFFECT="", SE="", P=""):
    tbx = pysam.TabixFile(SNPFil) # bgziped tabixed SNP bed file
    head = tbx.header[0].split("\t")
    for gene in Genes.values():
        if gene.Chr in ["chrX", "chrY", "chrM", "X", "Y", "M"]:
            continue
        for row in tbx.fetch(gene.Chr, max(0, gene.Start - Padding), gene.End + Padding):
            row = row.split("\t")
            #print(row)
            #print(len(row))
            #if len(row) != 13:
            #    print(row)
            #    break
            row = dict(zip(head, row))
            snp = SNP(row[CHROM], row[POS], float(row[EFFECT]), row[SE], row[P])
            Genes[gene.GeneID].SNPs.append(snp) 
        max_eff = 0
        avg_eff = 0
        for i,snp in enumerate(Genes[gene.GeneID].SNPs):
            if snp.Effect > max_eff:
                max_eff = snp.Effect
            avg_eff += snp.Effect
        Genes[gene.GeneID].max_eff = max_eff
        Genes[gene.GeneID].avg_eff = avg_eff
    return Genes
            
def LoadPsychencodeRGN(Symbol2ENSG):
    Brain_Reg_Net2_csv = pd.read_csv("../dat/network/psychencode/INT-14_ElasticNet_Filtered_Cutoff_0.1_GRN_2.csv")
    Brain_Reg_Net2_csv_trim = Brain_Reg_Net2_csv.drop_duplicates(
        subset=["Transcription_Factor", "Target_Gene"], keep="first")
    edges, regions, weights = [], [], []
    TFs = []; Targets = []
    VerticeK = 0
    Vertices = {}
    UnmappedSymbols = set([])
    ENSGIDs = []
    for i, row in Brain_Reg_Net2_csv_trim.iterrows():
        symbol1, symbol2, region, weight = row["Transcription_Factor"], row["Target_Gene"], row["Enhancer_Region"], row["Edge_Weight"]
        try:
            g1, g2 = Symbol2ENSG[symbol1], Symbol2ENSG[symbol2] # Convert Symbol to ENSG ID, which we always does
        except:
            if g1 not in Symbol2ENSG:
                UnmappedSymbols.add(g1)
            if g2 not in Symbol2ENSG:
                UnmappedSymbols.add(g2)
            continue
        if g1 not in Vertices:
            Vertices[g1] = VerticeK
            VerticeK += 1
            ENSGIDs.append(g1)
        if g2 not in Vertices:
            Vertices[g2] = VerticeK
            VerticeK += 1
            ENSGIDs.append(g2)
        u, v = Vertices[g1], Vertices[g2]
        edges.append((u, v))
        regions.append(region)
        weights.append(float(weight))
        TF = row["Transcription_Factor"]; Target = row["Target_Gene"]
        TFs.append(TF); Targets.append(Target)
    RGN = ig.Graph(edges, edge_attrs={"weight": weights, "region":regions, "TF":TFs, "Taget":Targets})
    RGN.vs["ENSGID"] = ENSGIDs 
    ig.save(RGN, "../dat/network/saved/psychencode.rgn2.gml")
    return RGN

def LoadHumanBase(fil, Entrez2ENSG, pcut=0.2):
    HumanBaseNet = pd.read_csv(fil, delimiter="\t", header=None, names=["Gene1", "Gene2", "Prob"])
    HumanBaseNet = HumanBaseNet[HumanBaseNet["Prob"]>pcut]
    #Brain_Reg_Net2_csv_trim = Brain_Reg_Net2_csv.drop_duplicates(
    #    subset=["Transcription_Factor", "Target_Gene"], keep="first")
    #HumanBase
    edges, regions, weights = [], [], []
    TFs = []; Targets = []
    VerticeK = 0
    Vertices = {}
    UnmappedSymbols = set([])
    ENSGIDs = []
    for i, row in HumanBaseNet.iterrows():
        entrez1, entrez2, weight = row["Gene1"], row["Gene2"], row["Prob"]
        try:
            g1, g2 = Entrez2ENSG[entrez1], Entrez2ENSG[entrez2] # Convert Symbol to ENSG ID, which we always does
        except:
            if entrez1 not in Entrez2ENSG:
                UnmappedSymbols.add(entrez1)
            if entrez2 not in Entrez2ENSG:
                UnmappedSymbols.add(entrez2)
            continue
        if g1 not in Vertices:
            Vertices[g1] = VerticeK
            VerticeK += 1
            ENSGIDs.append(g1)
        if g2 not in Vertices:
            Vertices[g2] = VerticeK
            VerticeK += 1
            ENSGIDs.append(g2)
        u, v = Vertices[g1], Vertices[g2]
        edges.append((u, v))
        weights.append(float(weight))
    HBNet = ig.Graph(edges, edge_attrs={"weight": weights})
    HBNet.vs["ENSGID"] = ENSGIDs 
    #ig.save(HBNet, "../dat/network/saved/humanbasenet.gml")
    return HBNet, UnmappedSymbols

def volcano(pvalues, effects, title="volcano plot"):
    Qvalues = [-1*math.log(x,10) for x in pvalues]
    #effects = [math.log(x,2) for x in effects]
    effects = [x for x in effects]
    plt.scatter(effects, Qvalues, s=1)
    plt.title(title)
    plt.show()

def QQplot(pvalues, title="QQ plot"):
    pvalues = sorted(pvalues, reverse=True)
    Qvalues = []
    for x in pvalues:
        try:
            Qvalues.append(min(10, -math.log(x,10)))
        except:
            print(x)
    top = int(Qvalues[-1]) + 1
    NumTest = len(Qvalues)
    Qvalues = [0] * (19000-NumTest) + Qvalues
    Qexp = []
    for i in range(len(Qvalues)): 
        Qexp.append(float(i+1)/NumTest)
    Qexp.sort(reverse=True)
    Qexp = [-1*math.log(x,10) for x in Qexp]
    plt.subplot()
    plt.scatter(Qexp, Qvalues, s=2,  alpha=0.5)
    plt.plot([0, top], [0, top], ls="-")
    plt.title(title)
    plt.xlabel('Exp Q')
    plt.ylabel('Obs Q')
    plt.show()


def QQAndEffectplot(DF, CoreGenes, PLabel="P", Elabel="ZSTAT", title="QQ plot"):
    pvalues = DF[PLabel].values
    pvalues = sorted(pvalues, reverse=True)
    Qvalues = []
    for x in pvalues:
        try:
            Qvalues.append(min(10, -math.log(x,10)))
        except:
            print(x)
    top = int(Qvalues[-1]) + 1
    NumTest = len(Qvalues)
    Qvalues = [0] * (19000-NumTest) + Qvalues
    Qexp = []
    for i in range(len(Qvalues)): 
        Qexp.append(float(i+1)/NumTest)
    Qexp.sort(reverse=True)
    Qexp = [-1*math.log(x,10) for x in Qexp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3), dpi=120)
    ax1.scatter(Qexp, Qvalues, s=2,  alpha=0.5)
    ax1.plot([0, top], [0, top], ls="-", color="black")
    ax1.hlines(y=-math.log(0.05/19000, 10), xmin=0, xmax=top, color="grey", linestyles="dotted")
    ax1.set_title(title)
    ax1.set_xlabel('Exp Q')
    ax1.set_ylabel('Obs Q')

    ax2.hist(DF["ZSTAT"].values, density=1, color="blue", alpha=0.5)
    if len(CoreGenes):
        CoreZ = DF[DF.index.isin(CoreGenes)]["ZSTAT"].values
        ax2.hist(CoreZ, density=1, color="red", alpha=0.5)
    ax2.set_xlabel(Elabel)
    plt.show()
    return 

# PrePPI High-confidence predictions (prob>0.5) https://honiglab.c2b2.columbia.edu/PrePPI/ref/preppi_final600.txt.tar.gz
# PrePPI use Uniprot ID, need to convert to ENSID
def LoadNetworkPrePPI(Uniprot2ENSG):
    PrePPI_Fil = "/home/jw3514/Work/NB_proposal/dat/network/preppi_final600.txt"
    PrePPI = pd.read_csv(PrePPI_Fil, delimiter="\t")
    edges, weights = [], []
    VerticeK = 0
    Vertices = {}
    UnmappedUniprot = set([])
    ENSGIDs = []
    for i, row in PrePPI.iterrows():
        prot1, prot2, weight = row["prot1"], row["prot2"], row["final_score"]
        try:
            g1, g2 = Uniprot2ENSG[prot1], Uniprot2ENSG[prot2]
        except:
            if prot1 not in Uniprot2ENSG:
                UnmappedUniprot.add(prot1)
            if prot2 not in Uniprot2ENSG:
                UnmappedUniprot.add(prot2)
            continue
        if g1 not in Vertices:
            Vertices[g1] = VerticeK
            VerticeK += 1
            ENSGIDs.append(g1)
        if g2 not in Vertices:
            Vertices[g2] = VerticeK
            VerticeK += 1
            ENSGIDs.append(g2)
        u, v = Vertices[g1], Vertices[g2]
        edges.append((u, v))
        weights.append(float(weight))
    PPI = ig.Graph(edges, edge_attrs={"weight": weights})
    PPI.vs["ENSGID"] = ENSGIDs
    os.makedirs("../dat/network/saved", exist_ok=True)
    ig.save(PPI, "../dat/network/saved/PrePPI.gml")
    return PPI, UnmappedUniprot
    
    
def PlotEffDist(GeneDict):
    Effects = []
    for k, v in GeneDict.items():
        if len(v.SNPs) == 0:
            continue
        Effects.append(v.max_eff)
    plt.hist(Effects, bins=100)
    plt.xlim([0,2])
    plt.show()

def ProcessLoadGeneTestAndPlot(GeneTestFil, CoreGenes=None):
    GeneTest = pd.read_csv(GeneTestFil, index_col="GENE", delimiter="\t")
    QQAndEffectplot(GeneTest, CoreGenes)
    return GeneTest

def MapGenes2NetWork(GeneTest, subPrePPI, CoreGenes, Symbol2ENSG,ENSG2Symbol):
    #Core_Gene = GeneTest[GeneTest["ZSTAT"]>Zcut].index
    Core_Gene = CoreGenes
    ENSID2IDX = {}
    for idx, v in enumerate(subPrePPI.vs):
        ENSID2IDX[v["ENSGID"]] = idx
    degreesEff = {}
    degreesEff[0] = []
    degreesNode = {}
    degreesNode[0] = []
    unmapped = []
    for gene in Core_Gene:
        try:
            ens_id = Symbol2ENSG[gene]
            degreesEff[0].append(GeneTest.loc[gene, "ZSTAT"])
            degreesNode[0].append(ens_id)
        except:
            unmapped.append(gene)
    PreNodes = set(degreesNode[0])
    #print(PreNodes)
    Not_In_Network = []
    for i in range(1,11,1):
        degreesEff[i] = []
        degreesNode[i] = []
        for gene in degreesNode[i-1]:
            try:
                NodeID = ENSID2IDX[gene]
            except:
                Not_In_Network.append(gene)
                continue
            if gene in ENSID2IDX:
                for edge in subPrePPI.es.select(_source = NodeID):
                    target = edge.tuple[1]
                    if subPrePPI.vs[target]["ENSGID"] not in PreNodes:
                        degreesNode[i].append(subPrePPI.vs[target]["ENSGID"])
                for edge in subPrePPI.es.select(_target = NodeID):
                    source = edge.tuple[0]
                    if subPrePPI.vs[source]["ENSGID"] not in PreNodes:
                        degreesNode[i].append(subPrePPI.vs[source]["ENSGID"])
        degreesNode[i] = list(set(degreesNode[i]))
        for ensgid in degreesNode[i]:
            gene = ENSG2Symbol[ensgid]
            if gene in GeneTest.index:
                degreesEff[i].append(GeneTest.loc[gene, "ZSTAT"])
        #print(len(degreesNode[i]))
        PreNodes = PreNodes.union(set(degreesNode[i]))
    return degreesEff

def search_neighbors(node, graph):
    neighbors = []
    for edge in graph.es.select(_source = node):
        target = edge.tuple[1]
        neighbors.append(target)
    for edge in graph.es.select(_target = node):
        source = edge.tuple[0]
        neighbors.append(source)
    return list(set(neighbors))

def symbol2node(symbol, Symbol2ENSG, ENSID2IDX):
    try:
        ensid = Symbol2ENSG[symbol]
        node = ENSID2IDX[ensid]
    except:
        node = None
    return node


def Clusters_step1(GeneList, graph, Symbol2ENSG):
    ENSG2Symbol = {v: k for k,v in Symbol2ENSG.items()}
    ENSID2IDX = {}
    IDX2SYMBOL = {}
    for idx, v in enumerate(graph.vs):
        ENSID2IDX[v["ENSGID"]] = idx
        IDX2SYMBOL[idx] = ENSG2Symbol[v["ENSGID"]] 
    RES = {}
    for gene in GeneList:
        node = symbol2node(gene, Symbol2ENSG, ENSID2IDX)
        if node == None:
            continue
        RES[gene] = {}
        nb_1st = search_neighbors(node, graph)
        nb_1st = [IDX2SYMBOL[x] for x in nb_1st if IDX2SYMBOL[x] in GeneList]
        nb_2st = []
        for nb in nb_1st:
            nb_2st.extend(search_neighbors(nb, graph))
        nb_2st = list(set(nb_2st).difference(set(nb_1st)))
        nb_2st = [IDX2SYMBOL[x] for x in nb_2st if IDX2SYMBOL[x] in GeneList]
        RES[gene][1] = list(nb_1st)
        RES[gene][2] = list(nb_2st)
    return RES

def find_intersection(m_list):
    last_len = len(m_list)
    while 1:
        #print(len(m_list), m_list)
        BREAK = False
        for i in range(len(m_list)):
            for j in range(i+1, len(m_list)):
                if len(m_list[i].intersection(m_list[j])) > 0:
                    m_list[i] = m_list[i].union(m_list.pop(j))
                    BREAK = True
                    break
            if BREAK:
                break
        if len(m_list) == last_len:
            break
        last_len = len(m_list)
    return m_list

def Clusters_step2(RES, dist=1):
    Clusters = []
    for gene in RES.keys():
        if len(RES[gene][1]) < 2:
            continue
        if dist == 1:
            genes = [gene] + RES[gene][1]
        elif dist == 2:
            genes = [gene] + RES[gene][1] + RES[gene][2]
        genes = set(genes)
        #if gene == "CUL3":
        #    print(genes)
        FLAG_NEW_CLUST = True
        for i, cluster in enumerate(Clusters):
            if len(genes.intersection(cluster)) >= 1:
                Clusters[i] = cluster.union(genes)
                FLAG_NEW_CLUST = False
        if FLAG_NEW_CLUST:
            Clusters.append(genes)
    # Merge Clusters
    Clusters = find_intersection(Clusters) 
    return Clusters


def PlotDegreeDecay(_degreesEff, N = 5, Zcut = None):
    degreesEff = _degreesEff.copy()
    if Zcut != None:
        for i in range(len(degreesEff)):
            #print(degreesEff[i])
            degreesEff[i] = [x for x in degreesEff[i] if x >= Zcut]
            #print(degreesEff[i])
    data_to_plot = [degreesEff[i] for i in range(N)]
    means = ([round(np.mean(degreesEff[i]),4) for i in range(N)])
    std = ([round(np.std(degreesEff[i])/math.sqrt(len(degreesEff[i])),4) for i in range(N)])
    for i in range(N):
        print("%d\tmean=%.3f(+-%.3f)\tN=%d" % (i, means[i], std[i], len(degreesEff[i])))

    #print([round(len(degreesEff[i]),4) for i in range(N)])
    #print([round(np.median(degreesEff[i]),4) for i in range(N)])
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot)
    ax.scatter(range(1,N+1), [round(np.mean(degreesEff[i]),4) for i in range(N)])
    ax.plot(range(1,N+1), [round(np.mean(degreesEff[i]),4) for i in range(N)])
    #ax.errbar
    
    ax.set_xticklabels([str(i) for i in range(N)])
    ax.set_xlabel("Degree")
    #ax.set_ylabel("Effect size (Odds Ratio)")
    ax.set_ylabel("Effect size (ZSTAT)")
    plt.show()

def PairwiseTest(degreesEff, N=5):
    for i,j in itertools.combinations(range(N), 2):
        t, p = stats.mannwhitneyu(degreesEff[i], degreesEff[j])
        print("%d\t%d\t%.3e"%(i,j,p))

def GeneSymbol2nodes(graph, Genes, Symbol2ENSG):
    ENSID2IDX = {}
    for idx, v in enumerate(graph.vs):
        ENSID2IDX[v["ENSGID"]] = idx
        CoreGenePLen = []
    Genes = [Symbol2ENSG[g] for g in Genes if g in Symbol2ENSG]
    Nodes = [ENSID2IDX[g] for g in Genes if g in ENSID2IDX]
    return Nodes

def CompareShortestPathDist(graph, Genes, Symbol2ENSG):
    ENSID2IDX = {}
    for idx, v in enumerate(graph.vs):
        ENSID2IDX[v["ENSGID"]] = idx
    Genes = [Symbol2ENSG[g] for g in Genes if g in Symbol2ENSG]
    Nodes = [ENSID2IDX[g] for g in Genes if g in ENSID2IDX]
    #print(len(Nodes))
    CoreGenePLen = []
    for n1, n2 in itertools.combinations(Nodes, r=2):
        pLen = graph.shortest_paths_dijkstra(n1, n2, mode="ALL")[0][0]
        CoreGenePLen.append(pLen)
    random_nodes = np.random.choice(graph.vs, len(Nodes))
    RDMGenePLen = []
    for n1, n2 in itertools.combinations(random_nodes, r=2):
        pLen = graph.shortest_paths_dijkstra(n1, n2, mode="ALL")[0][0]
        RDMGenePLen.append(pLen)
    return CoreGenePLen, RDMGenePLen

def CompareShortestPathDist2(graph, dismat, Genes, Symbol2ENSG):
    ENSID2IDX = {}
    for idx, v in enumerate(graph.vs):
        ENSID2IDX[v["ENSGID"]] = idx
    Genes = [Symbol2ENSG[g] for g in Genes if g in Symbol2ENSG]
    Nodes = [ENSID2IDX[g] for g in Genes if g in ENSID2IDX]
    print(len(Nodes))
    CoreGenePLen = []
    for n1, n2 in itertools.combinations(Nodes, r=2):
        pLen = dismat[n1][n2]
        CoreGenePLen.append(pLen)
    RND_DAT = []
    for i in range(100):
        random_nodes = np.random.choice(graph.vs, len(Nodes), replace=False)
        RDMGenePLen = []
        for n1, n2 in itertools.combinations(random_nodes, r=2):
            pLen = dismat[n1.index][n2.index]
            RDMGenePLen.append(pLen)
        RND_DAT.append(RDMGenePLen)
    return CoreGenePLen,RND_DAT 

def SubgraphNodeIDMap(graph, subgraph, geneENS):
    pass

def CompareNearestPath(graph, dismat, Genes, Symbol2ENSG):
    ENSID2IDX = {}
    for idx, v in enumerate(graph.vs):
        ENSID2IDX[v["ENSGID"]] = idx
    Genes = [Symbol2ENSG[g] for g in Genes if g in Symbol2ENSG]
    Nodes = [ENSID2IDX[g] for g in Genes if g in ENSID2IDX]
    print(len(Nodes))
    CoreGenePLen = []
    for i_node in Nodes:
        dists = []
        for j_node in Nodes:
            if j_node == i_node:
                continue
            else:
                dists.append(dismat[i_node][j_node])
        pLen = min(dists) #dismat[n1][n2]
        CoreGenePLen.append(pLen)
    RND_DAT = []
    for i in range(100):
        random_nodes = np.random.choice(graph.vs, len(Nodes), replace=False)
        RDMGenePLen = []
        for i_node in random_nodes: # ditertools.combinations(random_nodes, r=2):
            dists = []
            for j_node in random_nodes:
                if i_node == j_node:
                    continue
                else:
                    dists.append(dismat[i_node.index][j_node.index])
            pLen = min(dists) #dismat[n1.index][n2.index]
            RDMGenePLen.append(pLen)
        RND_DAT.append(RDMGenePLen)
    return CoreGenePLen,RND_DAT 

def PlotShortestPathDist(CoreGenePLen, rndGenePLen):
    #CoreGenePLen = [x for x in CoreGenePLen if x < 100]
    pass

def PlotShortestPathDist(CoreGenePLen, rndGenePLen):
    #CoreGenePLen = [x for x in CoreGenePLen if x < 100]
    #rndGenePLen = [x for x in rndGenePLen if x < 100]
    cap = max([x for x in CoreGenePLen + rndGenePLen if x < 100])
    CoreGenePLen = [min(x,cap) for x in CoreGenePLen]
    rndGenePLen = [min(x,cap) for x in rndGenePLen]
    t, p = stats.mannwhitneyu(CoreGenePLen, rndGenePLen)
    print("Core gene avg:%.3f\trandom genes avg:%.3f\tp=%.3e"%(np.mean(CoreGenePLen), np.mean(rndGenePLen), p))
    bins = np.arange(0, max(CoreGenePLen)+1, 1)
    plt.hist(CoreGenePLen, bins=bins, align='left', alpha=0.5, color="red", label="Core genes")
    plt.hist(rndGenePLen, bins=bins, align='left', alpha=0.5, color="blue", label="Random genes")
    plt.legend()
    plt.show()

def ConnectivityTest(graph, GeneList, Symbol2ENSG):
    nodes = GeneSymbol2nodes(graph, GeneList, Symbol2ENSG)
    test_graph = graph.subgraph(nodes)
    res = len(test_graph.es)
    ES = []
    for i in range(1000):
        _nodes = np.random.choice(graph.vs, len(nodes))
        _test_sub_graph = graph.subgraph(_nodes)
        es = len(_test_sub_graph.es)
        ES.append(es)
    PlotPermutationP(ES, res)

