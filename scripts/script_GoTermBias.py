#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_bootstrapping_mutations.py
# ========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
sys.path.insert(1, '/home/jw3514/Work/UNIMED/src')
from CellType_PSY import *
from UNIMED import *

import multiprocessing
from multiprocessing import Pool


HumanCT_Mat = "/home/jw3514/Work/UNIMED/TDEP-sLDSC/data/cluster.specificity_matrix_entrez_percentile.csv"
MouseCT_Mat = "/home/jw3514/Work/CellType_Psy/dat/Test.BiasMat/MouseCT.cluster.filtTPM.spec.percentile.csv"
MouseSTR_Mat = "/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv"

#HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol, allen_mouse_genes = LoadGeneINFO()
Uniprot2Entrez = {}
HGNC = pd.read_csv("/home/jw3514/Work/data/GeneOntology/custom.txt", delimiter="\t", low_memory=False)
for i, row in HGNC.iterrows():
    Uniprot=row["UniProt ID(supplied by UniProt)"]
    Entrez=row["NCBI Gene ID(supplied by NCBI)"]
    if Uniprot!=Uniprot or Entrez !=Entrez:
        continue
    Uniprots = Uniprot.split()
    for _ in Uniprots:
        Uniprot2Entrez[_] = int(Entrez)
Go2Uniprot = pk.load(open("/home/jw3514/Work/CellType_Psy/dat3/Goterms/Go2Uniprot.pk", 'rb'))

from goatools.base import download_go_basic_obo
from goatools.base import download_ncbi_associations
from goatools.associations import read_ncbi_gene2go
from goatools.go_search import GoSearch
from goatools import obo_parser

obo_fname = download_go_basic_obo()
go = obo_parser.GODag(obo_fname)

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
    return df

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

class script_GotermBias:
    def __init__(self, args):
        #self.idx = args.idx
        #self.GoDF = pd.read_csv(args.goterms)
        self.GoDF = pd.read_csv("/home/jw3514/Work/CellType_Psy/dat3/Goterms/go.terms.selected.csv", sep="\t")
        self.ZscoreMatFil = args.mat_bias
        self.ExpMatFil = args.mat_exp
        self.mode = args.mode
        self.DIR = args.outdir
        self.n_processes = args.n_processes
    def run(self):
        os.makedirs("{}/Go_Biases/".format(self.DIR), exist_ok=True)
        #os.makedirs("{}/Go_ExpL/".format(self.DIR), exist_ok=True)
        #self.GoDF = pd.read_csv("../dat3/Goterms/go.terms.selected.csv", sep="\t")
        JobArrays = np.arange(self.GoDF.shape[0])
        if self.mode == "1" or self.mode == "human_ct":
            self.ZscoreMat = pd.read_csv(self.ZscoreMatFil, index_col=0)
            #max_Z, min_Z = 3, -3
            #self.ZscoreMat = self.ZscoreMat.clip(upper=max_Z, lower=min_Z)
            #self.ExpMat = pd.read_csv(self.ExpMatFil, index_col=0)
            self.Anno = Anno
            pool = multiprocessing.Pool(processes=self.n_processes)
            results = pool.starmap(self.run_human_ct, [(idx,) for idx in JobArrays])
            pool.close()
            pool.join()
        if self.mode == "2" or self.mode == "mouse_ct":
            JobArrays = JobArrays[1960:]
            self.ZscoreMat = pd.read_csv(self.ZscoreMatFil, index_col=0)
            #max_Z, min_Z = 3, -3
            #self.ZscoreMat = self.ZscoreMat.clip(upper=max_Z, lower=min_Z)
            #self.ExpMat = pd.read_csv(self.ExpMatFil, index_col=0)
            self.Anno = pd.read_excel("/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx", sheet_name = "cluster_annotation", index_col="cluster_id_label") 
            pool = multiprocessing.Pool(processes=self.n_processes)
            results = pool.starmap(self.run_mouse_ct, [(idx,) for idx in JobArrays])
            pool.close()
            pool.join()

        if self.mode == "3" or self.mode == "mouse_str":
            self.ZscoreMat = pd.read_csv(self.ZscoreMatFil, index_col=0)
            #self.ExpMat = pd.read_csv(self.ExpMatFil, index_col=0)
            pool = multiprocessing.Pool(processes=self.n_processes)
            print([(idx) for idx in JobArrays])
            results = pool.starmap(self.run_mouse_str, [(idx,) for idx in JobArrays])
            pool.close()
            pool.join()


    def run_human_ct(self, idx):
        try:
            _go = self.GoDF.loc[idx, "GoID"]
            print(_go)
            genes = GetGeneOfGo2(go, _go)
            gws = dict(zip(genes, [1]*len(genes)))
            fname = _go.split(":")[1]
            Go_Z2_Bias = AvgCTZ_Weighted(self.ZscoreMat, gws, Method = 1)
            Go_Z2_Bias = AnnotateCTDat(Go_Z2_Bias, self.Anno)
            Go_Z2_Bias.to_csv("{}/Go_Biases/GoBias.{}.Z2.csv".format(self.DIR, fname))
        except:
            print(_go, self.GoDF.loc[idx, "GoName"], "Not Found")
            pass

        #Go_ExpL = AvgCTZ_Weighted(self.ExpMat, gws, Method = 1)
        #Go_ExpL = AnnotateCTDat(Go_ExpL, self.Anno)
        #Go_ExpL.to_csv("{}/Go_ExpL/GoExpL.{}.csv".format(self.DIR, fname))

    def run_mouse_ct(self, idx):
        _go = self.GoDF.loc[idx, "GoID"]
        print(_go)
        genes = GetGeneOfGo2(go, _go)
        gws = dict(zip(genes, [1]*len(genes)))
        fname = _go.split(":")[1]
        Go_Z2_Bias = ABC_AvgCTZ_Weighted(self.ZscoreMat, gws)
        Go_Z2_Bias = add_class(Go_Z2_Bias, self.Anno)
        Go_Z2_Bias.to_csv("{}/Go_Biases/GoBias.{}.Z2.csv".format(self.DIR, fname))

        #Go_ExpL = ABC_AvgCTZ_Weighted(self.ExpMat, gws)
        #Go_ExpL = add_class(Go_ExpL, self.Anno)
        #Go_ExpL.to_csv("{}/Go_ExpL/GoExpL.{}.csv".format(self.DIR, fname))

    def run_mouse_str(self, idx):
        _go = self.GoDF.loc[idx, "GoID"]
        print(_go)
        genes = GetGeneOfGo2(go, _go)
        gws = dict(zip(genes, [1]*len(genes)))
        fname = _go.split(":")[1]
        Go_Z2_Bias = AvgSTRZ_Weighted(self.ZscoreMat, gws, Method = 1)
        Go_Z2_Bias.to_csv("{}/Go_Biases/GoBias.{}.Z2.csv".format(self.DIR, fname))

        # Go_ExpL = AvgSTRZ_Weighted(self.ExpMat, gws, Method = 1)
        # Go_ExpL.to_csv("{}/Go_ExpL/GoExpL.{}.csv".format(self.DIR, fname))

def GetOptions():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--idx', type=int, required=True, help='index')
    parser.add_argument('-m', '--mode', type=str, help="mode in [1:human ct; 2:mouse ct; 3:mouse str]")
    parser.add_argument('--mat_exp', type=str, help='expression matrix')
    parser.add_argument('--mat_bias', type=str, help='bias matrix')
    parser.add_argument('--outdir', type=str, help='output dir')
    # add number of processes
    parser.add_argument('--n_processes', type=int, default=20, help='number of processes')
    args = parser.parse_args()

    return args


def main():
    args = GetOptions()
    ins = script_GotermBias(args)
    ins.run()
    return


if __name__ == '__main__':
    main()
