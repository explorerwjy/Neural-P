#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_control_simulation.py
# Geretate Random Genes with specified Mutation Weights and Cal. Bias
# ========================================================================================================

import argparse
import sys
sys.path.insert(1, '/home/jw3514/Work/UNIMED/src/')
from CellType_PSY import *

def add_class(BiasDF, ClusterAnn):
    for cluster, row in BiasDF.iterrows():
        BiasDF.loc[cluster, "class_id_label"] = ClusterAnn.loc[cluster, "class_id_label"]
        BiasDF.loc[cluster, "CCF_broad.freq"] = ClusterAnn.loc[cluster, "CCF_broad.freq"]
        BiasDF.loc[cluster, "CCF_acronym.freq"] = ClusterAnn.loc[cluster, "CCF_acronym.freq"]
        BiasDF.loc[cluster, "v3.size"] = ClusterAnn.loc[cluster, "v3.size"]
        BiasDF.loc[cluster, "v2.size"] = ClusterAnn.loc[cluster, "v2.size"]
    return BiasDF
def GenerateRand_MouseCT(Arr, NGenes, Z2Mat, Anno, OutDIR):
    RandG = np.random.choice(Z2Mat.index.values, NGenes)
    RandGW = dict(zip(RandG, [1]*NGenes))
    Dict2Fil(RandGW, "{}/GW.Rand{}.{}.Z2.csv".format(OutDIR, NGenes, Arr))
    Rand_Bias = ABC_AvgCTZ_Weighted(Z2Mat, RandGW)
    Rand_Bias = add_class(Rand_Bias, Anno)
    Rand_Bias.to_csv("{}/Bias.Rand{}.{}.Z2.csv".format(OutDIR, NGenes, Arr))
    return (RandGW, Rand_Bias)

def GenerateRand_MouseSTR(Arr, NGenes, Z2Mat, OutDIR):
    RandG = np.random.choice(Z2Mat.index.values, NGenes)
    RandGW = dict(zip(RandG, [1]*NGenes))
    Dict2Fil(RandGW, "{}/GW.Rand{}.{}.Z2.csv".format(OutDIR, NGenes, Arr))
    Rand_Bias = AvgSTRZ_Weighted(Z2Mat, RandGW, Method = 1,
          csv_fil = "{}/Bias.Rand{}.{}.Z2.csv".format(OutDIR, NGenes, Arr))
    return (RandGW, Rand_Bias)

def GenerateRand_HumanCT(Arr, NGenes, Z2Mat, Anno, OutDIR):
    RandG = np.random.choice(Z2Mat.index.values, NGenes)
    RandGW = dict(zip(RandG, [1]*NGenes))
    Dict2Fil(RandGW, "{}/GW.Rand{}.{}.Z2.csv".format(OutDIR, NGenes, Arr))
    Rand_Bias = AvgCTZ_Weighted(Z2Mat, RandGW, Method = 1)
    Rand_Bias = AnnotateCTDat(Rand_Bias, Anno)
    Rand_Bias.to_csv("{}/Bias.Rand{}.{}.Z2.csv".format(OutDIR, NGenes, Arr))
    return (RandGW, Rand_Bias)

class script_control_simulation:
    def __init__(self, args):
        self.Data = args.Data
        self.matrix = args.matrix
        self.NGenes = int(args.NGenes)
        self.arr = args.input
        self.outDir = args.location
        self.weights = args.weights
        self.prob = args.prob

    def run(self):
        Z2Mat = pd.read_csv(self.matrix, index_col=0)
        print(self.NGenes)
        if self.Data == "MouseCT":
            Anno = pd.read_excel("/home/jw3514/Work/data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx", sheet_name = "cluster_annotation", index_col="cluster_id_label")
            GenerateRand_MouseCT(self.arr, self.NGenes, Z2Mat, Anno, self.outDir)
        elif self.Data == "MouseSTR":
            GenerateRand_MouseSTR(self.arr, self.NGenes, Z2Mat, self.outDir)
        elif self.Data == "HumanCT":
            Anno = LoadHumanCTAnno()
            GenerateRand_HumanCT(self.arr, self.NGenes, Z2Mat, Anno, self.outDir)


def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=str, required=True, help='Which Col from Match Set')
    parser.add_argument('--Data', type=str, required=True, help='Which Dataset using')
    parser.add_argument('--NGenes', type=int, required=True,
                        help='Number of genes')
    parser.add_argument('--weights', type=str, help='Which Col from Match Set')
    parser.add_argument('--prob', type=str, help='Fil with Probability of each gene being sampled [optional]')
    parser.add_argument('-i', '--input', type=int,
                        required=True,  help='Index of simulation')
    parser.add_argument('-l', '--location', type=str,
                        default="dat/SimulateControlBiasDefault/",  help='location to store results')

    parser.add_argument('--mat', type=str)
    parser.add_argument('--graph', type=str)
    args = parser.parse_args()

    return args

def main():
    args = GetOptions()
    ins = script_control_simulation(args)
    ins.run()

    return


if __name__ == '__main__':
    main()
    print("Done")
