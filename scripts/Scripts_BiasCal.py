# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# Scripts_BiasCal.py
# ========================================================================================================
import multiprocessing
from multiprocessing import Pool
import argparse
import sys

sys.path.insert(1, '/home/jw3514/Work/UNIMED/src/')
from CellType_PSY import *

def ABC_CellTypeBiasCal(Z2Mat, gw, Anno, outname):
    BiasDF = ABC_AvgCTZ_Weighted(Z2Mat, gw)
    BiasDF = add_class(BiasDF, Anno)
    BiasDF.to_csv(outname)
    return BiasDF

def HumanCellTypeBiasCal(Z2Mat, gw, Anno, outname):
    BiasDF = AvgSTRZ_Weighted(ExpZ2, gw, Method = 1)
    BiasDF = AnnotateCTDat(BiasDF, Anno)
    BiasDF.to_csv(outname)
    return BiasDF

def MouseSTRBiasCal(Z2Mat, gw, outname):
    BiasDF = AvgSTRZ_Weighted(Z2Mat, gw, Method = 1)
    BiasDF.to_csv(outname)
    return BiasDF


def BiasCal(gw_file, Prefix, MouseCT, MouseCT_Mat, MouseCT_Ann, HumanCT, HumanCT_Mat, HumanCT_Ann, MouseSTR, MouseSTR_Mat):
    gw = Fil2Dict(gw_file)
    if MouseCT:
        outname = "{}.ABC.Z2.csv".format(Prefix)
        ABC_CellTypeBiasCal(MouseCT_Mat, gw, MouseCT_Ann, outname)
    if HumanCT:
        outname = "{}.HCT.Z2.csv".format(Prefix)
        HumanCellTypeBiasCal(HumanCT_Mat, gw, HumanCT_Ann, outname)
    if MouseSTR:
        outname = "{}.STR.Z2.csv".format(Prefix)
        MouseSTRBiasCal(MouseSTR_Mat, gw, outname)

def GetOption():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InpFil', type=str, required=True, help='file contain list of gene weight files')
    parser.add_argument('--MouseCT', type=bool)
    parser.add_argument('--HumanCT', type=bool)
    parser.add_argument('--MouseSTR', type=bool)
    parser.add_argument('--outDir', type=str)
    args = parser.parse_args()
    return args

def process_file(filename, Z2Mat, outDIR, Model, Anno):
    name = ".".join(filename.split("/")[-1].split(".")[0:1])
    GWAS_DF = pd.read_csv(filename, sep="\t", index_col="GENE")
    GWAS_DF = GWAS_DF[GWAS_DF.index.isin(Z2Mat.index.values)]
    print(filename)
    print(name)
    if Model == "HumanCT":
        outname = f"{outDIR}/HumanCT.Bias.{name}.Z2.csv"
        HumanCellTypeBiasCal(Z2Mat, GWAS_DF, Anno, outname)
    elif Model == "MouseCT":
        outname = f"{outDIR}/MouseCT.Bias.{name}.Z2.csv"
        ABC_CellTypeBiasCal(Z2Mat, GWAS_DF, Anno, outname)
    elif Model == "MouseSTR":
        outname = f"{outDIR}/MouseSTR.Bias.{name}.Z2.csv"
        MouseSTRBiasCal(Z2Mat, GWAS_DF, Anno, outname)

def process_batch(files, Z2Mat, outDIR, Model, Anno):
    pool = multiprocessing.Pool(processes=20)
    pool.starmap(process_file, [(filename, Z2Mat, outDIR, Model, Anno) for filename in files])
    pool.close()
    pool.join()

def main():
    args = GetOption()
    input_files = [x.strip() for x in open(args.InpFil, 'rt').readlines()]
    outDIR = args.outDir if args.outDir else "."
    
    if args.mode == "HumanCT":
        Annotat = Anno
        if args.biasMat == None:
            Z2Mat = pd.read_csv("/home/jw3514/Work/CellType_Psy/dat/HumanCTExpressionMats/Human.Cluster.Log2Mean.Z1clip5.Z2.clip3.Dec30.csv", index_col=0)
        else:
            Z2Mat = pd.read_csv(args.biasMat, index_col=0)
        Z2Mat.columns = Z2Mat.columns.astype(int)
    elif args.mode == "MouseSTR":
        Annotat = STR2Region()
        Z2Mat = pd.read_csv("/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv", index_col=0)
    elif args.mode == "MouseCT":
        Annotat = pd.read_excel("../../data/Allen_Mouse_Brain_Cell_Atlas/SuppTables/41586_2023_6812_MOESM8_ESM.xlsx",
                          sheet_name="cluster_annotation", index_col="cluster_id_label")
        Z2Mat = pd.read_csv("/home/jw3514/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/Cluster_Z2Mat_ISHMatch.z1clip3.csv", index_col=0)
    
    process_batch(input_files, Z2Mat, outDIR, args.mode, Annotat)

if __name__ == '__main__':
    main()
