#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

# ========================================================================================================
# script_GoTermBias.py
# Module 1: Query GO terms and save gene lists to JSON
# Module 2: Compute GO term biases using saved gene lists
# ========================================================================================================

import argparse
import sys
import json
import os
import pandas as pd
import gzip
import requests
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
sys.path.insert(1, '/home/jw3514/Work/UNIMED/src')
from CellType_PSY import *
from UNIMED import *
import multiprocessing

# Constants
HumanCT_Mat = "/home/jw3514/Work/CellType_Psy/dat/Test.BiasMat/HumanCT.TPMFilt.Spec.Percentile.csv"
MouseCT_Mat = "/home/jw3514/Work/CellType_Psy/dat/Test.BiasMat/MouseCT.cluster.filtTPM.spec.percentile.csv"
MouseSTR_Mat = "/home/jw3514/Work/ASD_Circuits/dat/allen-mouse-exp/AllenMouseBrain_Z2bias.csv"

HumanCT_SaveDir = "/home/jw3514/Work/UNIMED/dat/GoTerms/HumanCT/"
MouseCT_SaveDir = "/home/jw3514/Work/UNIMED/dat/GoTerms/MouseCT/"
MouseSTR_SaveDir = "/home/jw3514/Work/UNIMED/dat/GoTerms/MouseSTR/"

def load_go_resources():
    """Load GO-related resources only when needed for query mode"""
    from goatools.base import download_go_basic_obo
    from goatools import obo_parser
    
    # Load gene mappings
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

    # Load GO DAG
    obo_fname = download_go_basic_obo()
    go = obo_parser.GODag(obo_fname)
    
    return Uniprot2Entrez, Go2Uniprot, go

def GetALLGo(go, GoID):
    Root = go[GoID]
    all_go = Root.get_all_children()
    all_go.add(GoID)
    return all_go

def GetGeneOfGo2(go, GoID, Go2Uniprot, Uniprot2Entrez):
    goset = GetALLGo(go, GoID)
    Total_Genes = set([])
    for i, tmpgo in enumerate(goset):
        if tmpgo in Go2Uniprot:
            geneset = set([Uniprot2Entrez.get(x, 0) for x in Go2Uniprot[tmpgo]])
            Total_Genes = Total_Genes.union(geneset)
    return Total_Genes

class GoTermQuery:
    def __init__(self, goterms_file, output_json):
        self.GoDF = pd.read_csv(goterms_file, sep="\t")
        self.output_json = output_json
        # Load GO resources only when query mode is used
        self.Uniprot2Entrez, self.Go2Uniprot, self.go = load_go_resources()
        
    def run(self):
        go_gene_dict = {}
        for idx in range(self.GoDF.shape[0]):
            _go = self.GoDF.loc[idx, "GoID"]
            try:
                print(f"Processing {_go}")
                genes = GetGeneOfGo2(self.go, _go, self.Go2Uniprot, self.Uniprot2Entrez)
                if genes:
                    go_gene_dict[_go] = list(genes)
            except:
                print(f"{_go} ({self.GoDF.loc[idx, 'GoName']}) not found")
                continue
                
        with open(self.output_json, 'w') as f:
            json.dump(go_gene_dict, f)
        print(f"Saved GO term gene lists to {self.output_json}")

class GoBiasCompute:
    def __init__(self, args):
        self.go_gene_file = args.go_gene_file
        self.n_processes = args.n_processes
        
        # Load GO:gene mappings
        with open(self.go_gene_file) as f:
            self.go_gene_dict = json.load(f)
            
        # Load matrices
        self.human_ct_mat = pd.read_csv(HumanCT_Mat, index_col=0)
        self.mouse_ct_mat = pd.read_csv(MouseCT_Mat, index_col=0)
        self.mouse_str_mat = pd.read_csv(MouseSTR_Mat, index_col=0)
        
        # Load annotations
        self.mouse_ct_anno = pd.read_csv("/home/jw3514/Work/UNIMED/dat/MouseCT_Cluster_Anno.csv", index_col="cluster_id_label")
            
    def run(self):
        # Create output directories
        os.makedirs(f"{HumanCT_SaveDir}/Go_Biases/", exist_ok=True)
        os.makedirs(f"{MouseCT_SaveDir}/Go_Biases/", exist_ok=True) 
        os.makedirs(f"{MouseSTR_SaveDir}/Go_Biases/", exist_ok=True)

        # Process all GO terms in parallel
        pool = multiprocessing.Pool(processes=self.n_processes)
        results = pool.starmap(self.process_go_term, [(go_id,) for go_id in self.go_gene_dict.keys()])
        pool.close()
        pool.join()

    def process_go_term(self, go_id):
        try:
            genes = self.go_gene_dict[go_id]
            gws = dict(zip(genes, [1]*len(genes)))
            fname = go_id.split(":")[1]
            
            # Human cell types
            try:
                Go_Z2_Bias = HumanCT_AvgZ_Weighted(self.human_ct_mat, gws)
                Go_Z2_Bias = AnnotateCTDat(Go_Z2_Bias, Anno)
                Go_Z2_Bias.to_csv(f"{HumanCT_SaveDir}/Go_Biases/GoBias.{fname}.specpecentile.csv")
            except Exception as e:
                print(f"Error processing human CT for {go_id}: {str(e)}")
            
            # # Mouse cell types  
            # try:
            #     Go_Z2_Bias = MouseCT_AvgZ_Weighted(self.mouse_ct_mat, gws)
            #     Go_Z2_Bias = add_class(Go_Z2_Bias, self.mouse_ct_anno)
            #     Go_Z2_Bias.to_csv(f"{MouseCT_SaveDir}/Go_Biases/GoBias.{fname}.specpecentile.csv")
            # except Exception as e:
            #     print(f"Error processing mouse CT for {go_id}: {str(e)}")
                
            # # Mouse structures
            # try:
            #     Go_Z2_Bias = MouseSTR_AvgZ_Weighted(self.mouse_str_mat, gws)
            #     Go_Z2_Bias.to_csv(f"{MouseSTR_SaveDir}/Go_Biases/GoBias.{fname}.specpecentile.csv")
            # except Exception as e:
            #     print(f"Error processing mouse STR for {go_id}: {str(e)}")
                
        except Exception as e:
            print(f"Error processing {go_id}: {str(e)}")

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['query', 'compute'], required=True,
                      help='Run mode: query GO terms or compute biases')
    parser.add_argument('--goterms', type=str, help='GO terms input file')
    parser.add_argument('--go_gene_file', type=str, default="/home/jw3514/Work/UNIMED/dat/GoTerms/genes_method_uniprot.json", help='JSON file with GO:gene mappings')
    parser.add_argument('--n_processes', type=int, default=20, help='Number of processes')
    return parser.parse_args()

def main():
    args = GetOptions()
    
    if args.mode == 'query':
        if not args.goterms or not args.go_gene_file:
            print("Error: --goterms and --go_gene_file required for query mode")
            sys.exit(1)
        querier = GoTermQuery(args.goterms, args.go_gene_file)
        querier.run()
        
    elif args.mode == 'compute':
        if not args.go_gene_file:
            print("Error: --go_gene_file required for compute mode")
            sys.exit(1)
        computer = GoBiasCompute(args)
        computer.run()

if __name__ == '__main__':
    main()

# usage example
# python script_GoTermBias.v2.py --mode query --goterms /home/jw3514/Work/UNIMED/dat/GoTerms/GoTerm.txt --go_gene_file /home/jw3514/Work/UNIMED/dat/GoTerms/genes_method_uniprot.json
# python script_GoTermBias.v2.py --mode compute --go_gene_file /home/jw3514/Work/UNIMED/dat/GoTerms/genes_method_uniprot.json
