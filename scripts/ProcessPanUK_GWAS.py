#Author: jywang	explorerwjy@gmail.com

#========================================================================================================
# ProcessPanUK_GWAS.py
#========================================================================================================

import argparse
import pickle as pk
import csv
import gzip as gz
import numpy as np
import time 

class ProcessPanUK_GWAS:
    def __init__(self, args):
        self.Var2SNP = pk.load(open("/mnt/data0/UKBB_GWAS/Var2SNP.pk", 'rb'))
        self.Fname = args.input 
        self.outDIR = args.DIR

    def run(self):
        fin = gz.open(self.Fname, 'rt')
        head = fin.readline().strip().split("\t")
        New_Head = ["variant", "rsid", "beta_EUR", "se_EUR", "pval"]
        #New_Head = ["variant", "rsid", "af_cases_EUR", "af_controls_EUR", "beta_EUR", "se_EUR", "pval"]
        # idx                           20              26                32          38        44 neglog10
        idx_beta_EUR = head.index("beta_EUR")
        idx_se_EUR = head.index("se_EUR")
        idx_neglogpval_EUR = head.index("neglog10_pval_EUR") 
        outname = self.Fname.split("/")[-1]
        #outname = outname.rstrip(".tsv.bgz") + ".asso.tsv.gz"
        outname = outname.replace(".tsv.bgz", "") + ".asso.tsv.gz"
        outname = self.outDIR + "/" + outname
        print(outname)
        writer = csv.writer(gz.open(outname, 'wt'), delimiter="\t")
        writer.writerow(New_Head)
        N_unmapped = 0
        N_mapped = 0
        for l in fin.readlines():
            row = l.strip().split("\t")
            CHR, POS, REF, ALT = row[0:4]
            Var = "{}:{}:{}:{}".format(CHR, POS, REF, ALT)
            SNP = self.Var2SNP.get(Var, 0)
            if SNP == 0:
                N_unmapped += 1
                continue
            N_mapped += 1
            #af_cases_EUR = row[20]
            #af_controls_EUR = row[26]
            #beta_EUR = row[32]
            #se_EUR = row[38]
            #neglogpval = float(row[44])
            beta_EUR = row[idx_beta_EUR]
            se_EUR = row[idx_se_EUR]
            neglogpval = float(row[idx_neglogpval_EUR])
            pval = np.float_power(10, -neglogpval)
            #writer.writerow([Var, SNP, af_cases_EUR, af_controls_EUR, beta_EUR, se_EUR, pval])
            writer.writerow([Var, SNP, beta_EUR, se_EUR, pval])
        print("N_Mapped:{}   N_Unmapped:{}".format(N_mapped, N_unmapped))

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', required=True,  type=str, help = '')
    parser.add_argument('-d','--DIR', default="./", type=str, help = '')
    args = parser.parse_args()
    return args

def main():
    args = GetOptions()
    start_time = time.time()
    ins = ProcessPanUK_GWAS(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print("Elapsed time (Load SNPID):", elapsed_time, "seconds")
    ins.run()

    return

if __name__=='__main__':
    main()
