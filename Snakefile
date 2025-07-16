MODES = config["assoc_modes"]
BIAS_TYPES = config["bias_types"]
EXCLUDE_LIST = config["exclude_gene_list"]
BIAS_MATS = config["bias_matrices"]
EXCLUDE_FILES = config["exclude_files"]
GWAS_SOURCES = config["gwas_sources"]

ROOT = "/home/jw3514/Work/UNIMED"

rule all:
    input:
        expand(ROOT + "/results/pca/{source}/{mode}/{bias}.{exclude}_pca.tsv",
               source=GWAS_SOURCES,
               mode=MODES,
               bias=BIAS_TYPES,
               exclude=EXCLUDE_LIST)

rule run_bias_batch:
    input:
        gwas_list=lambda wc: config["gwas_lists"][wc.source],
        biasmat=lambda wc: BIAS_MATS[wc.mode],
        excludelist=lambda wc: EXCLUDE_FILES[wc.exclude]
    output:
        done=ROOT + "/results/assoc/{source}/{mode}/{bias}.{exclude}/.run_bias_batch_done"
    params:
        outdir=ROOT + "/results/assoc/{source}/{mode}/{bias}.{exclude}"
    threads: 8
    shell:
        """
        echo "Working directory: $(pwd)"
        echo "Creating directory: {params.outdir}"
        mkdir -p "{params.outdir}"
        echo "Directory created, contents:"
        ls -la "{params.outdir}/"

        cd {ROOT}
        python scripts/script_gwas_bias_{wildcards.bias}.py \
            --InpFil "{input.gwas_list}" \
            --outDir "{params.outdir}" \
            --mode {wildcards.mode} \
            --biasMat "{input.biasmat}" \
            --processes {threads} \
            --exclude "{input.excludelist}"

        touch "{output.done}"
        """

rule run_pca_batch:
    input:
        done=ROOT + "/results/assoc/{source}/{mode}/{bias}.{exclude}/.run_bias_batch_done"
    output:
        ROOT + "/results/pca/{source}/{mode}/{bias}.{exclude}_pca.tsv"
    params:
        assoc_dir=ROOT + "/results/assoc/{source}/{mode}/{bias}.{exclude}",
        pca_output_dir=ROOT + "/results/pca/{source}/{mode}/{bias}.{exclude}"
    shell:
        """
        cd {ROOT}
        
        # Create unique output directory for this specific combination
        mkdir -p {params.pca_output_dir}
        
        python scripts/analyze_bias_pca.py {params.assoc_dir} \
            --output_dir {params.pca_output_dir}
        
        # Copy the PC scores to the expected output filename
        cp {params.pca_output_dir}/pc_scores.csv {output}
        """