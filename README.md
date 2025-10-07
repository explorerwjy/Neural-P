# Neural-P: Cell Type-Specific Analysis of Psychiatric GWAS Data

**Neural-P** is a computational framework for analyzing psychiatric genomics data through cell type-specific expression patterns and bias calculations. The pipeline enables systematic investigation of how genetic associations with psychiatric disorders map onto specific cell types and brain structures.

## Overview

Neural-P implements multiple bias calculation algorithms to assess enrichment of GWAS-identified genes in cell type-specific expression profiles:

- **AvgZ**: Average Z-score based bias calculation
- **CT_Correlation**: Cell type correlation-based enrichment
- **Top Gene Enrichment**: Analysis of top-ranked genes in cell type expression profiles

The framework supports analysis across:
- **HumanCT**: Human cell types
- **MouseCT**: Mouse cell types
- **MouseSTR**: Mouse brain structures (Allen Brain Atlas)

## Key Features

- Automated Snakemake workflow for reproducible analysis
- Parallel processing for computational efficiency
- Principal Component Analysis (PCA) of bias matrices
- Gene Ontology (GO) term enrichment analysis
- Support for custom gene exclusion lists (e.g., synaptic genes, rare variants)
- Modular architecture for easy extension

## Installation

### Requirements

- Python 3.8+
- Conda/Mamba package manager
- Snakemake workflow manager

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/explorerwjy/Neural-P.git
cd Neural-P
```

2. Create and activate the conda environment:
```bash
# Create environment with required packages
conda create -n neuralp python=3.9
conda activate neuralp

# Install core dependencies
conda install -c conda-forge snakemake pandas numpy scipy scikit-learn matplotlib statsmodels requests
conda install -c conda-forge jupyterlab ipython

# Install additional dependencies
pip install pyarrow  # for parquet file support
```

3. Install external dependencies:

Neural-P depends on the `CellType_PSY` module for cell type analysis utilities. Ensure this is available in your Python path or install separately.

### Configuration

Edit `config/config.yaml` to specify:
- GWAS gene list paths
- Expression matrix files
- Gene exclusion lists
- Output directories

Key configuration parameters:
```yaml
gwas_sources:
  - PGC  # Add your GWAS source

gwas_lists:
  PGC: "/path/to/GWAS.Magma.Gene.list"

bias_matrices:
  HumanCT: "/path/to/HumanCT.expression.parquet"
  MouseCT: "/path/to/MouseCT.expression.parquet"
  MouseSTR: "/path/to/AllenMouseBrain.parquet"
```

## Usage

### Running the Main Pipeline

The Snakemake pipeline orchestrates all analysis steps:

```bash
# Dry run to preview workflow
snakemake -n

# Execute complete pipeline with 20 cores
snakemake --cores 20

# Generate workflow visualization
snakemake --dag | dot -Tpdf > workflow.pdf
```

### Pipeline Steps

1. **Bias Calculation**: Computes cell type enrichment for GWAS gene sets
2. **PCA Analysis**: Performs dimensionality reduction on bias matrices

Output structure:
```
results/
├── assoc/{source}/{mode}/{bias_type}.{exclude}/
│   └── *_bias.csv
└── pca/{source}/{mode}/{bias_type}.{exclude}/
    └── *_pca.tsv
```

### Running Specific Analyses

**Bias calculations only:**
```bash
snakemake run_bias_batch --cores 20
```

**PCA analysis only:**
```bash
snakemake run_pca_batch --cores 20
```

**Manual script execution:**
```bash
# Run bias calculation
python scripts/script_gwas_bias_AvgZ.py \
    --InpFil data/GWAS_genes.txt \
    --outDir results/test \
    --mode HumanCT \
    --biasMat data/HumanCT.parquet \
    --processes 20 \
    --exclude data/exclude_genes.txt

# Run PCA analysis
python scripts/analyze_bias_pca.py results/assoc/PGC/HumanCT/AvgZ.Empty \
    --output_dir results/pca/PGC/HumanCT/AvgZ.Empty
```

### Interactive Analysis with Jupyter Notebooks

Specialized analyses are available in Jupyter notebooks:

```bash
# Launch Jupyter Lab
jupyter lab

# Key notebooks:
# - notebooks/notebook_PGC_HumanCT.ipynb: Human cell type analysis
# - notebooks/notebook_PGC_MouseCT.ipynb: Mouse cell type analysis
# - notebooks/notebook_PGC_MouseSTR.ipynb: Mouse brain structure analysis
# - notebooks/GoTermAnalysis.ipynb: Gene Ontology enrichment
# - notebooks/P_Factor_PanInf.ipynb: P-factor analysis
```

### Input Data Format

**GWAS Gene Lists** (MAGMA format):
```
GENE    CHR    START    STOP    NSNPS    NPARAM    N    ZSTAT    P
GENE1   1      12345    67890   10       5         1000  2.5      0.012
GENE2   2      23456    78901   15       5         1000  1.8      0.072
```

**Expression Matrices**: Parquet files with genes as rows, cell types/structures as columns

**Exclusion Lists**: Text files with gene symbols (one per line) or CSV with Entrez IDs

## Methods

### Bias Calculation Algorithms

1. **AvgZ**: Computes average Z-scores of gene expression across cell types for GWAS-associated genes
2. **CT_Correlation**: Calculates correlation between GWAS statistics and cell type-specific expression
3. **Top Gene Enrichment**: Tests enrichment of GWAS genes in top percentiles of cell type expression

### Gene Exclusion Strategies

- **Empty**: No gene exclusion
- **Synapse**: Exclude synaptic genes
- **SynapseTopRare**: Exclude synaptic genes and top rare variant genes

### Statistical Analysis

- Principal Component Analysis (PCA) for dimensionality reduction
- Hierarchical clustering of cell types
- Multiple testing correction (FDR)

## Project Structure

```
Neural-P/
├── config/              # Configuration files
│   └── config.yaml
├── dat/                 # Input data (not tracked)
│   ├── ExpMat/         # Expression matrices
│   └── GoTerms/        # Gene Ontology data
├── scripts/            # Analysis scripts
│   ├── script_gwas_bias_*.py
│   └── analyze_bias_pca.py
├── src/                # Core modules
│   ├── UNIMED.py
│   └── analyze_bias_pca.py
├── notebooks/          # Jupyter analysis notebooks
├── results/            # Output files (not tracked)
├── Snakefile          # Workflow definition
└── README.md
```

## Citation

If you use Neural-P in your research, please cite:

```
[Citation to be added upon publication]
```

## Dependencies

Core Python packages:
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- statsmodels
- snakemake

External resources:
- CellType_PSY module
- MAGMA gene analysis tool (for input preparation)

## Acknowledgments

Development assisted by Claude Code.

## License

[License to be determined]

## Contact

For questions and support, please open an issue on GitHub.
