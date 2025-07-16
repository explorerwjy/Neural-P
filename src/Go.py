#!/usr/bin/env python3
"""
GO Annotation Database

A comprehensive tool for downloading, parsing, and querying GO annotations locally.
This allows efficient batch processing of GO terms without relying on external APIs.

Usage:
    # Download and initialize the database
    python go_annotation_db.py --download --data-dir /home/jw3514/Work/data/GeneOntology
    
    # Query genes for a specific GO term
    python go_annotation_db.py --go-id GO:0014069 --output psd_genes.csv --data-dir /home/jw3514/Work/data/GeneOntology
    
    # Search GO terms by keyword
    python go_annotation_db.py --search synapse --output synapse_terms.csv --data-dir /home/jw3514/Work/data/GeneOntology
"""

import os
import gzip
import pandas as pd
import urllib.request
import argparse
from tqdm import tqdm
import sys
import time

class GOAnnotationDatabase:
    """
    A class to download, parse, and query the Gene Ontology Annotation (GOA) database locally.
    """
    
    def __init__(self, data_dir="/home/jw3514/Work/data/GeneOntology"):
        """
        Initialize the GOAnnotationDatabase with a directory to store data.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store downloaded GO annotation files
            Default: "/home/jw3514/Work/data/GeneOntology"
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        
        # File paths
        self.goa_human_file = os.path.join(data_dir, "goa_human.gaf.gz")
        self.go_obo_file = os.path.join(data_dir, "go.obo")
        
        # DataFrames to store parsed data
        self.annotations_df = None
        self.go_terms_df = None
        
        # Column names for the GAF 2.2 format
        # http://geneontology.org/docs/go-annotation-file-gaf-format-2.2/
        self.gaf_columns = [
            "db", "db_object_id", "db_object_symbol", "qualifier", 
            "go_id", "db_reference", "evidence_code", "with_or_from", 
            "aspect", "db_object_name", "db_object_synonym", 
            "db_object_type", "taxon", "date", "assigned_by", 
            "annotation_extension", "gene_product_form_id"
        ]
    
    def download_goa_human(self, force=False):
        """
        Download the latest GOA human annotation file if not already present.
        
        Parameters:
        -----------
        force : bool
            If True, download the file even if it already exists
        """
        if force or not os.path.exists(self.goa_human_file):
            print(f"Downloading GOA human annotations to {self.goa_human_file}...")
            
            # URL for the GOA human annotation file
            url = "http://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.goa_human_file), exist_ok=True)
            
            # Download with progress tracking
            try:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
                    def report_hook(count, block_size, total_size):
                        if count == 0:
                            t.total = total_size
                        t.update(block_size)
                    
                    urllib.request.urlretrieve(url, self.goa_human_file, reporthook=report_hook)
                print(f"Download complete: {self.goa_human_file}")
            except Exception as e:
                print(f"Error downloading file: {str(e)}")
                if os.path.exists(self.goa_human_file):
                    os.remove(self.goa_human_file)
                raise
        else:
            print(f"Using existing GOA human annotations file: {self.goa_human_file}")
    
    def download_go_obo(self, force=False):
        """
        Download the latest GO OBO file if not already present.
        
        Parameters:
        -----------
        force : bool
            If True, download the file even if it already exists
        """
        if force or not os.path.exists(self.go_obo_file):
            print(f"Downloading GO OBO file to {self.go_obo_file}...")
            
            # URL for the GO OBO file
            url = "http://purl.obolibrary.org/obo/go.obo"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.go_obo_file), exist_ok=True)
            
            # Download with progress tracking
            try:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
                    def report_hook(count, block_size, total_size):
                        if count == 0:
                            t.total = total_size
                        t.update(block_size)
                    
                    urllib.request.urlretrieve(url, self.go_obo_file, reporthook=report_hook)
                print(f"Download complete: {self.go_obo_file}")
            except Exception as e:
                print(f"Error downloading file: {str(e)}")
                if os.path.exists(self.go_obo_file):
                    os.remove(self.go_obo_file)
                raise
        else:
            print(f"Using existing GO OBO file: {self.go_obo_file}")
    
    def download_all(self, force=False):
        """
        Download all required GO annotation files.
        
        Parameters:
        -----------
        force : bool
            If True, download files even if they already exist
        """
        self.download_goa_human(force=force)
        self.download_go_obo(force=force)
    
    def load_goa_human(self):
        """
        Load and parse the GOA human annotation file into a DataFrame.
        """
        print("Loading GOA human annotations...")
        
        # Check if file exists, download if not
        if not os.path.exists(self.goa_human_file):
            print(f"GOA human file not found at {self.goa_human_file}. Downloading...")
            self.download_goa_human()
        
        # Parse the GAF file
        annotations = []
        
        try:
            with gzip.open(self.goa_human_file, 'rt', encoding='utf-8') as f:
                # Skip header lines
                for line in tqdm(f, desc="Parsing annotations", unit="lines"):
                    if line.startswith('!'):
                        continue
                    
                    # Parse line
                    parts = line.strip().split('\t')
                    
                    # Ensure correct number of columns
                    if len(parts) != len(self.gaf_columns):
                        # Handle lines with missing columns
                        parts = parts + [''] * (len(self.gaf_columns) - len(parts))
                    
                    # Create a dictionary with named columns
                    annotation = dict(zip(self.gaf_columns, parts))
                    annotations.append(annotation)
                    
                    # Process in batches to save memory
                    if len(annotations) >= 100000:
                        if self.annotations_df is None:
                            self.annotations_df = pd.DataFrame(annotations)
                        else:
                            self.annotations_df = pd.concat([self.annotations_df, pd.DataFrame(annotations)])
                        annotations = []
            
            # Add any remaining annotations
            if annotations:
                if self.annotations_df is None:
                    self.annotations_df = pd.DataFrame(annotations)
                else:
                    self.annotations_df = pd.concat([self.annotations_df, pd.DataFrame(annotations)])
        
        except Exception as e:
            print(f"Error parsing GOA human file: {str(e)}")
            raise
        
        # Process data
        if self.annotations_df is not None and not self.annotations_df.empty:
            # Filter for human annotations (should be all human in goa_human, but double-check)
            self.annotations_df = self.annotations_df[self.annotations_df['taxon'].str.contains('9606', na=False)]
            
            # Add gene ID column if available
            if 'db_object_id' in self.annotations_df.columns:
                # Extract numerical Entrez Gene ID if available
                self.annotations_df['entrez_id'] = self.annotations_df['db_object_id'].apply(
                    lambda x: x.split(':')[-1] if ':' in str(x) else x
                )
            
            print(f"Loaded {len(self.annotations_df)} GO annotations for human genes.")
        else:
            print("No annotations were loaded.")
            self.annotations_df = pd.DataFrame(columns=self.gaf_columns + ['entrez_id'])
    
    def parse_go_obo(self):
        """
        Parse the GO OBO file to get term definitions and relationships.
        """
        # Check if file exists, download if not
        if not os.path.exists(self.go_obo_file):
            print(f"GO OBO file not found at {self.go_obo_file}. Downloading...")
            self.download_go_obo()
        
        print("Parsing GO OBO file...")
        
        terms = []
        current_term = None
        
        try:
            with open(self.go_obo_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Parsing GO terms", unit="lines"):
                    line = line.strip()
                    
                    if line == '[Term]':
                        if current_term:
                            terms.append(current_term)
                        current_term = {}
                    elif line == '':
                        continue
                    elif current_term is not None and ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip()
                        
                        if key in current_term:
                            if isinstance(current_term[key], list):
                                current_term[key].append(value)
                            else:
                                current_term[key] = [current_term[key], value]
                        else:
                            current_term[key] = value
                
                # Add the last term
                if current_term:
                    terms.append(current_term)
        
        except Exception as e:
            print(f"Error parsing GO OBO file: {str(e)}")
            raise
        
        # Convert to DataFrame
        self.go_terms_df = pd.DataFrame(terms)
        print(f"Parsed {len(self.go_terms_df)} GO terms.")
    
    def get_genes_by_go_term(self, go_id):
        """
        Retrieve all genes associated with a specific GO term ID.
        
        Parameters:
        -----------
        go_id : str
            The GO term ID (e.g., 'GO:0014069' for postsynaptic density)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing gene symbols and their Entrez IDs
        """
        # Ensure data is loaded
        if self.annotations_df is None:
            self.load_goa_human()
        
        # Query the annotations
        result = self.annotations_df[self.annotations_df['go_id'] == go_id]
        
        if len(result) == 0:
            print(f"No genes found for GO term {go_id}")
            return pd.DataFrame(columns=['gene_symbol', 'entrez_id'])
        
        # Extract unique genes
        genes_df = result[['db_object_symbol', 'entrez_id']].drop_duplicates()
        genes_df = genes_df.rename(columns={'db_object_symbol': 'gene_symbol'})
        
        print(f"Found {len(genes_df)} genes associated with {go_id}")
        return genes_df
    
    def get_genes_by_multiple_go_terms(self, go_ids):
        """
        Retrieve genes for multiple GO terms in batch.
        
        Parameters:
        -----------
        go_ids : list
            List of GO term IDs
            
        Returns:
        --------
        dict
            Dictionary mapping GO terms to DataFrames of associated genes
        """
        # Ensure data is loaded
        if self.annotations_df is None:
            self.load_goa_human()
        
        results = {}
        
        for go_id in tqdm(go_ids, desc="Processing GO terms"):
            genes_df = self.get_genes_by_go_term(go_id)
            results[go_id] = genes_df
        
        return results
    
    def get_parent_terms(self, go_id):
        """
        Get all parent terms of a given GO term.
        
        Parameters:
        -----------
        go_id : str
            The GO term ID
            
        Returns:
        --------
        set
            Set of parent GO term IDs
        """
        # Ensure GO terms are loaded
        if self.go_terms_df is None:
            self.parse_go_obo()
        
        # Find the term
        term = self.go_terms_df[self.go_terms_df['id'] == go_id]
        
        if len(term) == 0:
            return set()
        
        # Get direct parents (is_a relationships)
        parents = set()
        
        if 'is_a' in term.columns:
            is_a_values = term['is_a'].iloc[0]
            
            if isinstance(is_a_values, list):
                for is_a in is_a_values:
                    parent_id = is_a.split(' ')[0]
                    parents.add(parent_id)
                    # Recursively add parents of parents
                    parents.update(self.get_parent_terms(parent_id))
            elif isinstance(is_a_values, str):
                parent_id = is_a_values.split(' ')[0]
                parents.add(parent_id)
                # Recursively add parents of parents
                parents.update(self.get_parent_terms(parent_id))
        
        return parents
    
    def get_child_terms(self, go_id):
        """
        Get all child terms of a given GO term.
        
        Parameters:
        -----------
        go_id : str
            The GO term ID
            
        Returns:
        --------
        set
            Set of child GO term IDs
        """
        # Ensure GO terms are loaded
        if self.go_terms_df is None:
            self.parse_go_obo()
        
        # Find all terms that have this term as a parent
        child_terms = set()
        
        for _, term in self.go_terms_df.iterrows():
            if 'is_a' not in term:
                continue
                
            is_a_values = term['is_a']
            
            if isinstance(is_a_values, list):
                for is_a in is_a_values:
                    if go_id in is_a:
                        child_terms.add(term['id'])
                        # Recursively add children of children
                        child_terms.update(self.get_child_terms(term['id']))
            elif isinstance(is_a_values, str):
                if go_id in is_a_values:
                    child_terms.add(term['id'])
                    # Recursively add children of children
                    child_terms.update(self.get_child_terms(term['id']))
        
        return child_terms
    
    def get_all_genes_including_child_terms(self, go_id):
        """
        Get all genes associated with a GO term and all its child terms.
        
        Parameters:
        -----------
        go_id : str
            The GO term ID
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing gene symbols and their Entrez IDs
        """
        # Ensure data is loaded
        if self.annotations_df is None:
            self.load_goa_human()
        
        if self.go_terms_df is None:
            self.parse_go_obo()
        
        # Find all child terms
        child_terms = self.get_child_terms(go_id)
        all_terms = child_terms.union({go_id})
        
        print(f"Found {len(child_terms)} child terms for {go_id}")
        
        # Get genes for all terms
        all_genes = []
        
        for term in tqdm(all_terms, desc=f"Getting genes for {go_id} and child terms"):
            genes = self.get_genes_by_go_term(term)
            if not genes.empty:
                all_genes.append(genes)
        
        # Combine and remove duplicates
        if all_genes:
            combined_genes = pd.concat(all_genes).drop_duplicates()
            print(f"Found {len(combined_genes)} unique genes for {go_id} including child terms")
            return combined_genes
        else:
            print(f"No genes found for {go_id} including child terms")
            return pd.DataFrame(columns=['gene_symbol', 'entrez_id'])
    
    def search_go_terms(self, keyword):
        """
        Search for GO terms containing a keyword.
        
        Parameters:
        -----------
        keyword : str
            Keyword to search for in GO term names or definitions
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing matching GO terms
        """
        # Ensure GO terms are loaded
        if self.go_terms_df is None:
            self.parse_go_obo()
        
        # Search in name and definition
        matches = pd.DataFrame()
        
        if 'name' in self.go_terms_df.columns:
            name_matches = self.go_terms_df[
                self.go_terms_df['name'].str.contains(keyword, case=False, na=False)
            ]
            matches = pd.concat([matches, name_matches])
        
        if 'def' in self.go_terms_df.columns:
            def_matches = self.go_terms_df[
                self.go_terms_df['def'].str.contains(keyword, case=False, na=False)
            ]
            matches = pd.concat([matches, def_matches])
        
        # Remove duplicates
        matches = matches.drop_duplicates(subset=['id'])
        
        if len(matches) > 0:
            return matches[['id', 'name', 'def', 'namespace']]
        else:
            return pd.DataFrame(columns=['id', 'name', 'def', 'namespace'])


def batch_process_go_terms(go_term_file, output_dir, data_dir='/home/jw3514/Work/data/GeneOntology', include_children=False):
    """
    Process a batch of GO terms from a file and save results to separate CSV files.
    
    Parameters:
    -----------
    go_term_file : str
        Path to a file containing GO term IDs (one per line)
    output_dir : str
        Directory to save output CSV files
    data_dir : str
        Directory where the GO annotation database is stored
    include_children : bool
        If True, include genes from child terms
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Read GO terms from file
    with open(go_term_file, 'r') as f:
        go_terms = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(go_terms)} GO terms from {go_term_file}")
    
    # Initialize the database with the specified data directory
    print(f"Initializing GO database from {data_dir}")
    godb = GOAnnotationDatabase(data_dir=data_dir)
    
    # Download data if needed
    print("Checking for GO annotation files...")
    if not os.path.exists(os.path.join(data_dir, "goa_human.gaf.gz")):
        print("GOA human annotations file not found. Downloading...")
        godb.download_goa_human()
    if not os.path.exists(os.path.join(data_dir, "go.obo")):
        print("GO OBO file not found. Downloading...")
        godb.download_go_obo()
    
    # Load annotations
    godb.load_goa_human()
    
    if include_children:
        godb.parse_go_obo()
    
    # Process each GO term
    results = {}
    for go_id in tqdm(go_terms, desc="Processing GO terms"):
        if include_children:
            genes_df = godb.get_all_genes_including_child_terms(go_id)
        else:
            genes_df = godb.get_genes_by_go_term(go_id)
        
        results[go_id] = genes_df
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{go_id.replace(':', '_')}_genes.csv")
        genes_df.to_csv(output_file, index=False)
    
    # Create a summary file
    summary_data = []
    for go_id, df in results.items():
        summary_data.append({
            'go_id': go_id,
            'num_genes': len(df),
            'output_file': f"{go_id.replace(':', '_')}_genes.csv"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'go_terms_summary.csv'), index=False)
    
    print(f"Processed {len(go_terms)} GO terms. Results saved to {output_dir}")
    print(f"Summary saved to {os.path.join(output_dir, 'go_terms_summary.csv')}")


# Command line interface
def main():
    parser = argparse.ArgumentParser(description='Query GO annotations locally')
    parser.add_argument('--download', action='store_true', help='Download GO annotation files')
    parser.add_argument('--go-id', type=str, help='GO term ID to query')
    parser.add_argument('--go-terms-file', type=str, help='File containing multiple GO term IDs (one per line)')
    parser.add_argument('--search', type=str, help='Search GO terms by keyword')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--output-dir', type=str, help='Directory for batch output files')
    parser.add_argument('--data-dir', type=str, default='/home/jw3514/Work/data/GeneOntology', 
                        help='Directory to store/find GO annotation files (default: /home/jw3514/Work/data/GeneOntology)')
    parser.add_argument('--include-children', action='store_true', help='Include genes from child terms')
    parser.add_argument('--force-download', action='store_true', help='Force download even if files exist')
    
    args = parser.parse_args()
    
    # Initialize the database
    godb = GOAnnotationDatabase(data_dir=args.data_dir)
    
    # Download files if requested
    if args.download:
        print(f"Downloading GO annotation files to {args.data_dir}...")
        godb.download_all(force=args.force_download)
        print("Download complete.")
    
    # Process a file with multiple GO terms
    if args.go_terms_file:
        if not args.output_dir:
            print("Error: --output-dir is required when using --go-terms-file")
            sys.exit(1)
        
        batch_process_go_terms(
            args.go_terms_file, 
            args.output_dir, 
            data_dir=args.data_dir, 
            include_children=args.include_children
        )
    
    # Search for GO terms
    if args.search:
        godb.parse_go_obo()
        results = godb.search_go_terms(args.search)
        
        if len(results) > 0:
            print(f"Found {len(results)} GO terms matching '{args.search}':")
            print(results[['id', 'name', 'namespace']])
            
            if args.output:
                results.to_csv(args.output, index=False)
                print(f"Results saved to {args.output}")
        else:
            print(f"No GO terms found matching '{args.search}'")
    
    # Query genes for a specific GO term
    if args.go_id:
        godb.load_goa_human()
        
        if args.include_children:
            godb.parse_go_obo()
            genes = godb.get_all_genes_including_child_terms(args.go_id)
        else:
            genes = godb.get_genes_by_go_term(args.go_id)
        
        if len(genes) > 0:
            print(f"Found {len(genes)} genes associated with {args.go_id}:")
            print(genes.head(10))
            
            if args.output:
                genes.to_csv(args.output, index=False)
                print(f"Results saved to {args.output}")
        else:
            print(f"No genes found for GO term {args.go_id}")


# Example usage
if __name__ == "__main__":
    # Run the command line interface
    main()