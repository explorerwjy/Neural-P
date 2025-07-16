import os
import sys
sys.path.insert(1, '/home/jw3514/Work/CellType_Psy/src')
sys.path.insert(1, '/home/jw3514/Work/UNIMED/src')
from CellType_PSY import *
from UNIMED import *
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict
import argparse
import pandas as pd
from scipy import stats
import numpy as np

def process_go_bias(args: Tuple[str, str, pd.DataFrame]) -> Tuple[str, str, float]:
    """
    Process a single GO bias file and calculate correlation with P-factor.
    
    Args:
        args: Tuple containing (GO ID, GO Name, MouseCT_bias_pc_scores_df)
        
    Returns:
        Tuple of (GO ID, GO Name, correlation values for PC1-PC5)
    """
    go_id, go_name, mouse_ct_df = args
    try:
        # Read the GO bias file
        go_bias = pd.read_csv(f"{GoBiasDIR}/GoBias.{go_id[3:]}.specpecentile.csv", index_col=0)
        
        # Add PC1-PC5 scores
        for pc in range(1,6):
            for ct in go_bias.index.values:
                go_bias.loc[ct, f"EFFECT_PC{pc}"] = mouse_ct_df.loc[ct, f"PC{pc}"]
        
        #print(go_bias.sort_values("EFFECT_PC1", ascending=False).head(10))
        # Drop any rows with NaN values
        #go_bias = go_bias.dropna()
        
        results = []
        results.append(go_id)
        results.append(go_name)
        
        # Calculate correlations for each PC
        for pc in range(1,6):
            # Drop rows with NaN values before computing correlation
            valid_data = pd.DataFrame({
                'effect': go_bias["EFFECT"],
                'pc': go_bias[f"EFFECT_PC{pc}"]
            }).dropna()
            
            r_bias_pc, p_bias_pc = stats.pearsonr(valid_data['effect'], valid_data['pc'])
            r_bias_pc_spearman, p_bias_pc_spearman = stats.spearmanr(valid_data['effect'], valid_data['pc'])
            results.extend([r_bias_pc, p_bias_pc, r_bias_pc_spearman, p_bias_pc_spearman])
            
        return tuple(results)
    except Exception as e:
        print(f"Error processing {go_id}: {str(e)}")
        # Return NaN for all correlation values
        return (go_id, go_name, *([np.nan] * 20))

def main(go_df_path: str, go_bias_dir: str, mouse_ct_pc_path: str, output_path: str, n_workers: int = None):
    """
    Main function to process GO bias files in parallel.
    
    Args:
        go_df_path: Path to the GO DataFrame file
        go_bias_dir: Directory containing GO bias files
        mouse_ct_pc_path: Path to the mouse cell type PC scores file
        output_path: Path to save the output DataFrame
        n_workers: Number of worker processes (default: number of CPU cores)
    """
    # Read input files
    print(go_df_path)
    go_df = pd.read_csv(go_df_path, index_col=0, delimiter="\t")
    mouse_ct_df = pd.read_csv(mouse_ct_pc_path, index_col=0)
    
    # Prepare arguments for multiprocessing
    args = [(go_id, row["GoName"], mouse_ct_df) for go_id, row in go_df.iterrows()]
    
    # Set number of workers
    if n_workers is None:
        n_workers = cpu_count()
    
    # Process files in parallel
    with Pool(n_workers) as pool:
        results = pool.map(process_go_bias, args)
    
    # Create column names for PC1-PC5 correlations
    columns = ["GoID", "GoName"]
    for pc in range(1,6):
        columns.extend([
            f"Bias_R_PC{pc}",
            f"Bias_R_PC{pc}_p",
            f"Bias_R_PC{pc}_spearman",
            f"Bias_R_PC{pc}_spearman_p"
        ])
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results, columns=columns)
    result_df = result_df.sort_values("Bias_R_PC1", ascending=False)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    result_df.to_csv(output_path)
    print(f"Results saved to {output_path}")
    
    # Print summary
    print(f"Processed {len(results)} GO terms")
    print(f"Top 5 correlations:")
    print(result_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GO bias files in parallel")
    parser.add_argument("--go_df", required=True, help="Path to GO DataFrame file")
    parser.add_argument("--go_bias_dir", required=True, help="Directory containing GO bias files")
    parser.add_argument("--mouse_ct_pc", required=True, help="Path to mouse cell type PC scores file")
    parser.add_argument("--output", required=True, help="Path to save output DataFrame")
    parser.add_argument("--workers", type=int, default=20, help="Number of worker processes (default: number of CPU cores)")
    
    args = parser.parse_args()
    
    # Set global variables
    global GoBiasDIR
    GoBiasDIR = args.go_bias_dir
    
    main(
        go_df_path=args.go_df,
        go_bias_dir=args.go_bias_dir,
        mouse_ct_pc_path=args.mouse_ct_pc,
        output_path=args.output,
        n_workers=args.workers
    ) 