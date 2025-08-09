#!/usr/bin/env python3
"""
Neural Network Multi-Digit Addition Error Analysis Script

This script analyzes errors in a neural network model performing multi-digit addition
by testing various "digit shift" hypotheses to explain incorrect answers.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


def parse_data_file(filepath):
    """
    Parse input text file and return DataFrame with columns: a, b, model_result
    
    Args:
        filepath: Path to the input file
        
    Returns:
        pandas.DataFrame with columns 'a', 'b', 'model_result' (all strings)
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line and '+' in line:
                # Split by '=' to separate inputs from result
                left_side, model_result = line.split('=')
                
                # Split left side by '+' to get a and b
                a, b = left_side.split('+')
                
                data.append({
                    'a': a,
                    'b': b, 
                    'model_result': model_result
                })
    
    return pd.DataFrame(data)


def analyze_digit_shifts(df):
    """
    Perform digit shift analysis on the parsed data.
    
    Args:
        df: DataFrame with columns 'a', 'b', 'model_result'
        
    Returns:
        pandas.DataFrame with conditional accuracy values for each hypothesis
    """
    # Generate all combinations of shifts (0, 1, 2) for a and b
    shifts = [0, 1, 2]
    shift_combinations = list(itertools.product(shifts, shifts))
    
    # Define hypotheses
    hypotheses = []
    for shift_a, shift_b in shift_combinations:
        hypotheses.append(f"a_shift_{shift_a}_b_shift_{shift_b}_res")
        hypotheses.append(f"a_shift_{shift_a}_b_shift_{shift_b}_using_correct_carry")
    
    # Initialize results dictionary
    results = {hyp: [0] * 13 for hyp in hypotheses}  # 13 digit positions (0-12)
    total_problems = len(df)
    
    # Process each problem
    for _, row in df.iterrows():
        a_orig = row['a']
        b_orig = row['b'] 
        model_result = row['model_result']
        
        # Pad all strings to standard length (12 for operands, 13 for result)
        a_padded = a_orig.ljust(12, '0')
        b_padded = b_orig.ljust(12, '0')
        result_padded = model_result.ljust(13, '0')
        
        # Calculate correct carries for each position
        correct_carries = [0]  # carry into position 0 is always 0
        for pos in range(12):
            digit_a = int(a_padded[pos]) if pos < len(a_padded) else 0
            digit_b = int(b_padded[pos]) if pos < len(b_padded) else 0
            digit_sum = digit_a + digit_b + correct_carries[pos]
            correct_carries.append(digit_sum // 10)  # carry to next position
        
        # For each hypothesis, calculate cascaded carries across all positions
        for shift_a, shift_b in shift_combinations:
            # Calculate cascaded carries for this shift combination
            cascaded_carries = [0]  # carry into position 0 is always 0
            
            for pos in range(12):
                # Get shifted digits for current position
                shifted_pos_a = pos - shift_a
                shifted_pos_b = pos - shift_b
                
                digit_a = int(a_padded[shifted_pos_a]) if 0 <= shifted_pos_a < len(a_padded) else 0
                digit_b = int(b_padded[shifted_pos_b]) if 0 <= shifted_pos_b < len(b_padded) else 0
                
                digit_sum = digit_a + digit_b + cascaded_carries[pos]
                cascaded_carries.append(digit_sum // 10)  # carry to next position
            
            # Test each digit position for this hypothesis
            for pos in range(13):
                model_digit = int(result_padded[pos]) if pos < len(result_padded) else 0
                
                # Get shifted digits for current position
                shifted_pos_a = pos - shift_a
                shifted_pos_b = pos - shift_b
                
                digit_a = int(a_padded[shifted_pos_a]) if 0 <= shifted_pos_a < len(a_padded) else 0
                digit_b = int(b_padded[shifted_pos_b]) if 0 <= shifted_pos_b < len(b_padded) else 0
                
                # Test cascaded error hypothesis (_res)
                hyp_name_res = f"a_shift_{shift_a}_b_shift_{shift_b}_res"
                carry_res = cascaded_carries[pos] if pos < len(cascaded_carries) else 0
                simulated_digit_res = (digit_a + digit_b + carry_res) % 10
                
                if simulated_digit_res == model_digit:
                    results[hyp_name_res][pos] += 1
                
                # Test isolated error hypothesis (_using_correct_carry)
                hyp_name_correct = f"a_shift_{shift_a}_b_shift_{shift_b}_using_correct_carry"
                carry_correct = correct_carries[pos] if pos < len(correct_carries) else 0
                simulated_digit_correct = (digit_a + digit_b + carry_correct) % 10
                
                if simulated_digit_correct == model_digit:
                    results[hyp_name_correct][pos] += 1
    
    # Convert counts to conditional accuracies
    accuracy_df = pd.DataFrame(results)
    accuracy_df = accuracy_df / total_problems
    
    return accuracy_df


def create_heatmap(accuracy_df, output_filename):
    """
    Create and save heatmap visualization of the accuracy results.
    
    Args:
        accuracy_df: DataFrame with accuracy values for each hypothesis
        output_filename: Filename to save the heatmap
    """
    # Transpose so hypotheses are rows and digit positions are columns
    plot_data = accuracy_df.T
    
    # Sort rows: _res hypotheses first, then _using_correct_carry, alphabetically within each group
    res_hypotheses = [col for col in plot_data.index if col.endswith('_res')]
    correct_carry_hypotheses = [col for col in plot_data.index if col.endswith('_using_correct_carry')]
    
    res_hypotheses.sort()
    correct_carry_hypotheses.sort()
    
    ordered_hypotheses = res_hypotheses + correct_carry_hypotheses
    plot_data = plot_data.reindex(ordered_hypotheses)
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(plot_data, 
                annot=False, 
                cmap='rocket', 
                vmin=0, 
                vmax=0.8,
                cbar_kws={'label': 'Conditional Accuracy'})
    
    plt.xlabel('Digit index (0 means unit place)')
    plt.ylabel('Hypothesis')
    plt.title('Digit Shift Analysis Results')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main execution function for command line usage.
    """
    # Process 12+12.txt
    print("Processing 12+12.txt...")
    df_12_12 = parse_data_file('12+12.txt')
    accuracy_12_12 = analyze_digit_shifts(df_12_12)
    create_heatmap(accuracy_12_12, 'heatmap_12+12.png')
    print("Saved heatmap_12+12.png")
    
    # Process 12+10.txt
    print("Processing 12+10.txt...")
    df_12_10 = parse_data_file('12+10.txt')
    accuracy_12_10 = analyze_digit_shifts(df_12_10)
    create_heatmap(accuracy_12_10, 'heatmap_12+10.png')
    print("Saved heatmap_12+10.png")


if __name__ == "__main__":
    main()