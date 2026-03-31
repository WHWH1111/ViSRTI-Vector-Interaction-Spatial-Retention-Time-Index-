#!/usr/bin/env python3
"""
Script to analyze the last 10 epochs from multiple MAEs.txt log files
"""

import sys
import os

def analyze_last_epochs(file_path, num_epochs=10):
    """
    Analyze the last N epochs from the MAEs.txt file
    
    Args:
        file_path (str): Path to the MAEs.txt file
        num_epochs (int): Number of last epochs to analyze
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
    
    # Read all lines from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Remove header line
    header = lines[0].strip().split('\t')
    data_lines = lines[1:]
    
    # Get the last num_epochs lines
    last_epochs = data_lines[-num_epochs:]
    
    print(f"\nAnalyzing file: {file_path}")
    print("=" * 80)
    
    # Print the last epochs data
    print(lines[0].strip())  # Print header
    for line in last_epochs:
        print(line.strip())
    
    print("\n" + "=" * 80)
    print(f"Average values for the last {num_epochs} epochs:")
    print("=" * 80)
    
    # Calculate averages for each column
    # Skip the first column (Epoch) as it's not a numeric value
    num_columns = len(header)
    
    # Initialize sums for each column
    sums = [0.0] * num_columns
    
    # Sum up values for each column
    for line in last_epochs:
        values = line.strip().split('\t')
        for i in range(1, num_columns):  # Skip first column (Epoch)
            try:
                sums[i] += float(values[i])
            except (ValueError, IndexError):
                pass  # Skip non-numeric values or missing columns
    
    # Calculate averages
    averages = [0.0] * num_columns
    for i in range(1, num_columns):
        averages[i] = sums[i] / len(last_epochs)
    
    # Print averages
    print('\t'.join(header))  # Print header
    avg_line = ['AVG']  # First column label
    for i in range(1, num_columns):
        avg_line.append(f"{averages[i]:.6f}")
    print('\t'.join(avg_line))
    
    return averages

def main():
    # Define file paths
    file_paths = [
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251105-002503_dim48_layerH6_layerO6_batch32_lr0.0001_iter150/MAEs.txt",
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251104-055350_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt",
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251104-055250_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt"
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251104-025502_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt"
        # neg
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251102-023702_dim48_layerH6_layerO6_batch64_lr0.0001_iter200/MAEs.txt",
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251102-023737_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt",
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251102-023807_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt",
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251110-200708_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt"
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251110-210035_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt",
        # "/home/data2/rhj/project/gnn/gnn-1/log/train_20251111-001553_dim48_layerH6_layerO6_batch64_lr0.0001_iter150/MAEs.txt"

        # Windows paths
        # "D:\\Projects\\python\\gnn-rt-1\\log\\train_20251111-201130_dim48_layerH6_layerO6_batch64_lr0.0001_iter150\\MAEs.txt",
        "D:\\Projects\\python\\gnn-rt-1\\log\\train_20251111-230530_dim48_layerH6_layerO6_batch64_lr0.0001_iter150\\MAEs.txt"
    ]
    
    # Allow custom file paths as command line arguments
    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
    
    results = []
    for file_path in file_paths:
        avg = analyze_last_epochs(file_path)
        if avg is not None:
            results.append((file_path, avg))
    
    # Print comparison if multiple files
    if len(results) > 1:
        print("\n\n" + "=" * 80)
        print("COMPARISON OF AVERAGE VALUES")
        print("=" * 80)
        
        # Print header
        headers = ["File", "MAE_train", "MAE_dev", "MAE_test"]
        print("{:<60} {:<12} {:<12} {:<12}".format(*headers))
        print("-" * 100)
        
        # Print comparison data
        for file_path, averages in results:
            # Extract key metrics (MAE_train, MAE_dev, MAE_test)
            # Based on the file format: Epoch, Time, Loss_train, MAE_train, MSE_train, R2_train, PCC_train, MAE_dev, MSE_dev, R2_dev, PCC_dev, MAE_test, MSE_test, R2_test, PCC_test
            mae_train = averages[3]   # MAE_train is at index 3
            mae_dev = averages[7]     # MAE_dev is at index 7
            mae_test = averages[11]   # MAE_test is at index 11
            
            # Shorten file path for display
            short_path = file_path.split('/')[-2] if '/' in file_path else file_path
            
            print("{:<60} {:<12.6f} {:<12.6f} {:<12.6f}".format(
                short_path, mae_train, mae_dev, mae_test))

if __name__ == "__main__":
    main()