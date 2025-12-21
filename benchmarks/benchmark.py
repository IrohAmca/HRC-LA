import os
import sys

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hrc_la import HRCMultiheadAttention
from hrc_la.utils import StandardAdapter, measure_performance, sync_weights


def run_ultimate_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    D_MODEL = 64
    NUM_HEADS = 4
    M_FEATURES = 256
    SEQ_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384] 
    
    # Initialize models
    std_model_raw = nn.MultiheadAttention(D_MODEL, NUM_HEADS, batch_first=True).to(device)
    hrc_model = HRCMultiheadAttention(D_MODEL, NUM_HEADS, M_FEATURES, batch_first=True).to(device)
    
    # Synchronize weights for fair comparison
    sync_weights(std_model_raw, hrc_model)
    std_model_raw.eval()
    hrc_model.eval()
    
    # Wrap standard model
    std_model = StandardAdapter(std_model_raw)
    
    results = {'N': [], 'Std_Time': [], 'HRC_Time': [], 'MSE_Error': []}
    
    print(f"\n{'N':<6} | {'Std Time (s)':<12} | {'HRC Time (s)':<12} | {'MSE Error':<10}")
    print("-" * 50)
    
    for N in SEQ_LENGTHS:
        x = torch.randn(1, N, D_MODEL).to(device)
        
        t_std, _, out_std = measure_performance(std_model, x, device)
        t_hrc, _, out_hrc = measure_performance(hrc_model, x, device)
        
        mse = 0.0
        if out_std is not None and out_hrc is not None:
            mse = torch.mean((out_std - out_hrc)**2).item()
        
        results['N'].append(N)
        results['Std_Time'].append(t_std)
        results['HRC_Time'].append(t_hrc)
        results['MSE_Error'].append(mse)
        
        t_s_str = f"{t_std:.4f}" if t_std else "Fail"
        t_hrc_str = f"{t_hrc:.4f}" if t_hrc else "Fail"
        mse_str = f"{mse:.6f}" if mse else "-"
        
        print(f"{N:<6} | {t_s_str:<12} | {t_hrc_str:<12} | {mse_str}")

    # Plotting results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['N'], results['Std_Time'], 'r-o', label='Standard O(N^2)')
    plt.plot(results['N'], results['HRC_Time'], 'g-o', label='HRC-LA O(N)')
    plt.title("Speed Benchmark")
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['N'], results['MSE_Error'], 'b-o', label='Approximation Error')
    plt.title("MSE Error")
    plt.xlabel("Sequence Length (N)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results.png')
    print("\nBenchmark plot saved as 'results.png'")
    plt.show()

if __name__ == "__main__":
    run_ultimate_benchmark()
