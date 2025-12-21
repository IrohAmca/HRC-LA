import time

import matplotlib.pyplot as plt
import pandas as pd
import torch


# --- Functions ---
def standard_attention(Q, K, V):
    d_k = Q.shape[-1]
    # O(N^2) Operation
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

def hrc_la_attention(Q_raw, K_raw, V_raw, M=256):
    # O(N) Operation
    d_k = Q_raw.shape[-1]
    scale = d_k ** 0.25
    Q, K = Q_raw / scale, K_raw / scale
    
    # Random Feature Map
    gen = torch.Generator(device=Q.device)
    gen.manual_seed(42)
    omega = torch.randn(M, d_k, device=Q.device, generator=gen)
    
    theta_q = Q @ omega.T
    theta_k = K @ omega.T
    
    def get_phi(u, theta):
        norm_sq = torch.sum(u**2, dim=-1, keepdim=True)
        scale = torch.exp(norm_sq/2) / (M**0.5)
        return scale * torch.complex(torch.cos(theta), torch.sin(theta))
    
    phi_q = get_phi(Q, theta_q)
    phi_k = get_phi(K, theta_k)
    
    V_c = V_raw.to(torch.complex64)
    kv = phi_k.mH @ V_c
    
    num = (phi_q @ kv).real
    denom = (phi_q @ phi_k.sum(0).unsqueeze(1)).real
    
    return num / (denom + 1e-6)

def measure_resources(func, args, device):
    # Warm-up
    for _ in range(3):
        try:
            func(*args)
        except RuntimeError:
            pass
            
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_time = time.time()
    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    if start_event: start_event.record()
    
    try:
        func(*args)
    except RuntimeError as e:
        if "out of memory" in str(e):
            return None, None
        raise e
        
    if end_event: 
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0
    else:
        elapsed = time.time() - start_time
        
    peak_mem = 0
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        
    return elapsed, peak_mem

def run_full_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    seq_lengths = [512, 1024, 2048, 4096, 8192] 
    d_model = 64
    M = 256 
    
    results = {'N': [], 'Std_Time': [], 'HRC_Time': [], 'Std_Mem': [], 'HRC_Mem': []}
    
    print(f"{'N':<6} | {'Std Time':<10} | {'HRC Time':<10} | {'Std Mem (MB)':<12} | {'HRC Mem (MB)':<12}")
    print("-" * 65)
    
    for N in seq_lengths:
        # Prepare Data
        Q = torch.randn(N, d_model, device=device)
        K = torch.randn(N, d_model, device=device)
        V = torch.randn(N, d_model, device=device)
        
        # Standard Measurement
        t_std, m_std = measure_resources(standard_attention, (Q, K, V), device)
        
        # HRC-LA Measurement
        t_hrc, m_hrc = measure_resources(hrc_la_attention, (Q, K, V, M), device)
        
        # Log Results
        t_std_str = f"{t_std:<10.4f}" if t_std else "OOM       "
        m_std_str = f"{m_std:<12.1f}" if m_std else "OOM         "
        print(f"{N:<6} | {t_std_str} | {t_hrc:<10.4f} | {m_std_str} | {m_hrc:<12.1f}")
        
        results['N'].append(N)
        results['Std_Time'].append(t_std)
        results['HRC_Time'].append(t_hrc)
        results['Std_Mem'].append(m_std)
        results['HRC_Mem'].append(m_hrc)

    return pd.DataFrame(results)

# Run
if __name__ == "__main__":
    df = run_full_benchmark()
    
    # Plots
    plt.figure(figsize=(12, 5))
    
    # Time Plot
    plt.subplot(1, 2, 1)
    plt.plot(df['N'], df['Std_Time'], 'r-o', label='Standard (O(N^2))')
    plt.plot(df['N'], df['HRC_Time'], 'g-o', label='HRC-LA (O(N))')
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Time (s)')
    plt.title('Computation Time')
    plt.legend()
    plt.grid(True)
    
    # Memory Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['N'], df['Std_Mem'], 'r-o', label='Standard Mem')
    plt.plot(df['N'], df['HRC_Mem'], 'g-o', label='HRC-LA Mem')
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Memory (MB)')
    plt.title('VRAM Consumption')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()