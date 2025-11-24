#!/usr/bin/env python3
"""
Profile poly3d_6.py to identify performance bottlenecks
"""
import numpy as np
import time
import poly_cuda_wrapper

def phi_exact(x, y, z):
    r = np.sqrt((x - 0.53)**2 + y*y + (z - 0.48)**2)
    return np.exp(x-0.5+2*(y-0.3)+3*(z-0.6))

def profile_run(N):
    print(f"\n{'='*60}")
    print(f"Profiling N={N}")
    print(f"{'='*60}")
    
    t_start = time.time()
    
    # Grid setup
    L = 1.6
    dx = L / N
    xs = np.linspace(-0.8, 0.8, N, dtype=np.float32)
    ys = np.linspace(-0.8, 0.8, N, dtype=np.float32)
    zs = np.linspace(-0.8, 0.8, N, dtype=np.float32)
    R = 0.8
    
    t_grid = time.time()
    print(f"Grid setup: {(t_grid - t_start)*1000:.2f} ms")
    
    # Classify active vs ghost (vectorized for speed)
    # Create meshgrid of all cell centers
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    R_all = np.sqrt(X*X + Y*Y + Z*Z)
    
    # Active cells: inside sphere
    active_mask = R_all < R
    
    # Ghost cells: outside sphere but adjacent to active
    # Dilate the active mask by 1 cell in all directions
    from scipy.ndimage import binary_dilation
    dilated_mask = binary_dilation(active_mask, iterations=1)
    ghost_mask = dilated_mask & ~active_mask
    
    # Get indices
    active_indices = np.argwhere(active_mask)
    ghost_indices = np.argwhere(ghost_mask)
    
    # Convert to coordinates (ensure C-contiguous)
    active_coords = np.stack([
        xs[active_indices[:, 0]],
        ys[active_indices[:, 1]],
        zs[active_indices[:, 2]]
    ], axis=1).astype(np.float32)
    active_coords = np.ascontiguousarray(active_coords)
    
    ghost = ghost_indices.astype(np.int32)
    ghost = np.ascontiguousarray(ghost)
    num_active = active_coords.shape[0]
    num_ghost = ghost.shape[0]
    
    t_classify = time.time()
    print(f"Cell classification: {(t_classify - t_grid)*1000:.2f} ms")
    
    print(f"  Active cells: {num_active}")
    print(f"  Ghost cells: {num_ghost}")
    
    # Compute active values
    active_vals = phi_exact(active_coords[:, 0],
                           active_coords[:, 1],
                           active_coords[:, 2]).astype(np.float32)
    
    t_vals = time.time()
    print(f"Computing active values: {(t_vals - t_classify)*1000:.2f} ms")
    
    # CUDA reconstruction
    tau1 = np.float32(2.0 * dx)
    tau2 = np.float32(4.0 * dx)
    
    reconstructed, exact_vals, rel_err, num_bad = poly_cuda_wrapper.reconstruct_ghost_cells_cuda(
        active_coords,
        active_vals,
        ghost,
        xs, ys, zs,
        np.float32(dx),
        tau1,
        tau2,
        np.float32(1e-7),
        120,  # max_neighbors
        200,  # max_neighbors_bad
        np.float32(0.2),
        np.float32(1e-3)
    )
    
    t_cuda = time.time()
    print(f"CUDA reconstruction: {(t_cuda - t_vals)*1000:.2f} ms")
    
    # Statistics
    abs_err = np.abs(reconstructed - exact_vals)
    t_stats = time.time()
    print(f"Statistics: {(t_stats - t_cuda)*1000:.2f} ms")
    
    print(f"\nTotal time: {(t_stats - t_start)*1000:.2f} ms")
    print(f"  Mean rel error: {rel_err.mean():.3e}")
    print(f"  Max rel error: {rel_err.max():.3e}")
    print(f"  Bad cells: {num_bad}")
    
    return {
        'N': N,
        'total_time': t_stats - t_start,
        'classify_time': t_classify - t_grid,
        'cuda_time': t_cuda - t_vals,
        'num_active': num_active,
        'num_ghost': num_ghost
    }

if __name__ == "__main__":
    results = []
    for N in [10, 20, 40, 80, 160]:
        result = profile_run(N)
        results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'N':>5} {'Active':>8} {'Ghost':>8} {'Classify':>10} {'CUDA':>10} {'Total':>10}")
    print(f"{'':>5} {'cells':>8} {'cells':>8} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['N']:>5} {r['num_active']:>8} {r['num_ghost']:>8} "
              f"{r['classify_time']*1000:>10.2f} {r['cuda_time']*1000:>10.2f} "
              f"{r['total_time']*1000:>10.2f}")
    
    print(f"\nBottleneck Analysis:")
    for r in results:
        classify_pct = 100 * r['classify_time'] / r['total_time']
        cuda_pct = 100 * r['cuda_time'] / r['total_time']
        print(f"  N={r['N']:2}: Classification {classify_pct:.1f}%, CUDA {cuda_pct:.1f}%")
