#!/usr/bin/env python3
"""
Simple validation script to debug CUDA implementation
"""
import numpy as np
import sys

# Test if module loads
try:
    import poly_cuda_wrapper
    print("✓ CUDA module imported successfully")
except Exception as e:
    print(f"✗ Failed to import CUDA module: {e}")
    sys.exit(1)

# Create minimal test case
print("\nRunning minimal test case...")
print("-" * 50)

# Helper function matching CUDA phi_exact
def phi_exact(x, y, z):
    r = np.sqrt((x - 0.53)**2 + y**2 + (z - 0.48)**2)
    return np.exp(x - 0.5 + 2*(y - 0.3) + 3*(z - 0.6))

# Create enough active points (need >= 10 for M3D=10)
active_coords_list = []
for i in range(15):  # Create 15 points
    x = i * 0.05
    y = (i % 3) * 0.05
    z = (i % 5) * 0.04
    active_coords_list.append([x, y, z])
active_coords = np.array(active_coords_list, dtype=np.float32)

# Values matching the actual function
active_vals = phi_exact(active_coords[:, 0], active_coords[:, 1], active_coords[:, 2]).astype(np.float32)

# One ghost point
ghost_indices = np.array([[5, 5, 5]], dtype=np.int32)

# Grid
N = 10
xs = np.linspace(-0.8, 0.8, N, dtype=np.float32)
ys = np.linspace(-0.8, 0.8, N, dtype=np.float32)
zs = np.linspace(-0.8, 0.8, N, dtype=np.float32)
dx = 1.6 / N

print(f"Active coords shape: {active_coords.shape}")
print(f"Active vals: {active_vals}")
print(f"Ghost indices: {ghost_indices}")
print(f"Ghost position: ({xs[5]}, {ys[5]}, {zs[5]})")
print(f"dx: {dx}")

try:
    recon, exact, err, nbad = poly_cuda_wrapper.reconstruct_ghost_cells_cuda(
        active_coords, 
        active_vals, 
        ghost_indices,
        xs, ys, zs,
        np.float32(dx),
        np.float32(2.0 * dx),  # tau1
        np.float32(4.0 * dx),  # tau2
        np.float32(1e-7),       # reg_lambda
        15,                      # max_neighbors (use all 15)
        15,                      # max_neighbors_bad
        np.float32(0.2),        # relerr_threshold
        np.float32(1e-3)        # phi_rel_clip
    )
    
    print(f"\n✓ CUDA function executed")
    print(f"Reconstructed value: {recon[0]}")
    print(f"Exact value: {exact[0]}")
    print(f"Relative error: {err[0]}")
    print(f"Num bad cells: {nbad}")
    
    # Check if result is reasonable
    if err[0] < 0.5:  # Allow 50% error for this simple test
        print("\n✓ Result looks reasonable (rel error < 0.5)")
    else:
        print(f"\n✗ Warning: Relative error too large: {err[0]}")
        print("   This suggests a bug in the CUDA implementation")
        
except Exception as e:
    print(f"\n✗ CUDA function failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Basic test completed")
print("=" * 50)
