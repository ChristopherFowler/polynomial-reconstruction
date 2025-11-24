#!/usr/bin/env python3
"""
Debug version of poly3d_5.py reconstruction to trace numerical steps
"""
import numpy as np
from numpy.linalg import solve

# ============================================================
# Problem parameters
# ============================================================

R0    = 0.4
WIDTH = 0.25

def phi_exact(x, y, z):
    r = np.sqrt((x - 0.53)**2 + y*y + (z - 0.48)**2)
    return np.exp(x-0.5+2*(y-0.3)+3*(z-0.6))

def gaussian_weight(r, tau):
    return np.exp(-(r/tau)**2)

# Basis exponents
EXP3 = []
for a in range(3):
    for b in range(3 - a):
        for c in range(3 - a - b):
            if a + b + c <= 2:
                EXP3.append((a, b, c))
EXP3 = np.array(EXP3, dtype=int)
M3D = EXP3.shape[0]

print(f"Basis functions: {M3D}")
print(f"Exponents:\n{EXP3}")

def build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M_scratch):
    K = stencil_pts.shape[0]
    a = EXP3[:, 0][None, :]
    b = EXP3[:, 1][None, :]
    c = EXP3[:, 2][None, :]

    X = (stencil_pts[:, 0] - x0) / dx
    Y = (stencil_pts[:, 1] - y0) / dy
    Z = (stencil_pts[:, 2] - z0) / dz

    M_scratch[:K, :] = (X[:, None]**a) * (Y[:, None]**b) * (Z[:, None]**c)

def reconstruct_poly(
    x0, y0, z0,
    stencil_pts, stencil_vals,
    dx, dy, dz,
    tau, reg_lambda,
    M_scratch, w_scratch, rhs_scratch, debug=False
):
    K = stencil_pts.shape[0]
    m = M3D
    q_size = min(K, m)  # Rank of the system

    M = M_scratch[:K, :]
    w = w_scratch[:K]
    rhs = rhs_scratch[:K]

    # Basis
    build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M)
    
    if debug:
        print(f"\n=== Debug Reconstruction at ({x0:.4f}, {y0:.4f}, {z0:.4f}) ===")
        print(f"K (stencil size): {K}")
        print(f"dx, dy, dz: {dx}, {dy}, {dz}")
        print(f"tau: {tau}")
        print(f"reg_lambda: {reg_lambda}")
        print(f"\nStencil points (first 3):")
        for i in range(min(3, K)):
            print(f"  [{i}]: ({stencil_pts[i, 0]:.4f}, {stencil_pts[i, 1]:.4f}, {stencil_pts[i, 2]:.4f})")
        print(f"\nStencil values (first 3): {stencil_vals[:min(3, K)]}")

    # Weights
    dX = stencil_pts[:, 0] - x0
    dY = stencil_pts[:, 1] - y0
    dZ = stencil_pts[:, 2] - z0
    r = np.sqrt(dX*dX + dY*dY + dZ*dZ)
    w[:] = gaussian_weight(r, tau)

    rhs[:] = stencil_vals
    
    if debug:
        print(f"\nDistances (first 3): {r[:min(3, K)]}")
        print(f"Weights (first 3): {w[:min(3, K)]}")
        print(f"\nBasis matrix M (first 3 rows, all columns):")
        for i in range(min(3, K)):
            print(f"  M[{i}]: {M[i, :]}")

    # QR
    Q, R = np.linalg.qr(M, mode="reduced")
    
    if debug:
        print(f"\nAfter QR decomposition:")
        print(f"Q shape: {Q.shape}")
        print(f"R shape: {R.shape}")
        print(f"R matrix:\n{R}")
        print(f"R diagonal: {np.diag(R)}")

    # Stable LSQ system
    WQ = w[:, None] * Q
    B = Q.T @ WQ
    d = Q.T @ (w * rhs)
    
    if debug:
        print(f"\nWeighted system:")
        print(f"B = Q^T @ diag(w) @ Q:")
        print(f"{B}")
        print(f"B diagonal: {np.diag(B)}")
        print(f"d = Q^T @ (w .* rhs): {d}")

    # Regularization
    if reg_lambda > 0.0:
        diag_max = np.max(np.abs(np.diag(B)))
        lam = reg_lambda * (diag_max if diag_max > 0 else 1.0)
        B += lam * np.eye(q_size)  # Use q_size not m
        if debug:
            print(f"\nAfter regularization (lambda={lam:.3e}):")
            print(f"B diagonal: {np.diag(B)}")

    c = solve(B, d)
    
    # Solve R * a = c
    # R is (q_size x m), c is (q_size,), need to find a of length m
    # Since R is from QR of M (K x m) with K < m, only first q_size equations are valid
    # We can only solve for the first q_size coefficients directly
    # For now, pad with zeros for the remaining coefficients
    a = np.zeros(m)
    if q_size == m:
        # Square case: solve normally
        a = solve(R, c)
    else:
        # Rectangular case: R is (K x m) with K < m
        # Extract the upper triangular part that we can solve
        # R * a = c means we have K equations for m unknowns
        # The system is underdetermined; we solve for a minimum-norm solution
        # Using least squares: a = R^T (R R^T)^{-1} c
        R_trunc = R[:q_size, :q_size]  # Use only the first K columns for a K x K system
        a_trunc = solve(R_trunc, c)
        a[:q_size] = a_trunc
    
    if debug:
        print(f"\nSolved coefficients:")
        print(f"c (intermediate): {c}")
        print(f"a (final polynomial coefficients): {a}")
        print(f"a[0] (constant term / reconstructed value): {a[0]}")

    return a[0]


# ============================================================
# Test with minimal example
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing reconstruction with minimal example")
    print("="*60)
    
    # Active points - need at least M3D=10 points for full rank
    active_coords_list = []
    for i in range(15):  # Create 15 points
        x = i * 0.05
        y = (i % 3) * 0.05
        z = (i % 5) * 0.04
        active_coords_list.append([x, y, z])
    active_coords = np.array(active_coords_list, dtype=float)
    
    active_vals = phi_exact(active_coords[:, 0], 
                           active_coords[:, 1], 
                           active_coords[:, 2])
    
    print(f"\nActive coordinates:\n{active_coords}")
    print(f"Active values: {active_vals}")
    
    # Ghost point
    N = 10
    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)
    zs = np.linspace(-0.8, 0.8, N)
    dx = 1.6 / N
    
    ghost_i, ghost_j, ghost_k = 5, 5, 5
    x0 = xs[ghost_i]
    y0 = ys[ghost_j]
    z0 = zs[ghost_k]
    
    print(f"\nGhost point indices: ({ghost_i}, {ghost_j}, {ghost_k})")
    print(f"Ghost point position: ({x0:.6f}, {y0:.6f}, {z0:.6f})")
    print(f"dx: {dx}")
    
    # Reconstruct
    K = len(active_coords)  # Use all points
    tau = 2.0 * dx
    reg_lambda = 1e-7
    
    M_scratch = np.empty((K, M3D), float)
    w_scratch = np.empty(K, float)
    rhs_scratch = np.empty(K, float)
    
    reconstructed = reconstruct_poly(
        x0, y0, z0,
        active_coords, active_vals,
        dx, dx, dx,
        tau, reg_lambda,
        M_scratch, w_scratch, rhs_scratch,
        debug=True
    )
    
    exact_val = phi_exact(x0, y0, z0)
    rel_err = abs(reconstructed - exact_val) / max(abs(exact_val), 1e-3)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Reconstructed value: {reconstructed}")
    print(f"Exact value:         {exact_val}")
    print(f"Absolute error:      {abs(reconstructed - exact_val)}")
    print(f"Relative error:      {rel_err}")
    print(f"{'='*60}")
