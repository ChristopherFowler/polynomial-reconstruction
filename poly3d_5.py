#!/usr/bin/env python3
import numpy as np
from numpy.linalg import solve
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse

# ============================================================
# Problem parameters: tanh spherical profile
# ============================================================

R0    = 0.4    # radius in phi_exact
WIDTH = 0.25   # interface thickness in physical units


# ============================================================
# Command-line argument parser
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="3D Ghost-Cell Reconstruction with 2-pass polynomial LSQ"
    )
    p.add_argument("--max_neighbors",
                   type=int, default=120,
                   help="Stencil size in first pass (default: 120)")
    p.add_argument("--max_neighbors_bad",
                   type=int, default=200,
                   help="Stencil size for 'bad' cells in second pass (default: 200)")
    p.add_argument("--tau_factor",
                   type=float, default=2.0,
                   help="Tau factor (tau = tau_factor*dx) for first pass (default: 2.0)")
    p.add_argument("--tau_factor_bad",
                   type=float, default=4.0,
                   help="Tau factor for 'bad' cells in second pass (default: 4.0)")
    p.add_argument("--reg_lambda",
                   type=float, default=1e-7,
                   help="Tikhonov regularization factor (default: 1e-7)")
    p.add_argument("--relerr_threshold",
                   type=float, default=0.2,
                   help="Relative error threshold to trigger 2nd pass (default: 0.2)")
    p.add_argument("--phi_rel_clip",
                   type=float, default=1e-3,
                   help="Minimum |phi_exact| used in relative error denominator (default: 1e-3)")
    p.add_argument("--top_k",
                   type=int, default=50,
                   help="Number of top-error ghost cells to highlight (default: 50)")
    return p.parse_args()


# ============================================================
# Exact manufactured function (tanh spherical)
# ============================================================

#def phi_exact(x, y, z):
#    r = np.sqrt((x - 0.53)**2 + y*y + (z - 0.48)**2)
#    return np.tanh((R0 - r) / WIDTH)

#def phi_exact(x, y, z):
#    r = np.sqrt((x - 0.53)**2 + y*y + (z - 0.48)**2)
#    return np.exp(r/WIDTH)


def phi_exact(x, y, z):
    r = np.sqrt((x - 0.53)**2 + y*y + (z - 0.48)**2)
    return np.exp(x-0.5+2*(y-0.3)+3*(z-0.6))

# ============================================================
# Gaussian kernel for polynomial LSQ weights
# ============================================================

def gaussian_weight(r, tau):
    return np.exp(-(r/tau)**2)


# ============================================================
# Quadratic 3D polynomial basis exponents (degree <= 2)
# ============================================================

EXP3 = []
for a in range(3):
    for b in range(3 - a):
        for c in range(3 - a - b):
            if a + b + c <= 2:
                EXP3.append((a, b, c))
EXP3 = np.array(EXP3, dtype=int)
M3D = EXP3.shape[0]   # should be 10


# ============================================================
# Build quadratic basis into scratch matrix
# ============================================================

def build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M_scratch):
    K = stencil_pts.shape[0]
    a = EXP3[:, 0][None, :]
    b = EXP3[:, 1][None, :]
    c = EXP3[:, 2][None, :]

    X = (stencil_pts[:, 0] - x0) / dx
    Y = (stencil_pts[:, 1] - y0) / dy
    Z = (stencil_pts[:, 2] - z0) / dz

    M_scratch[:K, :] = (X[:, None]**a) * (Y[:, None]**b) * (Z[:, None]**c)


# ============================================================
# Polynomial LSQ reconstruction (QR + Tikhonov)
# ============================================================

def reconstruct_poly(
    x0, y0, z0,
    stencil_pts, stencil_vals,
    dx, dy, dz,
    tau, reg_lambda,
    M_scratch, w_scratch, rhs_scratch
):
    K = stencil_pts.shape[0]
    m = M3D

    M = M_scratch[:K, :]
    w = w_scratch[:K]
    rhs = rhs_scratch[:K]

    # Basis
    build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M)

    # Weights
    dX = stencil_pts[:, 0] - x0
    dY = stencil_pts[:, 1] - y0
    dZ = stencil_pts[:, 2] - z0
    r = np.sqrt(dX*dX + dY*dY + dZ*dZ)
    w[:] = gaussian_weight(r, tau)

    rhs[:] = stencil_vals

    # QR
    Q, R = np.linalg.qr(M, mode="reduced")

    # Stable LSQ system
    WQ = w[:, None] * Q
    B = Q.T @ WQ
    d = Q.T @ (w * rhs)

    # Regularization
    if reg_lambda > 0.0:
        diag_max = np.max(np.abs(np.diag(B)))
        lam = reg_lambda * (diag_max if diag_max > 0 else 1.0)
        B += lam * np.eye(m)

    c = solve(B, d)
    a = solve(R, c)

    return a[0]


# ============================================================
# Visualization: 3D scatter of ghost-cell error
# ============================================================

def visualize_error_distribution(xs, ys, zs,
                                 ghost_indices,
                                 rel_err,
                                 title="", top_k=50):
    ghost = np.array(ghost_indices, dtype=int)
    Xg = xs[ghost[:,0]]
    Yg = ys[ghost[:,1]]
    Zg = zs[ghost[:,2]]

    top_k = min(top_k, len(rel_err))
    idx_top = np.argpartition(rel_err, -top_k)[-top_k:]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(Xg, Yg, Zg, c=rel_err, cmap="viridis", s=8)
    fig.colorbar(p, ax=ax, shrink=0.55, label="Relative Error")

    ax.scatter(Xg[idx_top], Yg[idx_top], Zg[idx_top],
               c='red', s=60, label="Top error cells")

    ax.set_title(f"Ghost Cell Error Distribution ({title})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ============================================================
# One resolution test with 2-pass reconstruction
# ============================================================

def run_test(N,
             max_neighbors, max_neighbors_bad,
             tau_factor, tau_factor_bad,
             reg_lambda, relerr_threshold,
             phi_rel_clip, top_k):

    # Grid and spacing
    L = 1.6
    dx = L / N
    dy = dx
    dz = dx

    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)
    zs = np.linspace(-0.8, 0.8, N)

    R = 0.8

    active = []
    ghost  = []

    # 26 neighbor directions
    dirs = [(i, j, k)
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            for k in [-1, 0, 1]
            if not (i == 0 and j == 0 and k == 0)]

    # Classify active vs ghost
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                r = np.sqrt(x*x + y*y + z*z)
                if r < R:
                    active.append((i, j, k))
                else:
                    for di, dj, dk in dirs:
                        ii = i + di
                        jj = j + dj
                        kk = k + dk
                        if 0 <= ii < N and 0 <= jj < N and 0 <= kk < N:
                            xr, yr, zr = xs[ii], ys[jj], zs[kk]
                            if np.sqrt(xr*xr + yr*yr + zr*zr) < R:
                                ghost.append((i, j, k))
                                break

    active_coords = np.array([(xs[i], ys[j], zs[k]) for (i, j, k) in active],
                             dtype=float)
    num_active = active_coords.shape[0]
    if num_active == 0:
        raise RuntimeError("No active points found")

    ghost = np.array(ghost, dtype=int)
    num_ghost = ghost.shape[0]

    # KD-tree for neighbor search
    tree = cKDTree(active_coords)

    # Preallocate LSQ arrays (use largest stencil size)
    Kmax = min(max_neighbors_bad, num_active)
    M_scratch  = np.empty((Kmax, M3D), float)
    w_scratch  = np.empty(Kmax, float)
    rhs_scratch= np.empty(Kmax, float)

    # ------------------------------------------------------------------
    # First pass reconstruction
    # ------------------------------------------------------------------
    tau1 = tau_factor * dx

    reconstructed1 = np.empty(num_ghost, float)
    exact_vals     = np.empty(num_ghost, float)

    for idx_g, (i, j, k) in enumerate(ghost):
        x0, y0, z0 = xs[i], ys[j], zs[k]

        K1 = min(max_neighbors, num_active)
        _, idx = tree.query([x0, y0, z0], k=K1)
        idx = np.array(idx, ndmin=1, dtype=int)

        stencil_pts  = active_coords[idx]
        stencil_vals = phi_exact(stencil_pts[:, 0],
                                 stencil_pts[:, 1],
                                 stencil_pts[:, 2])

        reconstructed1[idx_g] = reconstruct_poly(
            x0, y0, z0,
            stencil_pts, stencil_vals,
            dx, dy, dz,
            tau1, reg_lambda,
            M_scratch, w_scratch, rhs_scratch
        )
        exact_vals[idx_g] = phi_exact(x0, y0, z0)

    # Error stats for first pass
    denom1 = np.maximum(np.abs(exact_vals), phi_rel_clip)
    rel_err1 = np.abs(reconstructed1 - exact_vals) / denom1
    abs_err1 = np.abs(reconstructed1 - exact_vals)

    print(f"  First pass: mean={rel_err1.mean():.3e}, "
          f"median={np.median(rel_err1):.3e}, "
          f"p95={np.percentile(rel_err1,95):.3e}, "
          f"max_rel={rel_err1.max():.3e}, "
          f"max_abs={abs_err1.max():.3e}")

    # Identify "bad" ghost cells needing second pass
    bad_mask = rel_err1 > relerr_threshold
    bad_indices = np.nonzero(bad_mask)[0]
    num_bad = bad_indices.size
    print(f"  Bad cells above threshold ({relerr_threshold}): {num_bad} / {num_ghost}")

    # ------------------------------------------------------------------
    # Second pass: refine only bad cells with larger stencil & tau
    # ------------------------------------------------------------------
    reconstructed2 = reconstructed1.copy()
    tau2 = tau_factor_bad * dx

    if num_bad > 0:
        for idx_local in bad_indices:
            i, j, k = ghost[idx_local]
            x0, y0, z0 = xs[i], ys[j], zs[k]

            K2 = min(max_neighbors_bad, num_active)
            _, idx = tree.query([x0, y0, z0], k=K2)
            idx = np.array(idx, ndmin=1, dtype=int)

            stencil_pts  = active_coords[idx]
            stencil_vals = phi_exact(stencil_pts[:, 0],
                                     stencil_pts[:, 1],
                                     stencil_pts[:, 2])

            reconstructed2[idx_local] = reconstruct_poly(
                x0, y0, z0,
                stencil_pts, stencil_vals,
                dx, dy, dz,
                tau2, reg_lambda,
                M_scratch, w_scratch, rhs_scratch
            )

    # Final error stats after second pass
    denom2 = np.maximum(np.abs(exact_vals), phi_rel_clip)
    rel_err2 = np.abs(reconstructed2 - exact_vals) / denom2
    abs_err2 = np.abs(reconstructed2 - exact_vals)

    print(f"  Second pass: mean={rel_err2.mean():.3e}, "
          f"median={np.median(rel_err2):.3e}, "
          f"p95={np.percentile(rel_err2,95):.3e}, "
          f"max_rel={rel_err2.max():.3e}, "
          f"max_abs={abs_err2.max():.3e}")

    # Visualize final error distribution
    visualize_error_distribution(
        xs, ys, zs,
        ghost,
        rel_err2,
        title=f"N={N}",
        top_k=top_k
    )

    # Return max relative error (final)
    return rel_err2.max()


# ============================================================
# Main driver
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    print("\n============================================")
    print("   3D Ghost-Cell Polynomial Reconstruction")
    print("      Two-Pass LSQ with Diagnostics")
    print("============================================")
    print(f"max_neighbors (pass1):     {args.max_neighbors}")
    print(f"max_neighbors_bad (pass2): {args.max_neighbors_bad}")
    print(f"tau_factor (pass1):        {args.tau_factor}")
    print(f"tau_factor_bad (pass2):    {args.tau_factor_bad}")
    print(f"reg_lambda:                {args.reg_lambda}")
    print(f"relerr_threshold:          {args.relerr_threshold}")
    print(f"phi_rel_clip:              {args.phi_rel_clip}")
    print()

    N_list = [10, 20, 40, 80]
    errors = []

    for N in N_list:
        print(f"\n--- Resolution N={N} ---")
        e = run_test(
            N=N,
            max_neighbors=args.max_neighbors,
            max_neighbors_bad=args.max_neighbors_bad,
            tau_factor=args.tau_factor,
            tau_factor_bad=args.tau_factor_bad,
            reg_lambda=args.reg_lambda,
            relerr_threshold=args.relerr_threshold,
            phi_rel_clip=args.phi_rel_clip,
            top_k=args.top_k
        )
        errors.append(e)
        print(f"  Final max relative error (N={N}) = {e:.6e}")

    # Convergence plot (using final max relative error)
    hs = [1.6 / N for N in N_list]
    plt.figure()
    plt.loglog(hs, errors, marker="o", label="Max relative error (2-pass poly)")
    plt.gca().invert_xaxis()
    plt.xlabel("h = 1.6 / N")
    plt.ylabel("Max relative error on ghost cells")
    plt.title("3D Ghost-Cell Polynomial Reconstruction (Two-Pass)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

