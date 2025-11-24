#!/usr/bin/env python3
import numpy as np
from numpy.linalg import solve
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse

# ============================================================
# Global parameters for phi_exact (tanh spherical)
# ============================================================

R0 = 0.4
WIDTH = 0.3  # interface thickness in analytic phi_exact


# ============================================================
# Command-line argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Ghost-Cell Reconstruction: Polynomial vs RBF"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rbf",
        choices=["poly", "rbf"],
        help="Reconstruction method: 'poly' = quadratic LSQ, 'rbf' = RBF-Gaussian (default: rbf)"
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=150,
        help="Max nearest neighbors for LSQ/RBF stencil (default: 150)"
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=1e-7,
        help="Tikhonov regularization factor (dimensionless, default: 1e-7)"
    )
    parser.add_argument(
        "--tau_factor",
        type=float,
        default=2.0,
        help="Tau factor for polynomial Gaussian weights: tau = tau_factor * dx (default: 2.0)"
    )
    parser.add_argument(
        "--rbf_eps_factor",
        type=float,
        default=1.0,
        help="RBF shape parameter: eps = rbf_eps_factor / WIDTH (default: 1.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of highest-error ghost cells to highlight in visualization"
    )
    return parser.parse_args()


# ============================================================
# Exact Manufactured Test Function (tanh spherical)
# ============================================================

def phi_exact(x, y, z):
    r = np.sqrt((x-0.5)**2 + y*y + (z-0.5)**2)
    return np.tanh((R0 - r) / WIDTH)


# ============================================================
# Gaussian Weight Kernel for polynomial LSQ
# ============================================================

def gaussian_weight(r, tau):
    return np.exp(-(r/tau)**2)


# ============================================================
# 3D quadratic (degree <= 2) polynomial basis
# Monomials X^a Y^b Z^c with a+b+c <= 2 → 10 terms
# ============================================================

EXP3 = []
for a in range(3):
    for b in range(3 - a):
        for c in range(3 - a - b):
            if a + b + c <= 2:
                EXP3.append((a, b, c))
EXP3 = np.array(EXP3, dtype=int)
M3D = EXP3.shape[0]  # should be 10


# ============================================================
# Build polynomial basis into preallocated matrix
# ============================================================

def build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M_scratch):
    """
    Fill M_scratch[0:K, :] with quadratic 3D polynomial basis values,
    where K = stencil_pts.shape[0].
    """
    K = stencil_pts.shape[0]
    a = EXP3[:, 0][None, :]
    b = EXP3[:, 1][None, :]
    c = EXP3[:, 2][None, :]

    X = (stencil_pts[:, 0] - x0) / dx
    Y = (stencil_pts[:, 1] - y0) / dy
    Z = (stencil_pts[:, 2] - z0) / dz

    M_scratch[:K, :] = (X[:, None]**a) * (Y[:, None]**b) * (Z[:, None]**c)


# ============================================================
# Polynomial LSQ reconstruction with QR + Tikhonov
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

    # Build basis
    build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M)

    # Distances and weights
    dX = stencil_pts[:, 0] - x0
    dY = stencil_pts[:, 1] - y0
    dZ = stencil_pts[:, 2] - z0
    r = np.sqrt(dX*dX + dY*dY + dZ*dZ)
    w[:] = gaussian_weight(r, tau)

    rhs[:] = stencil_vals

    # QR factorization
    Q, R = np.linalg.qr(M, mode="reduced")

    # Stable LSQ system
    WQ = w[:, None] * Q
    B = Q.T @ WQ          # (m,m)
    d = Q.T @ (w * rhs)   # (m,)

    # Regularization
    if reg_lambda > 0.0:
        diag_max = np.max(np.abs(np.diag(B)))
        lam = reg_lambda * (diag_max if diag_max > 0 else 1.0)
        B += lam * np.eye(m)

    c = solve(B, d)
    a = solve(R, c)

    return a[0]


# ============================================================
# RBF-Gaussian kernel and reconstruction
# ============================================================

def rbf_gaussian_matrix_into(stencil_pts, eps, A_scratch):
    """
    Build Gaussian RBF matrix A_ij = exp(-(eps*r_ij)^2) into A_scratch[:K,:K].
    """
    K = stencil_pts.shape[0]
    pts = stencil_pts.reshape(K, 1, 3)
    diff = pts - pts.transpose(1, 0, 2)      # (K,K,3)
    r2 = np.sum(diff*diff, axis=2)           # (K,K)
    A_scratch[:K, :K] = np.exp(- (eps**2) * r2)


def reconstruct_rbf(
    x0, y0, z0,
    stencil_pts, stencil_vals,
    eps, reg_lambda,
    A_scratch, b_scratch
):
    """
    RBF reconstruction using Gaussian kernel:
        φ(x) ≈ sum_j a_j exp(-(eps*|x - x_j|)^2)
    with Tikhonov regularization for stability.
    """
    K = stencil_pts.shape[0]

    A = A_scratch[:K, :K]
    b = b_scratch[:K]

    # Build RBF matrix
    rbf_gaussian_matrix_into(stencil_pts, eps, A)

    # Regularization
    if reg_lambda > 0.0:
        diag_max = np.max(np.abs(np.diag(A)))
        lam = reg_lambda * (diag_max if diag_max > 0 else 1.0)
        A[np.diag_indices(K)] += lam

    b[:] = stencil_vals

    # Solve for RBF coefficients
    a = solve(A, b)

    # Evaluate at ghost point
    x0v = np.array([x0, y0, z0])
    diff0 = stencil_pts - x0v[None, :]
    r2_0 = np.sum(diff0*diff0, axis=1)
    kvec = np.exp(- (eps**2) * r2_0)

    phi_g = np.dot(a, kvec)
    return phi_g


# ============================================================
# Visualization of ghost-cell error distribution
# ============================================================

def visualize_error_distribution(xs, ys, zs,
                                 ghost_indices,
                                 reconstructed, exact_vals,
                                 title="", top_k=50):
    ghost = np.array(ghost_indices, dtype=int)
    Xg = xs[ghost[:,0]]
    Yg = ys[ghost[:,1]]
    Zg = zs[ghost[:,2]]

    eps = 1e-12
    err = np.abs(reconstructed - exact_vals) / np.maximum(np.abs(exact_vals), eps)

    top_k = min(top_k, len(err))
    idx_top = np.argpartition(err, -top_k)[-top_k:]

    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(Xg, Yg, Zg, c=err, cmap="viridis", s=20)
    fig.colorbar(p, ax=ax, shrink=0.55, label="Relative Error")

    ax.scatter(Xg[idx_top], Yg[idx_top], Zg[idx_top],
               c='red', s=80, label="Top error cells")

    ax.set_title(f"Ghost Cell Error Distribution ({title})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ============================================================
# One resolution test
# ============================================================

def run_test(N, method, max_neighbors, reg_lambda,
             tau_factor, rbf_eps_factor, top_k):

    # Setup grid
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
    dirs = [(i,j,k) for i in [-1,0,1]
                    for j in [-1,0,1]
                    for k in [-1,0,1]
                    if not (i==0 and j==0 and k==0)]

    # Identify active/ghost cells
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

    active_coords = np.array([(xs[i], ys[j], zs[k]) for (i, j, k) in active], float)
    num_active = active_coords.shape[0]

    if num_active == 0:
        raise RuntimeError("No active points found.")

    # KD-tree for fast neighbor search
    tree = cKDTree(active_coords)

    # Preallocate LSQ/RBF arrays
    Kmax = min(max_neighbors, num_active)

    # For polynomial
    M_scratch  = np.empty((Kmax, M3D), float)
    w_scratch  = np.empty(Kmax, float)
    rhs_scratch= np.empty(Kmax, float)

    # For RBF
    A_scratch  = np.empty((Kmax, Kmax), float)
    b_scratch  = np.empty(Kmax, float)

    reconstructed = []
    exact_vals    = []

    tau = tau_factor * dx
    eps = rbf_eps_factor / WIDTH   # RBF shape parameter

    for (i, j, k) in ghost:
        x0, y0, z0 = xs[i], ys[j], zs[k]

        # KD-tree search
        Kquery = min(max_neighbors, num_active)
        _, idx = tree.query([x0, y0, z0], k=Kquery)
        idx = np.array(idx, ndmin=1, dtype=int)

        stencil_pts  = active_coords[idx]
        stencil_vals = phi_exact(stencil_pts[:,0], stencil_pts[:,1], stencil_pts[:,2])

        if method == "poly":
            phi_g = reconstruct_poly(
                x0, y0, z0,
                stencil_pts, stencil_vals,
                dx, dy, dz,
                tau, reg_lambda,
                M_scratch, w_scratch, rhs_scratch
            )
        elif method == "rbf":
            phi_g = reconstruct_rbf(
                x0, y0, z0,
                stencil_pts, stencil_vals,
                eps, reg_lambda,
                A_scratch, b_scratch
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact(x0, y0, z0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    eps_rel = 1e-12
    rel_err = np.abs(reconstructed - exact_vals) / np.maximum(np.abs(exact_vals), eps_rel)
    max_err = np.max(rel_err)

    print(f"N={N:4d}, method={method}, max rel error = {max_err:.6e}")

    # Visualize error distribution for this resolution
    visualize_error_distribution(
        xs, ys, zs,
        ghost,
        reconstructed,
        exact_vals,
        title=f"N={N}, method={method}",
        top_k=top_k
    )

    return max_err


# ============================================================
# Main driver
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    print("\n============================================")
    print("   3D Ghost-Cell Reconstruction (poly vs RBF)")
    print("============================================")
    print(f"Method:          {args.method}")
    print(f"max_neighbors:   {args.max_neighbors}")
    print(f"reg_lambda:      {args.reg_lambda}")
    print(f"tau_factor:      {args.tau_factor}")
    print(f"rbf_eps_factor:  {args.rbf_eps_factor}")
    print()

    N_list = [10, 20, 40, 80]
    errors = []

    for N in N_list:
        e = run_test(
            N=N,
            method=args.method,
            max_neighbors=args.max_neighbors,
            reg_lambda=args.reg_lambda,
            tau_factor=args.tau_factor,
            rbf_eps_factor=args.rbf_eps_factor,
            top_k=args.top_k
        )
        errors.append(e)

    # Plot convergence
    hs = [1.6 / N for N in N_list]
    plt.figure()
    plt.loglog(hs, errors, marker="o", label=f"3D {args.method.upper()} reconstruction")
    plt.gca().invert_xaxis()
    plt.xlabel("h = 1.6 / N")
    plt.ylabel("Max Relative Error on Ghost Cells")
    plt.title(f"Convergence ({args.method.upper()} method)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

