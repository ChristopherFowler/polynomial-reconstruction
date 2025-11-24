#!/usr/bin/env python3
import numpy as np
from numpy.linalg import solve
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse


def rbf_kernel(r, eps):
    return np.exp(-(eps * r)**2)

def reconstruct_rbf(x0, y0, z0, stencil_pts, stencil_vals, eps, lam=1e-9):
    N = len(stencil_pts)
    A = np.zeros((N, N))

    # Build RBF matrix
    for i in range(N):
        xi = stencil_pts[i]
        for j in range(N):
            xj = stencil_pts[j]
            A[i,j] = rbf_kernel(np.linalg.norm(xi - xj), eps)

    # Tikhonov regularization
    A += lam * np.eye(N)

    # Solve for coefficients
    a = np.linalg.solve(A, stencil_vals)

    # Evaluate at ghost point
    x0v = np.array([x0, y0, z0])
    phi_g = 0.0
    for j in range(N):
        phi_g += a[j] * rbf_kernel(
            np.linalg.norm(x0v - stencil_pts[j]), eps
        )
    return phi_g



# ============================================================
# Command-line argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D ROD Reconstruction Test with Error Visualization"
    )
    parser.add_argument("--max_neighbors", type=int, default=200,
        help="Max nearest neighbors for LSQ stencil (default: 200)")
    parser.add_argument("--reg_lambda", type=float, default=1e-7,
        help="Tikhonov regularization strength (default: 1e-7)")
    parser.add_argument("--top_k", type=int, default=50,
        help="Number of highest-error ghost cells to highlight")
    return parser.parse_args()

# ============================================================
# Manufactured solution
# ============================================================

#def phi_exact(x, y, z):
#    R0=0.5
#    width=0.2
#    r = np.sqrt((x-0.5)*(x-0.5) + y*y + (z-0.5)*(z-0.5))
#    return np.tanh((R0 - r) / width)

def phi_exact(x, y, z):
    s = x + 2.0*y + 3.0*z
    return np.exp(s) / (1.0 + np.exp(s))


# ============================================================
# 3D Isosurface Visualization of phi_exact
# ============================================================

def visualize_phi_isosurface(xs, ys, zs, phi_func, iso=0.0, title="Exact Phi Isosurface"):
    """
    Build an NxNxN cube of exact phi values and plot an isosurface.
    Uses Plotly for interactive 3D visualization.
    """
    import plotly.graph_objects as go

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    PHI = phi_func(X, Y, Z)   # vectorized exact phi

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=PHI.flatten(),
        isomin=iso,
        isomax=iso,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale='Viridis',
        showscale=True
    ))

    fig.update_layout(
        title=title,
        width=800,
        height=700,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='data'
        )
    )

    fig.show()




# ============================================================
# Gaussian weight
# ============================================================

def gaussian_weight(r, tau):
    return np.exp(-(r/tau)**2)

# ============================================================
# Quadratic basis (degree ≤ 2) exponents
# ============================================================

EXP3 = []
for a in range(3):
    for b in range(3 - a):
        for c in range(3 - a - b):
            if a + b + c <= 2:
                EXP3.append((a, b, c))

EXP3 = np.array(EXP3, dtype=int)
M3D = EXP3.shape[0]   # = 10

# ============================================================
# Build basis into preallocated matrix
# ============================================================

def build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M_scratch):
    K = stencil_pts.shape[0]
    a = EXP3[:,0][None,:]
    b = EXP3[:,1][None,:]
    c = EXP3[:,2][None,:]

    X = (stencil_pts[:,0] - x0)/dx
    Y = (stencil_pts[:,1] - y0)/dy
    Z = (stencil_pts[:,2] - z0)/dz

    M_scratch[:K,:] = (X[:,None]**a) * (Y[:,None]**b) * (Z[:,None]**c)

# ============================================================
# One LSQ reconstruction
# ============================================================

def reconstruct_full3d(
    x0, y0, z0,
    stencil_pts, stencil_vals,
    dx, dy, dz,
    tau, reg_lambda,
    M_scratch, w_scratch, rhs_scratch
):
    K = stencil_pts.shape[0]
    M = M_scratch[:K,:]
    w = w_scratch[:K]
    rhs = rhs_scratch[:K]

    # Build basis
    build_basis_quad_3d_into(stencil_pts, x0, y0, z0, dx, dy, dz, M)

    # Compute Gaussian weights
    dxv = stencil_pts[:,0] - x0
    dyv = stencil_pts[:,1] - y0
    dzv = stencil_pts[:,2] - z0
    r = np.sqrt(dxv*dxv + dyv*dyv + dzv*dzv)
    w[:] = gaussian_weight(r, tau)
    rhs[:] = stencil_vals

    # QR
    Q, R = np.linalg.qr(M, mode="reduced")

    # Build stable LSQ system
    WQ = w[:,None] * Q
    B = Q.T @ WQ
    d = Q.T @ (w * rhs)

    # Regularization
    if reg_lambda > 0:
        diag_max = np.max(np.abs(np.diag(B)))
        lam = reg_lambda * (diag_max if diag_max > 0 else 1.0)
        B += lam * np.eye(M3D)

    # Solve system
    c = solve(B, d)
    a = solve(R, c)   # polynomial coefficients

    # Return p(x0)
    return a[0]

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

    # Find worst performers
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

def run_test(N, max_neighbors, reg_lambda, top_k):

    # Setup grid
    L = 1.6
    dx = L / N
    xs = np.linspace(-0.8,0.8,N)
    ys = np.linspace(-0.8,0.8,N)
    zs = np.linspace(-0.8,0.8,N)

    R = 0.8


    #if N == 20:
    #    visualize_phi_isosurface(xs, ys, zs, phi_exact, iso=0.0,
    #                             title=f"Exact Phi Isosurface (N={N})")


    active = []
    ghost  = []

    # 26 neighbor directions
    dirs = [(i,j,k) for i in [-1,0,1]
                    for j in [-1,0,1]
                    for k in [-1,0,1]
                    if not (i==0 and j==0 and k==0)]

    # Identify active/ghost cells
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            for k,z in enumerate(zs):
                r = np.sqrt(x*x + y*y + z*z)
                if r < R:
                    active.append((i,j,k))
                else:
                    for di,dj,dk in dirs:
                        ii = i+di; jj=j+dj; kk=k+dk
                        if 0 <= ii < N and 0 <= jj < N and 0 <= kk < N:
                            xr,yr,zr = xs[ii], ys[jj], zs[kk]
                            if np.sqrt(xr*xr + yr*yr + zr*zr) < R:
                                ghost.append((i,j,k))
                                break

    active_coords = np.array([(xs[i], ys[j], zs[k]) for (i,j,k) in active], float)
    num_active = active_coords.shape[0]

    # KD-tree for fast neighbor search
    tree = cKDTree(active_coords)

    # Preallocate LSQ arrays
    Kmax = min(max_neighbors, num_active)
    M_scratch  = np.empty((Kmax, M3D), float)
    w_scratch  = np.empty(Kmax, float)
    rhs_scratch= np.empty(Kmax, float)

    reconstructed = []
    exact_vals    = []

    tau = dx * 1.0

    # Solve for each ghost cell
    for (i,j,k) in ghost:
        x0, y0, z0 = xs[i], ys[j], zs[k]

        # KD-tree search
        Kquery = min(max_neighbors, num_active)
        _, idx = tree.query([x0,y0,z0], k=Kquery)
        idx = np.array(idx, ndmin=1, dtype=int)

        stencil_pts  = active_coords[idx]
        stencil_vals = phi_exact(stencil_pts[:,0], stencil_pts[:,1], stencil_pts[:,2])

#        phi_g = reconstruct_rbf(x0, y0, z0, stencil_pts, stencil_vals,
 #                       eps=50/dx, lam=1e-8)

        phi_g = reconstruct_full3d(
            x0, y0, z0,
            stencil_pts, stencil_vals,
            dx, dx, dx,
            tau, reg_lambda,
            M_scratch, w_scratch, rhs_scratch
        )

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact(x0, y0, z0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    eps = 1e-12
    rel_err = np.abs(reconstructed - exact_vals) / np.maximum(np.abs(exact_vals), eps)
    max_err = np.max(rel_err)

    print(f"N={N}, max rel error = {max_err:.6e}")

    # Visualize distribution of error
    visualize_error_distribution(
        xs, ys, zs,
        ghost,
        reconstructed,
        exact_vals,
        title=f"N={N}",
        top_k=top_k
    )

    return max_err

# ============================================================
# Main driver
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    N_list = [10, 20, 40, 80]
    errors = []




    for N in N_list:
        e = run_test(
            N=N,
            max_neighbors=args.max_neighbors,
            reg_lambda=args.reg_lambda,
            top_k=args.top_k
        )
        errors.append(e)

    # Plot convergence
    hs = [1.6/N for N in N_list]
    plt.figure()
    plt.loglog(hs, errors, marker="o", label="3D ROD LSQ")
    plt.gca().invert_xaxis()
    plt.xlabel("h = 1.6/N")
    plt.ylabel("Max Relative Error")
    plt.title("ROD3D Convergence")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

