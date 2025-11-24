import numpy as np
from numpy.linalg import solve
import argparse
import matplotlib.pyplot as plt


# ============================================================
#  Command-line argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D ROD Reconstruction Test with Convergence Plot"
    )
    parser.add_argument(
        "--use_dirichlet",
        type=int,
        default=0,
        help="(kept for compatibility, but unused here) 1 = Dirichlet ON, 0 = OFF (default)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="full3d",
        choices=["full3d"],
        help="full3d = 3D quadratic polynomial;"
    )
    return parser.parse_args()


# ============================================================
#  Exact Manufactured Test Function (3D)
# ============================================================

def phi_exact(x, y, z):
    s = x + 2.0*y + 3.0*z
    return np.exp(s) / (1.0 + np.exp(s))


# ============================================================
#  Gaussian Weight Kernel
# ============================================================

def gaussian_weight(r, tau):
    return np.exp(-(r/tau)**2)




# ============================================================
#  3D quadratic (degree <= 2) polynomial basis
# ============================================================

EXPONENTS_3D_QUAD = []
for a in range(0, 3):
    for b in range(0, 3 - a):
        for c in range(0, 3 - a - b):
            if a + b + c <= 3:
                EXPONENTS_3D_QUAD.append((a, b, c))
M3D = len(EXPONENTS_3D_QUAD)  # should be 10


def poly_basis_quad_3d(x, y, z, x0, y0, z0, dx, dy, dz):
    X = (x - x0)/dx
    Y = (y - y0)/dy
    Z = (z - z0)/dz

    vals = []
    for a, b, c in EXPONENTS_3D_QUAD:
        vals.append((X**a) * (Y**b) * (Z**c))
    return np.array(vals)


# ============================================================
#  Reconstruction method 1:
#  Full 3D quadratic LSQ
# ============================================================
def reconstruct_full3d_quadratic(x0, y0, z0,
                                 stencil_pts,
                                 stencil_vals,
                                 dx, dy, dz,
                                 tau,
                                 reg_lambda=1e-7):
    """
    Full 3D quadratic LSQ reconstruction with:
    - Gaussian weights
    - QR orthogonalization of the design matrix (BIG conditioning improvement)
    - Optional Tikhonov regularization on the stable system
    """

    l = len(stencil_pts)
    m = M3D  # number of quadratic monomials

    # ---------------------------------------------
    # Build weighted design
    # ---------------------------------------------
    M = np.zeros((l, m))
    w = np.zeros(l)
    rhs = np.zeros(l)

    for i, (x, y, z) in enumerate(stencil_pts):
        M[i, :] = poly_basis_quad_3d(x, y, z, x0, y0, z0, dx, dy, dz)
        r = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
        w[i] = gaussian_weight(r, tau)
        rhs[i] = stencil_vals[i]

    # ---------------------------------------------
    # QR decomposition of UNWEIGHTED basis matrix M
    # ---------------------------------------------
    # M = Q R   →  Q orthonormal, R upper-triangular
    Q, R = np.linalg.qr(M, mode='reduced')  # shapes: (l,m), (m,m)

    # ---------------------------------------------
    # Build the stabilized LSQ system
    # (Q^T W Q) c = Q^T W rhs
    # Then a = R^{-1} c
    # ---------------------------------------------
    WQ = (w[:, None] * Q)       # elementwise multiply each row by w[i]

    B = Q.T @ WQ                # stable version of M^T W M
    d = Q.T @ (w * rhs)         # stable version of M^T W rhs

    # ---------------------------------------------
    # Optional regularization on the stabilized system
    # ---------------------------------------------
    if reg_lambda > 0.0:
        diag_max = np.max(np.abs(np.diag(B)))
        lam = reg_lambda * (diag_max if diag_max > 0 else 1.0)
        B = B + lam * np.eye(m)

    # ---------------------------------------------
    # Solve for c, then recover the polynomial coefficients
    # ---------------------------------------------
    c = solve(B, d)
    a = solve(R, c)  # R * a = c

    # ---------------------------------------------
    # Return reconstructed ghost value = p(x0,y0,z0) = constant coefficient
    # ---------------------------------------------
    return a[0]




# ============================================================
#  Main Test: reconstruction on spherical domain
# ============================================================

def run_test(N=20, tau_factor=1.0, method="full3d"):
    L = 1.6
    dx = L / N
    dy = dx
    dz = dx

    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)
    zs = np.linspace(-0.8, 0.8, N)

    R = 0.8

    active = []
    ghost = []

    neighbors = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
        ( 1, 1, 0), ( 1,-1, 0), (-1, 1, 0), (-1,-1, 0),
        ( 1, 0, 1), ( 1, 0,-1), (-1, 0, 1), (-1, 0,-1),
        ( 0, 1, 1), ( 0, 1,-1), ( 0,-1, 1), ( 0,-1,-1),
        ( 1, 1, 1), ( 1, 1,-1), ( 1,-1, 1), ( 1,-1,-1),
        (-1, 1, 1), (-1, 1,-1), (-1,-1, 1), (-1,-1,-1)
    ]

    # classify active vs ghost
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                r = np.sqrt(x*x + y*y + z*z)
                if r < R:
                    active.append((i, j, k))
                else:
                    for di, dj, dk in neighbors:
                        ii = i + di
                        jj = j + dj
                        kk = k + dk
                        if 0 <= ii < N and 0 <= jj < N and 0 <= kk < N:
                            xr = xs[ii]
                            yr = ys[jj]
                            zr = zs[kk]
                            if np.sqrt(xr*xr + yr*yr + zr*zr) < R:
                                ghost.append((i, j, k))
                                break

    active_coords = np.array([(xs[i], ys[j], zs[k]) for (i, j, k) in active])

    reconstructed = []
    exact_vals    = []

    tau = tau_factor * dx

    for (i, j, k) in ghost:
        x0, y0, z0 = xs[i], ys[j], zs[k]

        dists = np.sqrt((active_coords[:,0] - x0)**2 +
                        (active_coords[:,1] - y0)**2 +
                        (active_coords[:,2] - z0)**2)
        idx   = np.argsort(dists)  # all neighbors
        stencil_pts  = active_coords[idx]
        stencil_vals = phi_exact(stencil_pts[:,0],
                                 stencil_pts[:,1],
                                 stencil_pts[:,2])

        if method == "full3d":
            phi_g = reconstruct_full3d_quadratic(
                x0, y0, z0,
                stencil_pts,
                stencil_vals,
                dx, dy, dz,
                tau
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact(x0, y0, z0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    eps = 1e-12
    rel_err = np.abs(reconstructed - exact_vals) / np.maximum(np.abs(exact_vals), eps)
    max_err = np.max(rel_err)

    print(f"N={N:4d}, method={method}, max rel error = {max_err:.6e}")
    return max_err


# ============================================================
#  Convergence plot
# ============================================================

def plot_convergence(N_list, errors, method, L=1.6):
    hs = [L / N for N in N_list]

    plt.figure()
    plt.loglog(hs, errors, marker="o", linestyle="-",
               label=f"3D ROD, {method}")
    plt.gca().invert_xaxis()
    plt.xlabel("h = L / N")
    plt.ylabel("max relative error on ghost cells")
    plt.title("3D ROD Convergence (spherical domain)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#  Main driver
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    method = args.method

    print("\n============================================")
    print("   3D ROD Reconstruction (full3d)")
    print("============================================")
    print(f"Method: {method}\n")

    N_list = [10, 20, 40, 80]
    errors = []

    for N in N_list:
        err = run_test(N, tau_factor=2.0, method=method)
        errors.append(err)

    print("\nSummary:")
    for N, e in zip(N_list, errors):
        print(f"  N={N:4d}, max rel error = {e:.6e}")

    plot_convergence(N_list, errors, method)

