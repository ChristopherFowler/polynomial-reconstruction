import numpy as np
from numpy.linalg import solve
import argparse
import matplotlib.pyplot as plt


# ============================================================
#  Command-line argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="3D ROD4 Quartic Reconstruction Test with Convergence Plot")
    parser.add_argument(
        "--use_dirichlet",
        type=int,
        default=1,
        help="1 = enforce Dirichlet BC (default), 0 = disable boundary constraints"
    )
    return parser.parse_args()


# ============================================================
#  Exact Manufactured Test Function (3D)
#  (Same functional form as 2D, just ignoring z)
# ============================================================

def phi_exact(x, y, z):
    """Exact 3D solution used for testing (depends only on x,y)."""
    s = x + 2.0*y + 3*z
    return np.exp(s)


# ============================================================
#  Gaussian Weight Kernel
# ============================================================

def gaussian_weight(r, tau):
    """Gaussian weight as a function of distance r and scale tau."""
    return np.exp(-(r/tau)**2)


# ============================================================
#  3D quartic (degree 4) polynomial basis
#  Monomials X^a Y^b Z^c with a+b+c <= 4  → 35 terms
# ============================================================

# Precompute exponent list once, shared by basis and gradient
EXPONENTS_3D = []
for a in range(0, 5):
    for b in range(0, 5 - a):
        for c in range(0, 5 - a - b):
            if a + b + c <= 2:  # was 4
                EXPONENTS_3D.append((a, b, c))
M_3D = len(EXPONENTS_3D)   # should be 35


def poly_basis4(x, y, z, x0, y0, z0, dx, dy, dz):
    """
    Quartic polynomial basis in normalized 3D coordinates:

    X = (x - x0)/dx
    Y = (y - y0)/dy
    Z = (z - z0)/dz

    Monomials X^a Y^b Z^c with a+b+c <= 4.
    """
    X = (x - x0) / dx
    Y = (y - y0) / dy
    Z = (z - z0) / dz

    vals = []
    for a, b, c in EXPONENTS_3D:
        vals.append((X**a) * (Y**b) * (Z**c))
    return np.array(vals)


# ============================================================
#  Gradient of quartic polynomial basis (3D)
# ============================================================

def poly_basis4_grad(x, y, z, x0, y0, z0, dx, dy, dz):
    """
    Gradient of the 3D quartic polynomial basis with respect to x, y, z.
    Uses chain rule with X=(x-x0)/dx, Y=(y-y0)/dy, Z=(z-z0)/dz.
    """
    X = (x - x0) / dx
    Y = (y - y0) / dy
    Z = (z - z0) / dz

    dXdx = 1.0 / dx
    dYdy = 1.0 / dy
    dZdz = 1.0 / dz

    dpx = []
    dpy = []
    dpz = []

    for a, b, c in EXPONENTS_3D:
        # d/dx: a * X^{a-1} Y^b Z^c * dXdx
        if a > 0:
            term_dx = a * (X**(a-1)) * (Y**b) * (Z**c) * dXdx
        else:
            term_dx = 0.0

        # d/dy: b * X^a Y^{b-1} Z^c * dYdy
        if b > 0:
            term_dy = b * (X**a) * (Y**(b-1)) * (Z**c) * dYdy
        else:
            term_dy = 0.0

        # d/dz: c * X^a Y^b Z^{c-1} * dZdz
        if c > 0:
            term_dz = c * (X**a) * (Y**b) * (Z**(c-1)) * dZdz
        else:
            term_dz = 0.0

        dpx.append(term_dx)
        dpy.append(term_dy)
        dpz.append(term_dz)

    return np.array(dpx), np.array(dpy), np.array(dpz)


# ============================================================
#  ROD4 Reconstruction (Dirichlet ON/OFF) in 3D
# ============================================================

def reconstruct_ROD4(x0, y0, z0,
                     stencil_pts,
                     stencil_vals,
                     collar_pts,
                     normals,
                     gvals,
                     dx, dy, dz,
                     tau,
                     use_dirichlet=True):
    """
    Perform quartic ROD reconstruction at a ghost cell (x0,y0,z0) in 3D.

    If use_dirichlet = False:
        Solve pure weighted LSQ: (Mᵀ W M) a = Mᵀ W φ.

    If use_dirichlet = True:
        Solve the KKT system:
            [A  Bᵀ][a] = [Mᵀ W φ]
            [B   0][λ]   [   b   ]
    """

    m = M_3D   # number of quartic coefficients (35)
    l = len(stencil_pts)

    # Build M, W, rhs
    M = np.zeros((l, m))
    W = np.zeros((l, l))
    rhs = np.zeros(l)

    for i, (x, y, z) in enumerate(stencil_pts):
        M[i, :] = poly_basis4(x, y, z, x0, y0, z0, dx, dy, dz)
        r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        W[i, i] = gaussian_weight(r, tau)
        rhs[i] = stencil_vals[i]

    MW = M.T @ W
    A = MW @ M   # normal matrix

    # --------------------------------------------------------
    # CASE 1: Dirichlet OFF → pure LSQ, no constraints
    # --------------------------------------------------------
    if not use_dirichlet:
        b_ls = MW @ rhs
        a = solve(A, b_ls)
        return a[0]

    # --------------------------------------------------------
    # CASE 2: Dirichlet ON → KKT system with boundary constraints
    # --------------------------------------------------------
    K = len(collar_pts)
    B = np.zeros((K, m))
    b = np.zeros(K)

    for r_idx in range(K):
        xr, yr, zr = collar_pts[r_idx]
        nx, ny, nz = normals[r_idx]

        alpha = 1.0
        beta  = 0.0  # only Dirichlet in this code (like 2D)

        basis = poly_basis4(xr, yr, zr, x0, y0, z0, dx, dy, dz)
        dpx, dpy, dpz = poly_basis4_grad(xr, yr, zr, x0, y0, z0, dx, dy, dz)

        # Same pattern as 2D: alpha*basis + beta*(grad·n)
        B[r_idx, :] = alpha * basis + beta * (dpx*nx + dpy*ny + dpz*nz)
        b[r_idx] = gvals[r_idx]

    Z = np.zeros((K, K))
    LHS = np.vstack((np.hstack((A, B.T)),
                     np.hstack((B, Z))))
    RHS = np.hstack((MW @ rhs, b))

    sol = solve(LHS, RHS)
    a = sol[:m]
    return a[0]


# ============================================================
#  Main Test: quartic ROD on spherical domain (3D)
# ============================================================

def run_test(N=20, tau_factor=1.0, use_dirichlet=True):
    """
    Run a single resolution test (3D):
      - Build spherical active/ghost sets
      - Reconstruct at ghost cells
      - Return max relative error
    """

    # Grid spacing
    L = 1.6
    dx = L / N
    dy = dx
    dz = dx

    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)
    zs = np.linspace(-0.8, 0.8, N)

    R = 0.8  # radius of spherical domain

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



    # Classify each grid point
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                r = np.sqrt(x*x + y*y + z*z)
                if r < R:
                    active.append((i, j, k))
                else:
                    # If any 6-neighbor is active, mark as ghost
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

    # Collar (boundary) points on sphere
    K_theta = 20
    K_phi   = 40
    theta = np.linspace(0.0, np.pi,   K_theta, endpoint=True)
    phi   = np.linspace(0.0, 2*np.pi, K_phi,   endpoint=False)

    collar_pts = []
    normals    = []
    gvals      = []

    for th in theta:
        for ph in phi:
            x = R * np.sin(th) * np.cos(ph)
            y = R * np.sin(th) * np.sin(ph)
            z = R * np.cos(th)
            collar_pts.append((x, y, z))
            nx, ny, nz = x/R, y/R, z/R
            normals.append((nx, ny, nz))
            gvals.append(phi_exact(x, y, z))

    collar_pts = np.array(collar_pts)
    normals    = np.array(normals)
    gvals      = np.array(gvals)

    reconstructed = []
    exact_vals    = []

    tau = tau_factor * dx

    # Precompute active coordinates for distance
    active_coords = np.array([(xs[i], ys[j], zs[k]) for (i,j,k) in active])

    # Loop over ghost cells and reconstruct
    for (i, j, k) in ghost:
        x0, y0, z0 = xs[i], ys[j], zs[k]

        # Interior stencil: nearest 100 active points (analogous to 45 in 2D)
        dists = np.sqrt((active_coords[:,0] - x0)**2 +
                        (active_coords[:,1] - y0)**2 +
                        (active_coords[:,2] - z0)**2)
        idx   = np.argsort(dists)[:100]

        stencil_pts  = active_coords[idx]
        stencil_vals = phi_exact(stencil_pts[:,0],
                                 stencil_pts[:,1],
                                 stencil_pts[:,2])

        # Boundary constraints: nearest 8 collar points (analogous to 4 in 2D)
        dcol = np.sqrt((collar_pts[:,0] - x0)**2 +
                       (collar_pts[:,1] - y0)**2 +
                       (collar_pts[:,2] - z0)**2)
        kk   = np.argsort(dcol)[:8]

        cp = collar_pts[kk]
        cn = normals[kk]
        gv = gvals[kk]

        phi_g = reconstruct_ROD4(
            x0, y0, z0,
            stencil_pts,
            stencil_vals,
            cp, cn, gv,
            dx, dy, dz,
            tau,
            use_dirichlet=use_dirichlet
        )

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact(x0, y0, z0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    # Relative max error (same pattern as 2D)
    eps = 1e-12
    rel_err = np.abs(reconstructed - exact_vals) / np.maximum(np.abs(exact_vals), eps)
    max_err = np.max(rel_err)

    print(f"N={N:4d}, Dirichlet={'ON' if use_dirichlet else 'OFF'}, max rel error = {max_err:.6e}")

    return max_err


# ============================================================
#  Convergence plot: error vs h = L/N (log-log)
# ============================================================

def plot_convergence(N_list, errors, use_dirichlet, L=1.6):
    """
    Plot error vs grid spacing h = L/N on a log-log scale.
    """
    hs = [L / N for N in N_list]

    plt.figure()
    plt.loglog(hs, errors, marker="o", linestyle="-",
               label=f"3D ROD4, Dirichlet {'ON' if use_dirichlet else 'OFF'}")
    plt.gca().invert_xaxis()
    plt.xlabel("h = L / N")
    plt.ylabel("max relative error on ghost cells")
    plt.title("3D ROD4 Convergence (spherical domain)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#  Main driver
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    use_dirichlet = bool(args.use_dirichlet)

    print("\n============================================")
    print("   3D ROD4 Quartic Polynomial Reconstruction")
    print("============================================")
    print(f"Dirichlet boundary condition: {'ON' if use_dirichlet else 'OFF'}\n")

    # Resolutions to test (3D gets expensive quickly)
    N_list = [10, 20, 40, 80]
    errors = []

    for N in N_list:
        err = run_test(N, tau_factor=2.0, use_dirichlet=use_dirichlet)
        errors.append(err)

    print("\nSummary:")
    for N, e in zip(N_list, errors):
        print(f"  N={N:4d}, max rel error = {e:.6e}")

    # Plot convergence curve
    plot_convergence(N_list, errors, use_dirichlet, L=1.6)

