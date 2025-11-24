import numpy as np
from numpy.linalg import solve
import argparse
import matplotlib.pyplot as plt


# ============================================================
#  Command-line argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ROD4 Quartic Reconstruction Test with Convergence Plot")
    parser.add_argument(
        "--use_dirichlet",
        type=int,
        default=1,
        help="1 = enforce Dirichlet BC (default), 0 = disable boundary constraints"
    )
    return parser.parse_args()


# ============================================================
#  Exact Manufactured Test Function
# ============================================================

def phi_exact(x, y):
    """Exact solution used for testing."""
    return np.exp(x + 2*y)/(1+np.exp(x+2*y))


# ============================================================
#  Gaussian Weight Kernel
# ============================================================

def gaussian_weight(r, tau):
    """Gaussian weight as a function of distance r and scale tau."""
    return np.exp(-(r/tau)**2)


# ============================================================
#  Quartic (degree 4) polynomial basis: 15 terms
# ============================================================

def poly_basis4(x, y, x0, y0, dx, dy):
    """
    Quartic polynomial basis in normalized coordinates:

    X = (x - x0)/dx
    Y = (y - y0)/dy

    Monomials with total degree <= 4 (15 terms).
    """
    X = (x - x0)/dx
    Y = (y - y0)/dy

    return np.array([
        1.0,           # 0: 1
        X,             # 1: X
        Y,             # 2: Y
        X*X,           # 3: X^2
        X*Y,           # 4: X Y
        Y*Y,           # 5: Y^2
        X**3,          # 6: X^3
        X*X*Y,         # 7: X^2 Y
        X*Y*Y,         # 8: X Y^2
        Y**3,          # 9: Y^3
        X**4,          # 10: X^4
        X**3 * Y,      # 11: X^3 Y
        X*X * Y*Y,     # 12: X^2 Y^2
        X * Y**3,      # 13: X Y^3
        Y**4           # 14: Y^4
    ])


# ============================================================
#  Gradient of quartic polynomial basis
# ============================================================

def poly_basis4_grad(x, y, x0, y0, dx, dy):
    """
    Gradient of the quartic polynomial basis with respect to x and y.
    Uses chain rule with X=(x-x0)/dx, Y=(y-y0)/dy.
    """
    X = (x - x0)/dx
    Y = (y - y0)/dy

    dXdx = 1.0/dx
    dYdy = 1.0/dy

    # ∂/∂x of each basis term
    dpx = np.array([
        0.0,                        # 1
        dXdx,                       # X
        0.0,                        # Y
        2*X*dXdx,                   # X^2
        Y*dXdx,                     # X Y
        0.0,                        # Y^2
        3*X*X*dXdx,                 # X^3
        2*X*Y*dXdx,                 # X^2 Y
        Y*Y*dXdx,                   # X Y^2
        0.0,                        # Y^3
        4*X**3*dXdx,                # X^4
        3*X**2*Y*dXdx,              # X^3 Y
        2*X*Y*Y*dXdx,               # X^2 Y^2
        Y**3*dXdx,                  # X Y^3
        0.0                         # Y^4
    ])

    # ∂/∂y of each basis term
    dpy = np.array([
        0.0,
        0.0,
        dYdy,
        0.0,
        X*dYdy,
        2*Y*dYdy,
        0.0,
        X*X*dYdy,
        2*X*Y*dYdy,
        3*Y*Y*dYdy,
        0.0,
        X**3*dYdy,
        2*X*X*Y*dYdy,
        3*X*Y*Y*dYdy,
        4*Y**3*dYdy
    ])

    return dpx, dpy


# ============================================================
#  ROD4 Reconstruction (Dirichlet ON/OFF)
# ============================================================

def reconstruct_ROD4(x0, y0,
                     stencil_pts,
                     stencil_vals,
                     collar_pts,
                     normals,
                     gvals,
                     dx, dy,
                     tau,
                     use_dirichlet=True):
    """
    Perform quartic ROD reconstruction at a ghost cell (x0,y0).

    If use_dirichlet = False:
        Solve pure weighted LSQ: (Mᵀ W M) a = Mᵀ W φ.

    If use_dirichlet = True:
        Solve the KKT system:
            [A  Bᵀ][a] = [Mᵀ W φ]
            [B   0][λ]   [   b   ]
    """

    m = 15     # number of quartic coefficients
    l = len(stencil_pts)

    # Build M, W, rhs
    M = np.zeros((l, m))
    W = np.zeros((l, l))
    rhs = np.zeros(l)

    for i, (x, y) in enumerate(stencil_pts):
        M[i, :] = poly_basis4(x, y, x0, y0, dx, dy)
        W[i, i] = gaussian_weight(np.sqrt((x - x0)**2 + (y - y0)**2), tau)
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

    for r in range(K):
        xr, yr = collar_pts[r]
        nx, ny = normals[r]

        alpha = 1.0
        beta  = 0.0  # only Dirichlet in this code

        basis = poly_basis4(xr, yr, x0, y0, dx, dy)
        dpx, dpy = poly_basis4_grad(xr, yr, x0, y0, dx, dy)

        B[r, :] = alpha * basis + beta * (dpx*nx + dpy*ny)
        b[r] = gvals[r]

    Z = np.zeros((K, K))
    LHS = np.vstack((np.hstack((A, B.T)),
                     np.hstack((B, Z))))
    RHS = np.hstack((MW @ rhs, b))

    sol = solve(LHS, RHS)
    a = sol[:m]
    return a[0]


# ============================================================
#  Main Test: quartic ROD on circular domain
# ============================================================


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
               label=f"ROD4, Dirichlet {'ON' if use_dirichlet else 'OFF'}")
    plt.gca().invert_xaxis()
    plt.xlabel("h = L / N")
    plt.ylabel("max error on ghost cells")
    plt.title("ROD4 Convergence (circular domain)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#  Main driver
# ============================================================

def run_test(N=40, tau_factor=1.0, use_dirichlet=True):
    """
    Run a single resolution test:
      - Build circular active/ghost sets
      - Visualize grid occupancy (active only, active+ghost)
      - Reconstruct at ghost cells
      - Return max error
    """

    # Grid spacing
    L = 1.6
    dx = L / N
    dy = dx

    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)

    R = 0.8  # radius of circular domain

    active = []
    ghost = []

    # Classify each grid point
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            r = np.sqrt(x*x + y*y)
            if r < R:
                active.append((i, j))
            else:
                # If any 4-neighbor is active → mark as ghost
                for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < N and 0 <= jj < N:
                        if np.sqrt(xs[ii]**2 + ys[jj]**2) < R:
                            ghost.append((i, j))
                            break

    # ----------------------------------------------------------------------
    # == NEW VISUALIZATION SECTION ========================================
    # ----------------------------------------------------------------------

    # Create occupancy grid for plotting
    occ_active = np.zeros((N, N))
    occ_ghost  = np.zeros((N, N))

    for i, j in active:
        occ_active[j, i] = 1.0

    for i, j in ghost:
        occ_ghost[j, i] = 1.0

    # ---- Plot 1: Active region only (fluid domain) ----
    plt.figure(figsize=(5,5))
    plt.imshow(occ_active, origin="lower", cmap="Blues")
    plt.title(f"Active Fluid Sites (N={N})")
    plt.xlabel("i index")
    plt.ylabel("j index")
    plt.colorbar(label="Active = 1")
    plt.tight_layout()
    #plt.show()

    # ---- Plot 2: Active + Ghost region ----
    combined = occ_active + 0.5 * occ_ghost   # lighter color for ghost

    plt.figure(figsize=(5,5))
    plt.imshow(combined, origin="lower", cmap="viridis")
    plt.title(f"Active + Ghost Sites (N={N})")
    plt.xlabel("i index")
    plt.ylabel("j index")
    plt.colorbar(label="Active=1, Ghost=0.5")
    plt.tight_layout()
    #plt.show()

    # ----------------------------------------------------------------------
    # == END VISUALIZATION SECTION =========================================
    # ----------------------------------------------------------------------

    # Collar (boundary) points
    Kc = 200
    theta = np.linspace(0, 2*np.pi, Kc, endpoint=False)
    collar_pts = np.column_stack((R*np.cos(theta), R*np.sin(theta)))
    normals    = collar_pts / R          
    gvals      = phi_exact(collar_pts[:,0], collar_pts[:,1])

    reconstructed = []
    exact_vals    = []

    tau = tau_factor * dx

    active_coords = np.array([(xs[i], ys[j]) for (i,j) in active])

    # Loop: reconstruct at ghost cells
    for (i, j) in ghost:
        x0, y0 = xs[i], ys[j]

        dists = np.sqrt((active_coords[:,0] - x0)**2 +
                        (active_coords[:,1] - y0)**2)
        idx   = np.argsort(dists)[:45]

        stencil_pts  = active_coords[idx]
        stencil_vals = phi_exact(stencil_pts[:,0], stencil_pts[:,1])

        dcol = np.sqrt((collar_pts[:,0] - x0)**2 +
                       (collar_pts[:,1] - y0)**2)
        kk   = np.argsort(dcol)[:4]

        cp = collar_pts[kk]
        cn = normals[kk]
        gv = gvals[kk]

        phi_g = reconstruct_ROD4(
            x0, y0,
            stencil_pts,
            stencil_vals,
            cp, cn, gv,
            dx, dy,
            tau,
            use_dirichlet=use_dirichlet
        )

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact(x0, y0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    max_err = np.max(np.abs(reconstructed - exact_vals)/np.abs(exact_vals))
    print(f"N={N:4d}, Dirichlet={'ON' if use_dirichlet else 'OFF'}, max error = {max_err:.6e}")

    return max_err



# ============================================================
#  Main driver
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    use_dirichlet = bool(args.use_dirichlet)

    print("\n============================================")
    print("   ROD4 Quartic Polynomial Reconstruction")
    print("============================================")
    print(f"Dirichlet boundary condition: {'ON' if use_dirichlet else 'OFF'}\n")

    # Resolutions to test
    N_list = [20, 40, 80, 160]
    errors = []

    for N in N_list:
        err = run_test(N, tau_factor=2, use_dirichlet=use_dirichlet)
        errors.append(err)

    print("\nSummary:")
    for N, e in zip(N_list, errors):
        print(f"  N={N:4d}, max error = {e:.6e}")

    # Plot convergence curve
    plot_convergence(N_list, errors, use_dirichlet, L=1.6)

