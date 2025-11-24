import numpy as np

# ============================================================
#  Exact manufactured test function in 3D
# ============================================================
# Smooth function used as the "true" solution:
#
#   φ(x,y,z) = exp(x + 2 y + 3 z)
#
# We will reconstruct φ in ghost cells and compare.
# ============================================================

def phi_exact_3d(x, y, z):
    return np.exp(x + 2.0*y + 3.0*z)


# ============================================================
#  Gaussian weight kernel
# ============================================================
# w(r) = exp( - (r / τ)^2 )
#
# τ ("tau") is a length scale; we'll set τ ≈ tau_factor * dx.
# ============================================================

def gaussian_weight(r, tau):
    return np.exp(-(r / tau)**2)


# ============================================================
#  Generate monomial exponents for 3D polynomial up to degree p
# ============================================================
#
# We work with normalized coordinates:
#   X = (x - x0) / dx, etc.
#
# The polynomial basis consists of all monomials:
#
#   X^i Y^j Z^k   with  i + j + k ≤ degree.
#
# This function returns a list of exponent triplets (i,j,k)
# ordered by total degree, then lexicographically.
#
# Number of basis terms m = C(degree+3, 3) = (degree+3 choose 3).
# ------------------------------------------------------------

def generate_exponents_3d(degree):
    exps = []
    for d in range(degree + 1):     # total degree
        for i in range(d + 1):
            for j in range(d - i + 1):
                k = d - i - j
                exps.append((i, j, k))
    return exps


# ============================================================
#  Evaluate general 3D polynomial basis for given exponents
# ============================================================
#
# Given:
#   - coordinate (x,y,z)
#   - ghost cell center (x0,y0,z0)
#   - grid spacing (dx,dy,dz)
#   - exponents[(i,j,k), ...]
#
# We compute:
#   X = (x - x0)/dx, etc.
#   basis[m] = X^i Y^j Z^k
# ============================================================

def poly_basis_3d(x, y, z, x0, y0, z0, dx, dy, dz, exponents):
    X = (x - x0) / dx
    Y = (y - y0) / dy
    Z = (z - z0) / dz

    m = len(exponents)
    out = np.empty(m, dtype=float)
    for idx, (i, j, k) in enumerate(exponents):
        out[idx] = (X ** i) * (Y ** j) * (Z ** k)
    return out


# ============================================================
#  Stable 3D polynomial LSQ reconstruction at one ghost cell
# ============================================================
#
# We use a *penalized LSQ* approach instead of a KKT system to
# enforce boundary constraints, because it is much more robust:
#
#   Minimize over a:
#
#     ||W^{1/2} (M a - f)||^2  +  μ ||C a - g||^2 + λ ||a||^2
#
# where:
#   - M: interior stencil matrix (rows = basis at stencil points)
#   - f: interior values (exact φ at stencil points)
#   - W: diagonal weight matrix (Gaussian weights)
#   - C: boundary (collar) matrix (rows = basis at collar pts)
#   - g: Dirichlet values at collar pts (exact φ)
#   - μ: penalty weight for boundary (large, e.g. 1e4)
#   - λ: small Tikhonov regularization (e.g. 1e-8)
#
# Normal equations give:
#
#   A a = b
#
#   A = Mᵀ W M + μ Cᵀ C + λ I
#   b = Mᵀ W f + μ Cᵀ g
#
# A is symmetric positive definite for μ>0, λ>0 → no singular.
#
# Inputs:
#   x0,y0,z0       : ghost cell center
#   stencil_pts    : S×3 array of interior points
#   stencil_vals   : S array with φ at those points
#   collar_pts     : K×3 array of boundary points
#   collar_vals    : K array with φ at boundary points
#   dx,dy,dz       : grid spacings
#   exponents      : monomial exponents [(i,j,k),...]
#   tau            : kernel length scale
#   lam_reg        : regularization λ
#   mu_bc          : boundary penalty μ
#   cond_samples   : optional dict to record condition numbers
#
# Returns:
#   reconstructed φ at ghost cell center.
# ============================================================

def reconstruct_LSQ_3d(x0, y0, z0,
                       stencil_pts,
                       stencil_vals,
                       collar_pts,
                       collar_vals,
                       dx, dy, dz,
                       exponents,
                       tau,
                       lam_reg=1e-8,
                       mu_bc=1e4,
                       cond_samples=None):

    m = len(exponents)
    S = len(stencil_pts)
    K = len(collar_pts)

    # Build interior matrices M, W, rhs (f)
    M   = np.zeros((S, m), dtype=float)
    W   = np.zeros((S, S), dtype=float)
    rhs = np.asarray(stencil_vals, dtype=float)

    for i in range(S):
        x, y, z = stencil_pts[i]
        M[i, :] = poly_basis_3d(x, y, z, x0, y0, z0, dx, dy, dz, exponents)
        r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
        W[i, i] = gaussian_weight(r, tau)

    # Build boundary matrix C and vector g
    C = np.zeros((K, m), dtype=float)
    g = np.asarray(collar_vals, dtype=float)

    for k in range(K):
        xb, yb, zb = collar_pts[k]
        C[k, :] = poly_basis_3d(xb, yb, zb, x0, y0, z0, dx, dy, dz, exponents)

    # Assemble A and b (normal equations of penalized LSQ)
    MW = M.T @ W
    A = MW @ M                   # Mᵀ W M
    A += mu_bc * (C.T @ C)       # + μ Cᵀ C
    A += lam_reg * np.eye(m)     # + λ I

    b = MW @ rhs + mu_bc * (C.T @ g)

    # Optional: record condition number statistics
    if cond_samples is not None and cond_samples["count"] < cond_samples["max"]:
        cond_val = np.linalg.cond(A)
        cond_samples["values"].append(cond_val)
        cond_samples["count"] += 1

    # Solve A a = b (SPD, small dimension → cheap)
    a = np.linalg.solve(A, b)

    # Value at ghost center:
    # at X=Y=Z=0, only monomial (0,0,0) is 1, others vanish → a[0]
    return a[0]


# ============================================================
#  Build spherical test and run LSQ reconstruction for one N
# ============================================================
#
# Steps:
#   1) Build uniform grid in [-0.8,0.8]^3 (Nx×Ny×Nz with N cells each)
#   2) Mark:
#        active cells: inside sphere of radius R
#        ghost cells : outside but with at least one active neighbor
#   3) Build collar points on the sphere r=R
#   4) For each ghost cell:
#        - pick stencil of nearest active points
#        - pick boundary (collar) constraints
#        - call reconstruct_LSQ_3d
#        - compare with exact φ
#   5) Return max error and conditioning stats
# ============================================================

def run_test_3d_LSQ(degree,
                    N,
                    tau_factor=1.5,
                    lam_reg=1e-8,
                    mu_bc=1e4,
                    stencil_factor=4.0,
                    bc_factor=0.8,
                    max_cond_samples=200):

    # Generate exponent list for this degree
    exponents = generate_exponents_3d(degree)
    m = len(exponents)

    # Domain box and spacing
    L = 1.6
    dx = L / N
    dy = dx
    dz = dx

    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)
    zs = np.linspace(-0.8, 0.8, N)

    R = 0.8  # sphere radius

    active = []
    ghost  = []

    # Mark active and ghost cells
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                r = np.sqrt(x**2 + y**2 + z**2)
                if r < R:
                    active.append((i, j, k))
                else:
                    # Ghost if any 6-face-neighbor is active
                    for di, dj, dk in [
                        (1,0,0), (-1,0,0),
                        (0,1,0), (0,-1,0),
                        (0,0,1), (0,0,-1)
                    ]:
                        ii = i + di
                        jj = j + dj
                        kk = k + dk
                        if (0 <= ii < N and
                            0 <= jj < N and
                            0 <= kk < N):
                            xnb = xs[ii]
                            ynb = ys[jj]
                            znb = zs[kk]
                            if np.sqrt(xnb**2 + ynb**2 + znb**2) < R:
                                ghost.append((i, j, k))
                                break

    if not active or not ghost:
        raise RuntimeError("No active or ghost cells found; check N or geometry.")

    # Build collar points on the sphere r=R (spherical parameterization)
    n_theta = 24
    n_phi   = 48
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=True)
    phis   = np.linspace(0.0, 2.0*np.pi, n_phi, endpoint=False)

    collar_list = []
    for th in thetas:
        st = np.sin(th)
        ct = np.cos(th)
        for ph in phis:
            cp = np.cos(ph)
            sp = np.sin(ph)
            x = R * st * cp
            y = R * st * sp
            z = R * ct
            collar_list.append((x, y, z))

    collar_pts = np.array(collar_list, dtype=float)
    collar_vals = phi_exact_3d(collar_pts[:,0], collar_pts[:,1], collar_pts[:,2])

    # Precompute active coordinates
    active_coords = np.array([
        (xs[i], ys[j], zs[k]) for (i, j, k) in active
    ], dtype=float)

    reconstructed = []
    exact_vals    = []

    # Condition number sampling structure
    cond_samples = {
        "values": [],
        "count": 0,
        "max": max_cond_samples
    }

    tau = tau_factor * dx

    # Number of interior stencil points:
    #   S ≈ stencil_factor * m
    S_target = int(stencil_factor * m)

    # Number of boundary constraints (collar points):
    #   Kc ≈ max(bc_factor*m, m/2, 10)
    Kc_target = max(int(bc_factor * m), m // 2, 10)
    Kc_target = min(Kc_target, len(collar_pts))

    for (i, j, k) in ghost:
        x0, y0, z0 = xs[i], ys[j], zs[k]

        # Interior stencil: nearest S_target active points
        dists = np.linalg.norm(active_coords - np.array([x0, y0, z0]), axis=1)
        S = min(S_target, len(active_coords))
        stencil_idx = np.argsort(dists)[:S]
        stencil_pts  = active_coords[stencil_idx]
        stencil_vals = phi_exact_3d(stencil_pts[:,0],
                                    stencil_pts[:,1],
                                    stencil_pts[:,2])

        # Boundary constraints: Kc_target nearest collar points
        dcol = np.linalg.norm(collar_pts - np.array([x0, y0, z0]), axis=1)
        Kc = min(Kc_target, len(collar_pts))
        cidx = np.argsort(dcol)[:Kc]
        cp = collar_pts[cidx]
        gv = collar_vals[cidx]

        phi_g = reconstruct_LSQ_3d(
            x0, y0, z0,
            stencil_pts,
            stencil_vals,
            cp,
            gv,
            dx, dy, dz,
            exponents,
            tau,
            lam_reg=lam_reg,
            mu_bc=mu_bc,
            cond_samples=cond_samples
        )

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact_3d(x0, y0, z0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    # Error metric: max-norm on ghost cells
    max_err = np.max(np.abs(reconstructed - exact_vals))

    # Condition number statistics
    cond_values = np.array(cond_samples["values"], dtype=float) if cond_samples["values"] else None
    cond_stats = None
    if cond_values is not None and cond_values.size > 0:
        cond_stats = {
            "min": float(np.min(cond_values)),
            "max": float(np.max(cond_values)),
            "mean": float(np.mean(cond_values))
        }

    print(
        f"Degree={degree}, N={N}, ghost cells={len(ghost)}, "
        f"max error={max_err:.6e}"
    )
    if cond_stats is not None:
        print(
            f"  LSQ matrix condition number (sampled {len(cond_values)} cases): "
            f"min={cond_stats['min']:.3e}, "
            f"mean={cond_stats['mean']:.3e}, "
            f"max={cond_stats['max']:.3e}"
        )

    return max_err, cond_stats


# ============================================================
#  Run convergence study for a given degree and list of N's
# ============================================================
#
# Produces a small "convergence curve" (N vs error).
# Optionally attempts to plot log-log curves if matplotlib
# is installed; otherwise only prints.
# ============================================================

def run_convergence_suite(degree, N_list,
                          tau_factor=1.5,
                          lam_reg=1e-8,
                          mu_bc=1e4,
                          stencil_factor=4.0,
                          bc_factor=0.8):

    print(f"\n=== Convergence for degree {degree} ===")
    errors = []
    for N in N_list:
        err, _ = run_test_3d_LSQ(
            degree=degree,
            N=N,
            tau_factor=tau_factor,
            lam_reg=lam_reg,
            mu_bc=mu_bc,
            stencil_factor=stencil_factor,
            bc_factor=bc_factor
        )
        errors.append(err)

    # Print summary table
    print("\nSummary (degree {}, tau_factor={}):".format(degree, tau_factor))
    print("  N      max_error")
    for N, e in zip(N_list, errors):
        print(f"  {N:3d}   {e:.6e}")

    # Try to plot convergence curve if matplotlib exists
    try:
        import matplotlib.pyplot as plt
        hs = [1.6 / N for N in N_list]  # grid spacing
        plt.figure()
        plt.loglog(hs, errors, marker="o", label=f"degree {degree}")
        plt.gca().invert_xaxis()
        plt.xlabel("h (grid spacing)")
        plt.ylabel("max error on ghost cells")
        plt.title(f"3D LSQ reconstruction convergence (degree {degree})")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib not available: skipping plot.")


# ============================================================
#  Main driver
# ============================================================

if __name__ == "__main__":
    # You can adjust these as you like:
    #   - degrees: which polynomial degrees to test
    #   - N_list : grid resolutions

    degrees = [2, 3, 4]          # quadratic, cubic, quartic
    N_list  = [10, 14, 18, 22]   # moderate resolutions

    for deg in degrees:
        run_convergence_suite(deg, N_list)

