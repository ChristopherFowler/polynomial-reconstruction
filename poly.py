import numpy as np
from numpy.linalg import solve


# ============================================================
#  Exact Manufactured Test Function
# ============================================================
#
# In Section 3.2 of the paper, the authors evaluate the
# quality of the reconstruction using a smooth analytical
# function:
#
#     φ(x,y) = exp(x + 2 y)
#
# This provides an exact "ground truth" that allows us to
# measure reconstruction error at ghost cells.
#
# ============================================================

def phi_exact(x, y):
    return np.exp(x + 2*y)



# ============================================================
#  Gaussian Weight Kernel
# ============================================================
#
# ROD uses a weighted least-squares functional.
# The weight ω_ij = exp( - (r / τ)^2 ) biases the fit toward
# nearby stencil points and reduces conditioning problems.
#
# τ ("tau") controls kernel width:
#   - small τ → heavy weighting of close points
#   - large τ → nearly uniform weighting
#
# ============================================================

def gaussian_weight(r, tau):
    return np.exp(-(r/tau)**2)



# ============================================================
#  Polynomial Basis for ROD2 (Degree 2 in 2D)
# ============================================================
#
# We use all monomials up to degree 2:
#
#     1, x, y, x², x y, y²
#
# There are 6 basis functions → 6 unknown coefficients.
#
# IMPORTANT:
# The polynomial is centered at the ghost cell (x0, y0),
# AND normalized by grid spacing (dx, dy), exactly as the
# paper prescribes. This keeps the Vandermonde matrix well
# scaled, preventing numerical blow-up.
#
# ============================================================

def poly_basis(x, y, x0, y0, dx, dy):
    X = (x - x0)/dx
    Y = (y - y0)/dy
    return np.array([
        1,        # constant term
        X,        # x
        Y,        # y
        X*X,      # x^2
        X*Y,      # xy
        Y*Y       # y^2
    ])



# ============================================================
#  Gradient of the Polynomial Basis
# ============================================================
#
# We need ∇π(x,y) evaluated at boundary ("collar") points
# when imposing boundary constraints.
#
# For Dirichlet BC (α=1, β=0), this gradient is not used,
# but we implement it because:
#
#   1. It completes the ROD formula, and
#   2. You will need it for Neumann or Robin BC.
#
# ============================================================

def poly_basis_grad(x, y, x0, y0, dx, dy):
    X = (x - x0)/dx
    Y = (y - y0)/dy

    # d/dx basis
    dpx = np.array([
        0,          # d/dx 1     = 0
        1/dx,       # d/dx X     = 1/dx
        0,          # d/dx Y     = 0
        2*X/dx,     # d/dx X^2   = 2X (scaled)
        Y/dx,       # d/dx XY    = Y
        0           # d/dx Y^2   = 0
    ])

    # d/dy basis
    dpy = np.array([
        0,
        0,
        1/dy,
        0,
        X/dy,       # derivative of XY = X
        2*Y/dy
    ])

    return dpx, dpy



# ============================================================
#  Core ROD2 Reconstruction Routine
# ============================================================
#
# Given:
#   * a ghost cell (x0, y0)
#   * stencil interior points and values
#   * collar (boundary) points + normals
#   * Dirichlet boundary values g
#
# This function computes:
#   the ROD polynomial coefficients a
#   and returns φ(x0, y0) = a₀
#
# This implements Eq. (4) of the paper:
#
#     [ Mᵀ W M   Bᵀ ][ a ] = [ Mᵀ W Φ ]
#     [   B        0 ][ λ ]   [   g    ]
#
# Except:
#   we directly compute K = MᵀWM, and solve the block system.
#
# ============================================================

def reconstruct_ROD2(x0, y0,
                     stencil_pts,
                     stencil_vals,
                     collar_pts,
                     normals,
                     gvals,
                     dx, dy,
                     tau):

    # Number of polynomial coefficients for degree 2
    m = 6

    # Number of interior stencil points
    l = len(stencil_pts)

    # Allocate matrices
    M = np.zeros((l, m))     # Vandermonde-like matrix
    W = np.zeros((l, l))     # diagonal weight matrix
    rhs = np.zeros(l)        # interior data values

    # --------------------------------------------------------
    # Fill M, W, rhs using interior stencil points
    # --------------------------------------------------------
    for i, (x, y) in enumerate(stencil_pts):
        M[i,:] = poly_basis(x, y, x0, y0, dx, dy)
        W[i,i] = gaussian_weight(np.sqrt((x-x0)**2 + (y-y0)**2), tau)
        rhs[i] = stencil_vals[i]

    # Compute A = Mᵀ W M   (correct formula)
    MW = M.T @ W    # (m × l)
    A  = MW @ M     # (m × m)

    # --------------------------------------------------------
    # Boundary constraints (Dirichlet)
    # We use the *two closest collar points*,
    # which is exactly how the ROD method enforces BC.
    #
    # For Dirichlet:
    #     α = 1
    #     β = 0
    #
    # Each constraint is:
    #     α π(x_r,y_r) + β ∇π⋅n = g_r
    # --------------------------------------------------------
    B = np.zeros((2, m))
    b = np.zeros(2)

    for r in range(2):
        xr, yr = collar_pts[r]
        nx, ny = normals[r]

        alpha = 1.0   # Dirichlet
        beta  = 0.0

        basis = poly_basis(xr, yr, x0, y0, dx, dy)
        dpx, dpy = poly_basis_grad(xr, yr, x0, y0, dx, dy)

        B[r,:] = alpha * basis + beta * (dpx*nx + dpy*ny)
        b[r]   = gvals[r]

    # --------------------------------------------------------
    # Assemble the block matrix system
    #
    #     [ A   Bᵀ ] [a] = [MW rhs]
    #     [ B    0 ] [λ]   [  g   ]
    #
    # The Lagrange multipliers λ enforce the boundary
    # conditions exactly, while a contains our polynomial
    # coefficients.
    # --------------------------------------------------------

    Z = np.zeros((2,2))       # the lower-right 0 block
    top    = np.hstack((A, B.T))
    bottom = np.hstack((B, Z))
    LHS = np.vstack((top, bottom))

    RHS = np.hstack((MW @ rhs, b))

    # Solve the full KKT system
    sol = solve(LHS, RHS)

    a = sol[:m]   # polynomial coefficients

    # Value at center = constant term of polynomial
    return a[0]



# ============================================================
#  Main Test: ROD2 on a Circular Domain
# ============================================================
#
# This reproduces the spirit of the benchmark in Section 3.2:
#
#   * Build circular domain
#   * Identify active and ghost cells
#   * Build collar points (boundary samples)
#   * Reconstruct ghost values with ROD2
#   * Compare to exact φ(x,y)
#
# ============================================================

def run_test(N=40, tau_factor=1.0):

    # Grid spanning [-0.8, 0.8] × [-0.8, 0.8]
    L = 1.6
    dx = L / N
    dy = dx

    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)

    R = 0.8    # circle radius

    active = []
    ghost  = []

    # --------------------------------------------------------
    # Identify active and ghost cells
    # A cell is "active" if inside circle.
    # It's "ghost" if it is outside, but has a neighbor inside.
    # --------------------------------------------------------
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            r = np.sqrt(x**2 + y**2)

            if r < R:
                active.append((i,j))
            else:
                # test its Von Neumann neighbors
                for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < N and 0 <= jj < N:
                        if np.sqrt(xs[ii]**2 + ys[jj]**2) < R:
                            ghost.append((i,j))
                            break

    # --------------------------------------------------------
    # Build collar: uniformly sample the circle boundary
    # --------------------------------------------------------
    K = 200
    theta = np.linspace(0, 2*np.pi, K, endpoint=False)

    collar_pts = np.column_stack((R*np.cos(theta), R*np.sin(theta)))
    normals    = collar_pts / R                # outward normals
    gvals      = phi_exact(collar_pts[:,0],    # Dirichlet BC
                           collar_pts[:,1])

    reconstructed = []
    exact_vals   = []

    tau = tau_factor * dx   # kernel width

    # --------------------------------------------------------
    # Loop over ghost cells and reconstruct using ROD2
    # --------------------------------------------------------
    for (i,j) in ghost:

        x0, y0 = xs[i], ys[j]

        # Build stencil of nearest 12 active cells
        pts = np.array([(xs[ia], ys[ja]) for (ia,ja) in active])
        dists = np.sqrt((pts[:,0]-x0)**2 + (pts[:,1]-y0)**2)

        idx = np.argsort(dists)[:12]

        stencil_pts  = pts[idx]
        stencil_vals = phi_exact(stencil_pts[:,0], stencil_pts[:,1])

        # Pick 2 closest collar points
        dcol = np.sqrt((collar_pts[:,0]-x0)**2 + (collar_pts[:,1]-y0)**2)
        kk = np.argsort(dcol)[:2]

        cp = collar_pts[kk]
        cn = normals[kk]
        gv = gvals[kk]

        # Perform ROD2 reconstruction
        phi_g = reconstruct_ROD2(x0, y0,
                                 stencil_pts,
                                 stencil_vals,
                                 cp, cn, gv,
                                 dx, dy, tau)

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact(x0, y0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    # Compute error
    max_err = np.max(np.abs(reconstructed - exact_vals))
    print(f"N={N},  max error = {max_err:e}")

    return max_err



# ============================================================
#  Run several grids to observe convergence behavior
# ============================================================

if __name__ == "__main__":
    for N in [20, 40, 80, 160]:
        run_test(N, tau_factor=1.0)

