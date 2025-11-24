import numpy as np
from numpy.linalg import solve


# ============================================================
#  Exact Manufactured Test Function
# ============================================================

def phi_exact(x, y):
    return np.exp(x + 2*y)


# ============================================================
#  Gaussian weight
# ============================================================

def gaussian_weight(r, tau):
    return np.exp(-(r/tau)**2)



# ============================================================
#  Quartic polynomial basis (degree 4 in 2D)
#  15 monomials: X^i * Y^j  for i+j<=4
# ============================================================

def poly_basis4(x, y, x0, y0, dx, dy):
    X = (x - x0)/dx
    Y = (y - y0)/dy

    return np.array([
        1.0,           #  0
        X,             #  1
        Y,             #  2
        X*X,           #  3
        X*Y,           #  4
        Y*Y,           #  5
        X**3,          #  6
        X*X*Y,         #  7
        X*Y*Y,         #  8
        Y**3,          #  9
        X**4,          # 10
        X**3 * Y,      # 11
        X*X * Y*Y,     # 12
        X * Y**3,      # 13
        Y**4           # 14
    ])



# ============================================================
#  Gradient of quartic polynomial basis
# ============================================================

def poly_basis4_grad(x, y, x0, y0, dx, dy):
    X = (x - x0)/dx
    Y = (y - y0)/dy

    dXdx = 1.0/dx
    dYdy = 1.0/dy

    # d/dx: i * X^(i-1) Y^j * dXdx
    dpx = np.array([
        0.0,                        # 1
        dXdx,                       # X
        0.0,                        # Y
        2*X*dXdx,                   # X^2
        Y*dXdx,                     # XY
        0.0,                        # Y^2
        3*X*X*dXdx,                 # X^3
        2*X*Y*dXdx,                 # X^2Y
        Y*Y*dXdx,                   # XY^2
        0.0,                        # Y^3
        4*X**3*dXdx,                # X^4
        3*X**2*Y*dXdx,              # X^3Y
        2*X*Y*Y*dXdx,               # X^2Y^2
        Y**3*dXdx,                  # XY^3
        0.0                         # Y^4
    ])

    # d/dy: j * X^i Y^(j-1) * dYdy
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
#  ROD4 reconstruction (quartic)
# ============================================================

def reconstruct_ROD4(x0, y0,
                     stencil_pts,
                     stencil_vals,
                     collar_pts,
                     normals,
                     gvals,
                     dx, dy,
                     tau):

    m = 15  # number of quartic coefficients
    l = len(stencil_pts)

    M = np.zeros((l, m))
    W = np.zeros((l, l))
    rhs = np.zeros(l)

    # Fill matrices
    for i, (x, y) in enumerate(stencil_pts):
        M[i,:] = poly_basis4(x, y, x0, y0, dx, dy)
        W[i,i] = gaussian_weight(np.sqrt((x-x0)**2 + (y-y0)**2), tau)
        rhs[i] = stencil_vals[i]

    MW = M.T @ W
    A  = MW @ M  # (m x m)

    # Number of collar constraints
    K = len(collar_pts)
    B = np.zeros((K, m))
    b = np.zeros(K)

    for r in range(K):
        xr, yr = collar_pts[r]
        nx, ny = normals[r]

        alpha = 1.0
        beta  = 0.0   # Dirichlet

        basis = poly_basis4(xr, yr, x0, y0, dx, dy)
        dpx, dpy = poly_basis4_grad(xr, yr, x0, y0, dx, dy)

        B[r,:] = alpha*basis + beta*(dpx*nx + dpy*ny)
        b[r]   = gvals[r]

    # Block system:
    #  [A  Bᵀ] [a] = [MW rhs]
    #  [B   0 ] [λ]   [b     ]
    Z = np.zeros((K,K))
    LHS = np.vstack((
        np.hstack((A, B.T)),
        np.hstack((B, Z))
    ))
    RHS = np.hstack((MW @ rhs, b))

    sol = solve(LHS, RHS)
    a = sol[:m]

    return a[0]   # value at center = constant term



# ============================================================
#  Main Test
# ============================================================

def run_test(N=40, tau_factor=1.0):

    L = 1.6
    dx = L / N
    dy = dx

    xs = np.linspace(-0.8, 0.8, N)
    ys = np.linspace(-0.8, 0.8, N)

    R = 0.8

    active = []
    ghost  = []

    # classify cells
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            r = np.sqrt(x*x + y*y)
            if r < R:
                active.append((i,j))
            else:
                for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ii = i+di
                    jj = j+dj
                    if 0 <= ii < N and 0 <= jj < N:
                        if np.sqrt(xs[ii]**2 + ys[jj]**2) < R:
                            ghost.append((i,j))
                            break

    # collar
    Kc = 200
    th = np.linspace(0, 2*np.pi, Kc, endpoint=False)
    collar_pts = np.column_stack((R*np.cos(th), R*np.sin(th)))
    normals    = collar_pts / R
    gvals      = phi_exact(collar_pts[:,0], collar_pts[:,1])

    reconstructed = []
    exact_vals   = []

    tau = tau_factor * dx

    for (i,j) in ghost:
        x0, y0 = xs[i], ys[j]

        # use 45 nearest interior points
        pts  = np.array([(xs[a], ys[b]) for (a,b) in active])
        dist = np.sqrt((pts[:,0]-x0)**2 + (pts[:,1]-y0)**2)
        idx  = np.argsort(dist)[:45]

        stencil_pts  = pts[idx]
        stencil_vals = phi_exact(stencil_pts[:,0], stencil_pts[:,1])

        # use 4 collar constraints
        dcol = np.sqrt((collar_pts[:,0]-x0)**2 + (collar_pts[:,1]-y0)**2)
        kk   = np.argsort(dcol)[:4]

        cp = collar_pts[kk]
        cn = normals[kk]
        gv = gvals[kk]

        phi_g = reconstruct_ROD4(
            x0, y0,
            stencil_pts,
            stencil_vals,
            cp, cn, gv,
            dx, dy, tau
        )

        reconstructed.append(phi_g)
        exact_vals.append(phi_exact(x0, y0))

    reconstructed = np.array(reconstructed)
    exact_vals    = np.array(exact_vals)

    max_err = np.max(np.abs(reconstructed - exact_vals))
    print(f"N={N}, max error = {max_err:e}")

    return max_err



if __name__ == "__main__":
    for N in [20, 40, 80, 160]:
        run_test(N, tau_factor=1.0)

