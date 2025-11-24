import numpy as np
import matplotlib.pyplot as plt
from vedo import Points, Plotter, Arrows, Sphere, Text2D

# ===============================================================
# 0. Utility functions
# ===============================================================
def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

def compute_local_frame(normal):
    n = normalize(normal)
    # Pick a reference vector not parallel to n
    if abs(n[0]) < 0.8:
        ref = np.array([1,0,0])
    else:
        ref = np.array([0,1,0])
    t1 = normalize(np.cross(n, ref))
    t2 = normalize(np.cross(n, t1))
    return t1, t2, n

def ghost_distortion(X, Y, Z, mode="gradient"):
    if mode == "gradient":
        return 0.1*(X + Y + Z)
    elif mode == "random":
        return 0.2*(np.random.rand(*X.shape)-0.5)
    elif mode == "oscillatory":
        return 0.1*np.sin(10*X)*np.cos(12*Y)
    return 0.0


# ===============================================================
# 1. Build 3D grid and solid sphere
# ===============================================================
Nx = Ny = Nz = 25
x = np.linspace(0,1,Nx); dx = x[1]-x[0]
y = np.linspace(0,1,Ny); dy = y[1]-y[0]
z = np.linspace(0,1,Nz); dz = z[1]-z[0]

X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

center = np.array([0.5,0.5,0.5])
R = 0.28
phi = np.sqrt((X-center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2) - R

is_solid = phi < 0
is_fluid = ~is_solid

# ===============================================================
# 2. Exact field + analytic derivatives
# ===============================================================
def u_exact(x,y,z):
    return np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + 0.5*z**2

def grad_exact(x,y,z):
    return np.array([
        (2*np.pi)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y),
        -(2*np.pi)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y),
        z
    ])

def hess_exact(x,y,z):
    s = (2*np.pi)**2
    dxx = -s*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    dyy = -s*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    dzz = 1.0
    dxy = -s*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    return np.array([[dxx, dxy, 0.0],
                     [dxy, dyy, 0.0],
                     [0.0, 0.0, dzz]])


# ===============================================================
# 3. LS Reconstruction (Cartesian or Surface-Aligned)
# ===============================================================
def lsq_quad_reconstruct(u, is_fluid, idx0,
                         dx=(dx,dy,dz),
                         radius_cells=2.5,
                         ghost_weight=0.25,
                         normal=None,
                         use_surface_frame=False):

    Nx,Ny,Nz = u.shape
    i0,j0,k0 = idx0
    xc = np.array([x[i0], y[j0], z[k0]])

    # Build frame
    if use_surface_frame:
        assert normal is not None
        t1,t2,nrm = compute_local_frame(normal)
        Rmat = np.vstack([t1, t2, nrm])  # 3×3
    else:
        Rmat = None

    rows=[]
    vals=[]
    wts=[]

    # Physical radius
    h_char = (dx[0]+dx[1]+dx[2])/3
    Rphys = radius_cells*h_char

    # Collect neighbors
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                xp = np.array([x[i],y[j],z[k]])
                d = xp - xc
                r = np.linalg.norm(d)
                if r==0 or r>Rphys: continue

                if use_surface_frame:
                    d = Rmat @ d  # rotate into local frame

                dxp,dyp,dzp = d
                basis = np.array([
                    1.0,
                    dxp, dyp, dzp,
                    dxp**2, dyp**2, dzp**2,
                    dxp*dyp, dxp*dzp, dyp*dzp
                ])
                rows.append(basis)
                vals.append(u[i,j,k])

                wd = np.exp(-4*(r/Rphys)**2)
                wt = wd if is_fluid[i,j,k] else ghost_weight*wd
                wts.append(wt)

    A = np.array(rows)
    b = np.array(vals)
    W = np.sqrt(np.array(wts))

    Ahat = A * W[:,None]
    bhat = b * W

    a, *_ = np.linalg.lstsq(Ahat, bhat, rcond=None)
    a0,a1,a2,a3,a4,a5,a6,a7,a8,a9 = a

    grad = np.array([a1,a2,a3])
    Hess = np.array([
        [2*a4,   a7,   a8],
        [a7,     2*a5, a9],
        [a8,     a9,   2*a6]
    ])

    # If using surface frame, rotate gradient/Hessian back
    if use_surface_frame:
        grad = Rmat.T @ grad
        Hess = Rmat.T @ Hess @ Rmat.T

    return grad, Hess, a


# ===============================================================
# 4. Distortion sweep + error mapping
# ===============================================================
def distort_field(eps, mode="gradient"):
    u = u_exact(X,Y,Z)
    u[is_solid] += eps * ghost_distortion(X[is_solid],Y[is_solid],Z[is_solid],mode)
    return u

def find_fluid_adjacent():
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            for k in range(1,Nz-1):
                if not is_fluid[i,j,k]: continue
                neigh=[(i+1,j,k),(i-1,j,k),(i,j+1,k),(i,j-1,k),(i,j,k+1),(i,j,k-1)]
                for ii,jj,kk in neigh:
                    if is_solid[ii,jj,kk]:
                        return (i,j,k)
    raise RuntimeError("no interface fluid cell found")

i0,j0,k0 = find_fluid_adjacent()
xc = np.array([x[i0],y[j0],z[k0]])


# ===============================================================
# 5. Plot 1: LS vs Exact for a fixed epsilon
# ===============================================================
eps0 = 0.25
u_d0 = distort_field(eps0)
normal_here = normalize(np.array([X[i0,j0,k0]-center[0],
                                  Y[i0,j0,k0]-center[1],
                                  Z[i0,j0,k0]-center[2]]))

grad_ls, Hess_ls, _ = lsq_quad_reconstruct(u_d0, is_fluid,
                                           (i0,j0,k0),
                                           use_surface_frame=False)

grad_ex = grad_exact(*xc)

plt.figure(figsize=(7,5))
labels=['gx','gy','gz']
xpos=np.arange(3)
plt.bar(xpos-0.15, grad_ex, width=0.3, label="Exact")
plt.bar(xpos+0.15, grad_ls, width=0.3, label="LSQ")
plt.xticks(xpos,labels)
plt.title("Gradient Comparison at Fluid–Solid Interface")
plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ===============================================================
# 6. Plot 2: Error vs distortion epsilon
# ===============================================================
eps_list=[0.0,0.05,0.1,0.2,0.3,0.5,0.75]
err_list=[]

for eps in eps_list:
    u_d = distort_field(eps)
    grad_ls,_,_ = lsq_quad_reconstruct(u_d, is_fluid,
                                       (i0,j0,k0),
                                       use_surface_frame=False)
    err=np.linalg.norm(grad_ls - grad_ex)
    err_list.append(err)

plt.figure(figsize=(7,5))
plt.plot(eps_list, err_list, '-o')
plt.xlabel("Distortion ε")
plt.ylabel("||grad_LS - grad_exact||")
plt.title("Error vs Ghost Distortion Strength")
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()


# ===============================================================
# 7. Error map over the entire fluid–solid interface
# ===============================================================
interface_pts = []
interface_errs = []

u_dist_map = distort_field(0.3)   # pick distortion for visualization

for i in range(1,Nx-1):
    for j in range(1,Ny-1):
        for k in range(1,Nz-1):
            if not is_fluid[i,j,k]: continue
            # check adjacency
            neigh=[(i+1,j,k),(i-1,j,k),(i,j+1,k),(i,j-1,k),(i,j,k+1),(i,j,k-1)]
            if not any(is_solid[ii,jj,kk] for ii,jj,kk in neigh):
                continue

            grad_ls,_,_ = lsq_quad_reconstruct(u_dist_map,is_fluid,(i,j,k))
            grad_ex_loc=grad_exact(x[i],y[j],z[k])
            err = np.linalg.norm(grad_ls-grad_ex_loc)

            interface_errs.append(err)
            interface_pts.append([x[i],y[j],z[k]])

interface_pts=np.array(interface_pts)
interface_errs=np.array(interface_errs)

# Normalize error for colors
colors = (interface_errs - interface_errs.min())/(interface_errs.max()-interface_errs.min()+1e-12)



####################################################################
# 9. Interactive visualization: slider to control distortion ε
####################################################################
from vedo import Plotter, Points, Slider2D, Text2D
import numpy as np

# --- Precompute interface points (fixed geometry; only error changes) ---
interface_pts = []
interface_normals = []
interface_indices = []

for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        for k in range(1, Nz-1):
            if not is_fluid[i,j,k]:
                continue

            nbs = [(i+1,j,k),(i-1,j,k),(i,j+1,k),(i,j-1,k),(i,j,k+1),(i,j,k-1)]
            if not any(is_solid[ii,jj,kk] for ii,jj,kk in nbs):
                continue

            interface_pts.append([x[i],y[j],z[k]])
            interface_normals.append([
                X[i,j,k]-center[0],
                Y[i,j,k]-center[1],
                Z[i,j,k]-center[2]
            ])
            interface_indices.append((i,j,k))

interface_pts = np.array(interface_pts)
interface_normals = np.array(interface_normals)
interface_normals = np.array([n/np.linalg.norm(n) for n in interface_normals])

# -------------------------------------------------------------------------
# Compute LSQ gradient error for a given distortion epsilon
# -------------------------------------------------------------------------
def compute_error_field(eps):
    u_d = u_exact(X,Y,Z).copy()
    u_d[is_solid] += eps * ghost_distortion(
        X[is_solid], Y[is_solid], Z[is_solid], mode="gradient"
    )

    errs = []
    for idx, (i,j,k) in enumerate(interface_indices):
        grad_ls, _, _ = lsq_quad_reconstruct(u_d, is_fluid, (i,j,k))
        grad_ex_loc = grad_exact(x[i], y[j], z[k])
        errs.append(np.linalg.norm(grad_ls - grad_ex_loc))
    errs = np.array(errs)

    # Normalize for colors
    vmin, vmax = errs.min(), errs.max()
    if vmax - vmin < 1e-12:
        norm = np.zeros_like(errs)
    else:
        norm = (errs - vmin) / (vmax - vmin)

    return errs, norm


####################################################################
# 9. Interactive visualization: slider to control distortion ε
####################################################################
from vedo import Plotter, Points, Sphere, Text2D
import numpy as np

eps0 = 0.3
errors, colors = compute_error_field(eps0)

plt = Plotter(title="Interactive Distortion Slider", axes=1)

pf = Points(interface_pts, r=10)
pf.cmap("jet", colors)

solid_mesh = Sphere(pos=center, r=R, c='gray8', alpha=0.2)
txt = Text2D(f"ε = {eps0:.2f}", pos="top-left", s=1.2, c="white")


# ---------------------------------------------------------
# Slider callback (old-style Vedo)
# ---------------------------------------------------------
def slider_callback(widget, event):
    eps = widget.value  # <-- This is correct for add_slider3d()
    errors, colors = compute_error_field(eps)
    pf.cmap("jet", colors)
    txt.text(f"ε = {eps:.2f}")
    plt.render()


# ---------------------------------------------------------
# OLD API: add_slider3d(callback, min, max, value, p1, p2, title)
# ---------------------------------------------------------
p1 = list(center + np.array([R*1.4, 0.0, 0.0]))
p2 = list(center + np.array([R*1.4, 0.0, 0.5]))

plt.add_slider3d(
    slider_callback,     # callback
    0.0,                 # xmin
    0.6,                 # xmax
    eps0,                # initial value
    p1,                  # 3D point 1
    p2,                  # 3D point 2
    "Distortion ε"       # title
)

plt.show(pf, solid_mesh, txt, interactive=True)


# ===============================================================
# 8. 3D visualization with vedo
# ===============================================================
#pf = Points(interface_pts, r=10, c=None)
#pf.cmap("jet",colors)
#pf.name = "Interface Error Map"

#solid_coords = np.column_stack([X[is_solid],Y[is_solid],Z[is_solid]])
#ps = Points(solid_coords, r=3, c='gray').legend("Solid")

#ls_pt = Points([[x[i0],y[j0],z[k0]]], r=18, c='yellow').legend("Reconstruction point")

#vp = Plotter(title="Interface Error Map", axes=1)
#vp.show(ps, pf, ls_pt, Text2D("Error colorbar", pos="top-center"), interactive=True)

