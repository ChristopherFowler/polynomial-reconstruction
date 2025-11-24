# cython: language_level=3
import numpy as np
cimport numpy as np

# Declare external C function from CUDA code
cdef extern from "poly_cuda.h":
    void cuda_two_pass_reconstruction(
        const float* h_active_coords,
        const float* h_active_vals,
        const int* h_ghost_indices,
        const float* h_xs,
        const float* h_ys,
        const float* h_zs,
        int N_active,
        int N_ghost,
        int Nx, int Ny, int Nz,
        float dx,
        float tau1,
        float tau2,
        float reg_lambda,
        int max_neighbors,
        int max_neighbors_bad,
        float relerr_threshold,
        float phi_rel_clip,
        float* h_reconstructed,
        float* h_exact_vals,
        float* h_rel_err,
        int* h_num_bad
    )

def reconstruct_ghost_cells_cuda(
    np.ndarray[float, ndim=2, mode="c"] active_coords,
    np.ndarray[float, ndim=1, mode="c"] active_vals,
    np.ndarray[int, ndim=2, mode="c"] ghost_indices,
    np.ndarray[float, ndim=1, mode="c"] xs,
    np.ndarray[float, ndim=1, mode="c"] ys,
    np.ndarray[float, ndim=1, mode="c"] zs,
    float dx,
    float tau1,
    float tau2,
    float reg_lambda,
    int max_neighbors,
    int max_neighbors_bad,
    float relerr_threshold,
    float phi_rel_clip
):
    """
    Python wrapper for CUDA two-pass ghost cell reconstruction.
    
    Parameters
    ----------
    active_coords : np.ndarray, shape (N_active, 3), dtype=float32
        Coordinates of active cells
    active_vals : np.ndarray, shape (N_active,), dtype=float32
        Values at active cells
    ghost_indices : np.ndarray, shape (N_ghost, 3), dtype=int32
        Grid indices (i, j, k) of ghost cells
    xs, ys, zs : np.ndarray, dtype=float32
        Grid coordinate arrays
    dx : float
        Grid spacing
    tau1, tau2 : float
        Gaussian width parameters for pass 1 and pass 2
    reg_lambda : float
        Tikhonov regularization parameter
    max_neighbors : int
        Number of neighbors for first pass
    max_neighbors_bad : int
        Number of neighbors for second pass
    relerr_threshold : float
        Relative error threshold to trigger second pass
    phi_rel_clip : float
        Minimum denominator for relative error calculation
        
    Returns
    -------
    reconstructed : np.ndarray, shape (N_ghost,), dtype=float32
        Reconstructed values at ghost cells
    exact_vals : np.ndarray, shape (N_ghost,), dtype=float32
        Exact values at ghost cells
    rel_err : np.ndarray, shape (N_ghost,), dtype=float32
        Relative errors at ghost cells
    num_bad : int
        Number of cells that required second pass
    """
    
    cdef int N_active = active_coords.shape[0]
    cdef int N_ghost = ghost_indices.shape[0]
    cdef int Nx = xs.shape[0]
    cdef int Ny = ys.shape[0]
    cdef int Nz = zs.shape[0]
    
    # Allocate output arrays
    cdef np.ndarray[float, ndim=1, mode="c"] reconstructed = np.empty(N_ghost, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] exact_vals = np.empty(N_ghost, dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] rel_err = np.empty(N_ghost, dtype=np.float32)
    cdef int num_bad = 0
    
    # Call CUDA function
    cuda_two_pass_reconstruction(
        <const float*> active_coords.data,
        <const float*> active_vals.data,
        <const int*> ghost_indices.data,
        <const float*> xs.data,
        <const float*> ys.data,
        <const float*> zs.data,
        N_active,
        N_ghost,
        Nx, Ny, Nz,
        dx,
        tau1,
        tau2,
        reg_lambda,
        max_neighbors,
        max_neighbors_bad,
        relerr_threshold,
        phi_rel_clip,
        <float*> reconstructed.data,
        <float*> exact_vals.data,
        <float*> rel_err.data,
        &num_bad
    )
    
    return reconstructed, exact_vals, rel_err, num_bad
