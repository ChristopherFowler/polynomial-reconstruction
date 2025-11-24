#ifndef POLY_CUDA_H
#define POLY_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_two_pass_reconstruction(
    const float* h_active_coords,      // [N_active, 3] flattened
    const float* h_active_vals,        // [N_active]
    const int* h_ghost_indices,        // [N_ghost, 3] flattened
    const float* h_xs,                 // [Nx]
    const float* h_ys,                 // [Ny]
    const float* h_zs,                 // [Nz]
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
    float* h_reconstructed,            // [N_ghost] output
    float* h_exact_vals,               // [N_ghost] output
    float* h_rel_err,                  // [N_ghost] output
    int* h_num_bad                     // scalar output
);

#ifdef __cplusplus
}
#endif

#endif // POLY_CUDA_H
