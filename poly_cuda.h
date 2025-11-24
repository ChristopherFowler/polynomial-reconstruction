#ifndef POLY_CUDA_H
#define POLY_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for reconstruction context (holds all device memory)
typedef struct PolyReconstructionContext_t* PolyReconstructionContext;

// Allocate device memory and prepare context for repeated reconstructions
// Call this once before many reconstruction calls
PolyReconstructionContext cuda_reconstruction_init(
    int N_active,
    int N_ghost,
    int Nx, int Ny, int Nz
);

// Copy static domain data to device (grid coords, indices)
// Call this once after init, or whenever domain structure changes
// This is separate from values which change every timestep
void cuda_reconstruction_set_domain(
    PolyReconstructionContext ctx,
    const double* h_active_coords,      // [N_active, 3] flattened - positions don't change
    const int* h_ghost_indices,         // [N_ghost, 3] flattened - ghost positions don't change
    const double* h_xs,                 // [Nx] - grid doesn't change
    const double* h_ys,                 // [Ny]
    const double* h_zs                  // [Nz]
);

// Free all device memory associated with the context
// Call this once when done with all reconstructions
void cuda_reconstruction_free(PolyReconstructionContext ctx);

// Perform reconstruction using pre-allocated device memory
// This function ONLY copies active_vals (the data that changes each timestep)
// Most efficient - call this many times per timestep if needed
void cuda_reconstruction_execute(
    PolyReconstructionContext ctx,
    const double* h_active_vals,        // [N_active] - ONLY field that changes!
    double dx,
    double tau1,
    double tau2,
    double reg_lambda,
    int max_neighbors,
    int max_neighbors_bad,
    double relerr_threshold,
    double phi_rel_clip,
    double* h_reconstructed,            // [N_ghost] output
    double* h_exact_vals,               // [N_ghost] output
    double* h_rel_err,                  // [N_ghost] output
    int* h_num_bad                      // scalar output
);

// Legacy function: Execute reconstruction with all data (if domain changes)
// This copies everything - use only if domain structure changes
void cuda_reconstruction_execute_full(
    PolyReconstructionContext ctx,
    const double* h_active_coords,
    const double* h_active_vals,
    const int* h_ghost_indices,
    const double* h_xs,
    const double* h_ys,
    const double* h_zs,
    double dx,
    double tau1,
    double tau2,
    double reg_lambda,
    int max_neighbors,
    int max_neighbors_bad,
    double relerr_threshold,
    double phi_rel_clip,
    double* h_reconstructed,
    double* h_exact_vals,
    double* h_rel_err,
    int* h_num_bad
);

// Original all-in-one function (for backward compatibility)
// This allocates/frees memory on every call - use the above functions for better performance
void cuda_two_pass_reconstruction(
    const double* h_active_coords,      // [N_active, 3] flattened
    const double* h_active_vals,        // [N_active]
    const int* h_ghost_indices,        // [N_ghost, 3] flattened
    const double* h_xs,                 // [Nx]
    const double* h_ys,                 // [Ny]
    const double* h_zs,                 // [Nz]
    int N_active,
    int N_ghost,
    int Nx, int Ny, int Nz,
    double dx,
    double tau1,
    double tau2,
    double reg_lambda,
    int max_neighbors,
    int max_neighbors_bad,
    double relerr_threshold,
    double phi_rel_clip,
    double* h_reconstructed,            // [N_ghost] output
    double* h_exact_vals,               // [N_ghost] output
    double* h_rel_err,                  // [N_ghost] output
    int* h_num_bad                     // scalar output
);

#ifdef __cplusplus
}
#endif

#endif // POLY_CUDA_H
