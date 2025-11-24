#include "poly_cuda.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstdio>

// Constants
#define MAX_K 200
#define M3D 10  // Number of basis functions for quadratic 3D polynomial
#define BLOCK_SIZE 256
#define MAX_CELLS_PER_BIN 50  // Max active cells per grid bin

// Basis function exponents for degree <= 2
__constant__ int d_exp3[M3D][3] = {
    {0, 0, 0},  // 1
    {1, 0, 0},  // x
    {0, 1, 0},  // y
    {0, 0, 1},  // z
    {2, 0, 0},  // x^2
    {1, 1, 0},  // xy
    {1, 0, 1},  // xz
    {0, 2, 0},  // y^2
    {0, 1, 1},  // yz
    {0, 0, 2}   // z^2
};

// Device function: phi_exact
__device__ float phi_exact_device(float x, float y, float z) {
    float r = sqrtf((x - 0.53f) * (x - 0.53f) + y * y + (z - 0.48f) * (z - 0.48f));
    return expf(x - 0.5f + 2.0f * (y - 0.3f) + 3.0f * (z - 0.6f));
}

// Device function: Gaussian weight
__device__ float gaussian_weight_device(float r, float tau) {
    float ratio = r / tau;
    return expf(-ratio * ratio);
}

// Spatial hash grid neighbor search using pre-built grid structure
// Grid stores indices of active cells in each spatial bin
__device__ int find_neighbors_with_grid(
    const float* active_coords,
    const int* grid_data,      // [Nx*Ny*Nz*MAX_CELLS_PER_BIN] flattened grid
    const int* grid_counts,    // [Nx*Ny*Nz] number of cells in each bin
    int Nx, int Ny, int Nz,
    int ghost_i, int ghost_j, int ghost_k,
    int box_radius,
    int max_neighbors,
    int* neighbors, float* distances,
    float x0, float y0, float z0
) {
    int count = 0;
    
    // Iterate over box around ghost cell
    for (int di = -box_radius; di <= box_radius && count < max_neighbors; di++) {
        int i = ghost_i + di;
        if (i < 0 || i >= Nx) continue;
        
        for (int dj = -box_radius; dj <= box_radius && count < max_neighbors; dj++) {
            int j = ghost_j + dj;
            if (j < 0 || j >= Ny) continue;
            
            for (int dk = -box_radius; dk <= box_radius && count < max_neighbors; dk++) {
                int k = ghost_k + dk;
                if (k < 0 || k >= Nz) continue;
                
                // Get grid bin index
                int bin_idx = i * Ny * Nz + j * Nz + k;
                int bin_count = grid_counts[bin_idx];
                
                // Check all active cells in this bin
                for (int b = 0; b < bin_count && count < max_neighbors; b++) {
                    int active_idx = grid_data[bin_idx * MAX_CELLS_PER_BIN + b];
                    
                    float xn = active_coords[active_idx * 3 + 0];
                    float yn = active_coords[active_idx * 3 + 1];
                    float zn = active_coords[active_idx * 3 + 2];
                    
                    float dist = sqrtf((xn - x0) * (xn - x0) + 
                                      (yn - y0) * (yn - y0) + 
                                      (zn - z0) * (zn - z0));
                    
                    neighbors[count] = active_idx;
                    distances[count] = dist;
                    count++;
                }
            }
        }
    }
    
    return count;
}

// Fallback: Box-based neighbor search without grid (slower)
__device__ int find_neighbors_in_box(
    const float* active_coords, int N,
    float x0, float y0, float z0,
    float dx, float box_radius,
    int max_neighbors,
    int* neighbors, float* distances
) {
    int count = 0;
    float box_dist = box_radius * dx;
    
    // Find all neighbors within the box
    // Early exit if we have enough neighbors and box is full
    for (int n = 0; n < N; n++) {
        if (count >= max_neighbors) break;  // Early exit when full
        
        float xn = active_coords[n * 3 + 0];
        float yn = active_coords[n * 3 + 1];
        float zn = active_coords[n * 3 + 2];
        
        // Quick rejection test: check each dimension separately (early exit)
        float diff_x = fabsf(xn - x0);
        if (diff_x > box_dist) continue;
        
        float diff_y = fabsf(yn - y0);
        if (diff_y > box_dist) continue;
        
        float diff_z = fabsf(zn - z0);
        if (diff_z > box_dist) continue;
        
        // Point is within box
        float dist = sqrtf((xn - x0) * (xn - x0) + 
                          (yn - y0) * (yn - y0) + 
                          (zn - z0) * (zn - z0));
        neighbors[count] = n;
        distances[count] = dist;
        count++;
    }
    
    return count;
}

// Build basis matrix and compute weights
__device__ void build_weighted_system(
    const int* neighbors, int K,
    const float* active_coords,
    const float* active_vals,
    float x0, float y0, float z0,
    float dx, float dy, float dz,
    float tau,
    float* M, float* w, float* rhs
) {
    for (int k = 0; k < K; k++) {
        int idx = neighbors[k];
        float xk = active_coords[idx * 3 + 0];
        float yk = active_coords[idx * 3 + 1];
        float zk = active_coords[idx * 3 + 2];
        
        // Normalized coordinates
        float X = (xk - x0) / dx;
        float Y = (yk - y0) / dy;
        float Z = (zk - z0) / dz;
        
        // Build basis for this point
        for (int m = 0; m < M3D; m++) {
            int a = d_exp3[m][0];
            int b = d_exp3[m][1];
            int c = d_exp3[m][2];
            
            float val = 1.0f;
            for (int i = 0; i < a; i++) val *= X;
            for (int i = 0; i < b; i++) val *= Y;
            for (int i = 0; i < c; i++) val *= Z;
            
            M[k * M3D + m] = val;
        }
        
        // Weight
        float dist = sqrtf((xk - x0) * (xk - x0) + 
                          (yk - y0) * (yk - y0) + 
                          (zk - z0) * (zk - z0));
        w[k] = gaussian_weight_device(dist, tau);
        
        // RHS
        rhs[k] = active_vals[idx];
    }
}

// Simple QR decomposition using Gram-Schmidt (for small matrices)
__device__ void qr_decomposition_gs(float* M, int K, int m, float* Q, float* R) {
    // Initialize Q with M
    for (int i = 0; i < K * m; i++) {
        Q[i] = M[i];
    }
    
    // Initialize R to zero
    for (int i = 0; i < m * m; i++) {
        R[i] = 0.0f;
    }
    
    // Modified Gram-Schmidt
    for (int j = 0; j < m; j++) {
        // Compute norm of column j
        float norm = 0.0f;
        for (int i = 0; i < K; i++) {
            float val = Q[i * m + j];
            norm += val * val;
        }
        norm = sqrtf(norm);
        R[j * m + j] = norm;
        
        if (norm > 1e-10f) {
            // Normalize column j
            for (int i = 0; i < K; i++) {
                Q[i * m + j] /= norm;
            }
            
            // Orthogonalize remaining columns
            for (int k = j + 1; k < m; k++) {
                float dot = 0.0f;
                for (int i = 0; i < K; i++) {
                    dot += Q[i * m + j] * Q[i * m + k];
                }
                R[j * m + k] = dot;
                
                for (int i = 0; i < K; i++) {
                    Q[i * m + k] -= dot * Q[i * m + j];
                }
            }
        }
    }
}

// Solve upper triangular system Rx = b
__device__ void solve_upper_triangular(const float* R, const float* b, int m, float* x) {
    for (int i = m - 1; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < m; j++) {
            sum -= R[i * m + j] * x[j];
        }
        x[i] = (fabsf(R[i * m + i]) > 1e-10f) ? (sum / R[i * m + i]) : 0.0f;
    }
}

// LSQ solve with weighted normal equations + Tikhonov regularization
// Solves: (M^T * W * M + lambda*I) * a = M^T * W * rhs
// where W = diag(w^2) for consistency with QR formulation
__device__ float solve_lsq_reconstruction(
    float* M, const float* w, const float* rhs, int K,
    float reg_lambda
) {
    const int m = M3D;
    
    // Form normal equations: A = M^T * diag(w^2) * M
    float A[M3D * M3D];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += M[k * m + i] * w[k] * w[k] * M[k * m + j];
            }
            A[i * m + j] = sum;
        }
    }
    
    // Form RHS: b = M^T * diag(w^2) * rhs
    float b[M3D];
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += M[k * m + i] * w[k] * w[k] * rhs[k];
        }
        b[i] = sum;
    }
    
    // Add regularization: A += lambda * diag_max * I
    if (reg_lambda > 0.0f) {
        float diag_max = 0.0f;
        for (int i = 0; i < m; i++) {
            float val = fabsf(A[i * m + i]);
            if (val > diag_max) diag_max = val;
        }
        float lam = reg_lambda * fmaxf(diag_max, 1.0f);
        for (int i = 0; i < m; i++) {
            A[i * m + i] += lam;
        }
    }
    
    // Solve using Cholesky decomposition (A is symmetric positive definite)
    // A = L * L^T
    float L[M3D * M3D];
    for (int i = 0; i < m * m; i++) L[i] = 0.0f;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            if (i == j) {
                for (int k = 0; k < j; k++) {
                    sum += L[i * m + k] * L[i * m + k];
                }
                float val = A[i * m + i] - sum;
                L[i * m + j] = (val > 0.0f) ? sqrtf(val) : 1e-10f;
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i * m + k] * L[j * m + k];
                }
                if (fabsf(L[j * m + j]) > 1e-10f) {
                    L[i * m + j] = (A[i * m + j] - sum) / L[j * m + j];
                }
            }
        }
    }
    
    // Solve L * y = b
    float y[M3D];
    for (int i = 0; i < m; i++) {
        float sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * m + j] * y[j];
        }
        y[i] = (fabsf(L[i * m + i]) > 1e-10f) ? (sum / L[i * m + i]) : 0.0f;
    }
    
    // Solve L^T * a = y
    float a[M3D];
    for (int i = m - 1; i >= 0; i--) {
        float sum = y[i];
        for (int j = i + 1; j < m; j++) {
            sum -= L[j * m + i] * a[j];
        }
        a[i] = (fabsf(L[i * m + i]) > 1e-10f) ? (sum / L[i * m + i]) : 0.0f;
    }
    
    // Return constant term
    return a[0];
}

// Kernel: First pass reconstruction with spatial hash grid
__global__ void reconstruct_pass1_kernel(
    const float* active_coords,  // [N, 3]
    const float* active_vals,    // [N]
    const int* ghost_indices,    // [M, 3]
    const float* xs,
    const float* ys,
    const float* zs,
    const int* grid_data,        // [Nx*Ny*Nz*MAX_CELLS_PER_BIN]
    const int* grid_counts,      // [Nx*Ny*Nz]
    int N, int M,
    int Nx, int Ny, int Nz,
    float dx, float tau, float reg_lambda,
    int box_radius,              // Box radius in lattice sites
    int max_neighbors,
    float phi_rel_clip,
    float* reconstructed,  // [M]
    float* exact_vals,     // [M]
    float* rel_err         // [M]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    
    // Get ghost cell position
    int i = ghost_indices[idx * 3 + 0];
    int j = ghost_indices[idx * 3 + 1];
    int k = ghost_indices[idx * 3 + 2];
    float x0 = xs[i];
    float y0 = ys[j];
    float z0 = zs[k];
    
    // Find neighbors using spatial grid
    int neighbors[MAX_K];
    float distances[MAX_K];
    int K = find_neighbors_with_grid(active_coords, grid_data, grid_counts,
                                      Nx, Ny, Nz, i, j, k, box_radius,
                                      max_neighbors, neighbors, distances,
                                      x0, y0, z0);
    
    // Need at least M3D points for full rank system
    if (K < M3D) {
        reconstructed[idx] = 0.0f;
        exact_vals[idx] = phi_exact_device(x0, y0, z0);
        rel_err[idx] = 1.0f;
        return;
    }
    
    // Build weighted LSQ system
    float M_mat[MAX_K * M3D];
    float w[MAX_K];
    float rhs[MAX_K];
    build_weighted_system(neighbors, K, active_coords, active_vals,
                         x0, y0, z0, dx, dx, dx, tau,
                         M_mat, w, rhs);
    
    // Solve LSQ
    float recon = solve_lsq_reconstruction(M_mat, w, rhs, K, reg_lambda);
    reconstructed[idx] = recon;
    
    // Compute exact value and error
    float exact = phi_exact_device(x0, y0, z0);
    exact_vals[idx] = exact;
    
    float denom = fmaxf(fabsf(exact), phi_rel_clip);
    rel_err[idx] = fabsf(recon - exact) / denom;
}

// Kernel: Build spatial hash grid from active cell coordinates
__global__ void build_spatial_grid_kernel(
    const float* active_coords,
    const float* xs,
    const float* ys,
    const float* zs,
    int N_active,
    int Nx, int Ny, int Nz,
    int* grid_data,      // [Nx*Ny*Nz*MAX_CELLS_PER_BIN]
    int* grid_counts     // [Nx*Ny*Nz]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_active) return;
    
    float x = active_coords[idx * 3 + 0];
    float y = active_coords[idx * 3 + 1];
    float z = active_coords[idx * 3 + 2];
    
    // Find grid indices (closest grid point)
    int best_i = 0, best_j = 0, best_k = 0;
    float min_dist = 1e30f;
    
    for (int i = 0; i < Nx; i++) {
        float dx = x - xs[i];
        if (fabsf(dx) > min_dist) continue;
        
        for (int j = 0; j < Ny; j++) {
            float dy = y - ys[j];
            float dxy = dx*dx + dy*dy;
            if (dxy > min_dist * min_dist) continue;
            
            for (int k = 0; k < Nz; k++) {
                float dz = z - zs[k];
                float dist_sq = dxy + dz*dz;
                
                if (dist_sq < min_dist * min_dist) {
                    min_dist = sqrtf(dist_sq);
                    best_i = i;
                    best_j = j;
                    best_k = k;
                }
            }
        }
    }
    
    // Add to grid bin using atomic operation
    int bin_idx = best_i * Ny * Nz + best_j * Nz + best_k;
    int pos = atomicAdd(&grid_counts[bin_idx], 1);
    
    if (pos < MAX_CELLS_PER_BIN) {
        grid_data[bin_idx * MAX_CELLS_PER_BIN + pos] = idx;
    }
}

// Kernel: Identify bad cells
__global__ void identify_bad_cells_kernel(
    const float* rel_err,
    int M,
    float threshold,
    int* bad_mask,
    int* bad_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    
    if (rel_err[idx] > threshold) {
        bad_mask[idx] = 1;
        atomicAdd(bad_count, 1);
    } else {
        bad_mask[idx] = 0;
    }
}

// Kernel: Second pass reconstruction (only for bad cells)
// Kernel: Second pass reconstruction (only for bad cells) with spatial hash grid
__global__ void reconstruct_pass2_kernel(
    const float* active_coords,
    const float* active_vals,
    const int* ghost_indices,
    const float* xs,
    const float* ys,
    const float* zs,
    const int* grid_data,
    const int* grid_counts,
    const int* bad_mask,
    int N, int M,
    int Nx, int Ny, int Nz,
    float dx, float tau, float reg_lambda,
    int box_radius,            // Box radius in lattice sites (larger for pass 2)
    int max_neighbors,
    float phi_rel_clip,
    float* reconstructed,
    const float* exact_vals,
    float* rel_err
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M || bad_mask[idx] == 0) return;
    
    // Get ghost cell position
    int i = ghost_indices[idx * 3 + 0];
    int j = ghost_indices[idx * 3 + 1];
    int k = ghost_indices[idx * 3 + 2];
    float x0 = xs[i];
    float y0 = ys[j];
    float z0 = zs[k];
    
    // Find neighbors using spatial grid with larger radius
    int neighbors[MAX_K];
    float distances[MAX_K];
    int K = find_neighbors_with_grid(active_coords, grid_data, grid_counts,
                                      Nx, Ny, Nz, i, j, k, box_radius,
                                      max_neighbors, neighbors, distances,
                                      x0, y0, z0);
    
    // Need at least M3D points
    if (K < M3D) {
        return;  // Keep previous reconstruction
    }
    
    // Build weighted LSQ system
    float M_mat[MAX_K * M3D];
    float w[MAX_K];
    float rhs[MAX_K];
    build_weighted_system(neighbors, K, active_coords, active_vals,
                         x0, y0, z0, dx, dx, dx, tau,
                         M_mat, w, rhs);
    
    // Solve LSQ
    float recon = solve_lsq_reconstruction(M_mat, w, rhs, K, reg_lambda);
    reconstructed[idx] = recon;
    
    // Update error
    float exact = exact_vals[idx];
    float denom = fmaxf(fabsf(exact), phi_rel_clip);
    rel_err[idx] = fabsf(recon - exact) / denom;
}


// Host function: Two-pass reconstruction
extern "C" {

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
    float tau1, float tau2,
    float reg_lambda,
    int max_neighbors,
    int max_neighbors_bad,
    float relerr_threshold,
    float phi_rel_clip,
    float* h_reconstructed,
    float* h_exact_vals,
    float* h_rel_err,
    int* h_num_bad
) {
    // Allocate device memory
    float *d_active_coords, *d_active_vals;
    int *d_ghost_indices;
    float *d_xs, *d_ys, *d_zs;
    float *d_reconstructed, *d_exact_vals, *d_rel_err;
    int *d_bad_mask, *d_bad_count;
    
    cudaMalloc(&d_active_coords, N_active * 3 * sizeof(float));
    cudaMalloc(&d_active_vals, N_active * sizeof(float));
    cudaMalloc(&d_ghost_indices, N_ghost * 3 * sizeof(int));
    cudaMalloc(&d_xs, Nx * sizeof(float));
    cudaMalloc(&d_ys, Ny * sizeof(float));
    cudaMalloc(&d_zs, Nz * sizeof(float));
    cudaMalloc(&d_reconstructed, N_ghost * sizeof(float));
    cudaMalloc(&d_exact_vals, N_ghost * sizeof(float));
    cudaMalloc(&d_rel_err, N_ghost * sizeof(float));
    cudaMalloc(&d_bad_mask, N_ghost * sizeof(int));
    cudaMalloc(&d_bad_count, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_active_coords, h_active_coords, N_active * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_vals, h_active_vals, N_active * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ghost_indices, h_ghost_indices, N_ghost * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, h_xs, Nx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, h_ys, Ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, h_zs, Nz * sizeof(float), cudaMemcpyHostToDevice);
    
    int zero = 0;
    cudaMemcpy(d_bad_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate and build spatial hash grid
    int grid_size = Nx * Ny * Nz;
    int *d_grid_data, *d_grid_counts;
    cudaMalloc(&d_grid_data, grid_size * MAX_CELLS_PER_BIN * sizeof(int));
    cudaMalloc(&d_grid_counts, grid_size * sizeof(int));
    cudaMemset(d_grid_counts, 0, grid_size * sizeof(int));
    
    // Build the spatial grid
    int grid_blocks = (N_active + 255) / 256;
    build_spatial_grid_kernel<<<grid_blocks, 256>>>(
        d_active_coords, d_xs, d_ys, d_zs,
        N_active, Nx, Ny, Nz,
        d_grid_data, d_grid_counts
    );
    cudaDeviceSynchronize();
    
    // Launch first pass kernel
    int num_blocks = (N_ghost + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Box radius: for max_neighbors ~120, use radius ~3 (gives 7x7x7=343 cells max)
    // For max_neighbors_bad ~200, use radius ~4 (gives 9x9x9=729 cells max)
    float box_radius1 = 3.0f;  // First pass: 3 lattice sites = ~7x7x7 box
    float box_radius2 = 4.0f;  // Second pass: 4 lattice sites = ~9x9x9 box
    
    reconstruct_pass1_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_active_coords, d_active_vals, d_ghost_indices,
        d_xs, d_ys, d_zs,
        d_grid_data, d_grid_counts,
        N_active, N_ghost,
        Nx, Ny, Nz,
        dx, tau1, reg_lambda,
        3, max_neighbors,
        phi_rel_clip,
        d_reconstructed, d_exact_vals, d_rel_err
    );
    cudaDeviceSynchronize();
    
    // Identify bad cells
    identify_bad_cells_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_rel_err, N_ghost, relerr_threshold, d_bad_mask, d_bad_count
    );
    cudaDeviceSynchronize();
    
    // Get bad cell count
    cudaMemcpy(h_num_bad, d_bad_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Launch second pass kernel if needed
    if (*h_num_bad > 0) {
        reconstruct_pass2_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_active_coords, d_active_vals, d_ghost_indices,
            d_xs, d_ys, d_zs,
            d_grid_data, d_grid_counts, d_bad_mask,
            N_active, N_ghost,
            Nx, Ny, Nz,
            dx, tau2, reg_lambda,
            4, max_neighbors_bad,
            phi_rel_clip,
            d_reconstructed, d_exact_vals, d_rel_err
        );
        cudaDeviceSynchronize();
    }
    
    // Copy results back to host
    cudaMemcpy(h_reconstructed, d_reconstructed, N_ghost * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_exact_vals, d_exact_vals, N_ghost * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rel_err, d_rel_err, N_ghost * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_active_coords);
    cudaFree(d_active_vals);
    cudaFree(d_ghost_indices);
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_zs);
    cudaFree(d_reconstructed);
    cudaFree(d_exact_vals);
    cudaFree(d_rel_err);
    cudaFree(d_bad_mask);
    cudaFree(d_bad_count);
    cudaFree(d_grid_data);
    cudaFree(d_grid_counts);
}

} // extern "C"
