/**
 * Minimal example: Using CUDA reconstruction with plain C arrays (POD types)
 * 
 * KEY POINT: The CUDA function accepts raw pointers (double*, int*)
 * You can pass:
 *   - C-style arrays: double arr[100]
 *   - Dynamically allocated: double* arr = new double[100]
 *   - std::vector data: vec.data()
 * 
 * No std::vector required - works with plain old data (POD) types!
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include "poly_cuda.h"

using namespace std::chrono;

// Simple exact solution for testing
double phi_exact(double x, double y, double z) {
    return exp(x - 0.5 + 2.0 * (y - 0.3) + 3.0 * (z - 0.6));
}

int main() {
    // Setup: Large 160x160x160 grid
    int N = 160;
    double L = 1.6;
    double dx = L / N;  // Grid spacing
    double R = 0.8;  // Sphere radius
    
    std::cout << "Setting up " << N << "^3 grid...\n";
    std::cout << "Setting up " << N << "^3 grid...\n";
    
    auto t_start = high_resolution_clock::now();
    
    // Grid coordinates (dynamically allocated)
    double* xs = new double[N];
    double* ys = new double[N];
    double* zs = new double[N];
    
    for (int i = 0; i < N; i++) {
        xs[i] = -0.8 + i * dx;
        ys[i] = -0.8 + i * dx;
        zs[i] = -0.8 + i * dx;
    }
    
    // Find active cells (inside sphere) and ghost cells (adjacent to active)
    std::vector<double> active_coords;
    std::vector<double> active_vals;
    std::vector<int> ghost_indices;
    
    // First pass: identify active cells
    std::vector<bool> is_active(N * N * N, false);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double x = xs[i];
                double y = ys[j];
                double z = zs[k];
                double r = sqrt(x*x + y*y + z*z);
                
                if (r < R) {
                    is_active[i * N * N + j * N + k] = true;
                    active_coords.push_back(x);
                    active_coords.push_back(y);
                    active_coords.push_back(z);
                    active_vals.push_back(phi_exact(x, y, z));
                }
            }
        }
    }
    
    // Second pass: identify ghost cells (adjacent to active but not active)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                if (is_active[i * N * N + j * N + k]) continue;
                
                // Check if any neighbor is active
                bool has_active_neighbor = false;
                for (int di = -1; di <= 1 && !has_active_neighbor; di++) {
                    for (int dj = -1; dj <= 1 && !has_active_neighbor; dj++) {
                        for (int dk = -1; dk <= 1 && !has_active_neighbor; dk++) {
                            int ni = i + di;
                            int nj = j + dj;
                            int nk = k + dk;
                            if (ni >= 0 && ni < N && nj >= 0 && nj < N && nk >= 0 && nk < N) {
                                if (is_active[ni * N * N + nj * N + nk]) {
                                    has_active_neighbor = true;
                                }
                            }
                        }
                    }
                }
                
                if (has_active_neighbor) {
                    ghost_indices.push_back(i);
                    ghost_indices.push_back(j);
                    ghost_indices.push_back(k);
                }
            }
        }
    }
    
    int num_active = active_vals.size();
    int num_ghost = ghost_indices.size() / 3;
    
    auto t_setup = high_resolution_clock::now();
    double setup_time = duration_cast<milliseconds>(t_setup - t_start).count();
    
    std::cout << "Grid setup complete: " << setup_time << " ms\n";
    std::cout << "Active cells: " << num_active << "\n";
    std::cout << "Ghost cells: " << num_ghost << "\n";
    
    // Output arrays (dynamically allocated)
    double* reconstructed = new double[num_ghost];
    double* exact_vals = new double[num_ghost];
    double* rel_err = new double[num_ghost];
    int num_bad = 0;
    
    // Run reconstruction multiple times
    int num_iterations = 10;  // Change this to 100 for production runs
    
    std::cout << "Starting CUDA reconstruction (" << num_iterations << " iterations)...\n";
    auto t_cuda_start = high_resolution_clock::now();
    
    for (int iter = 0; iter < num_iterations; iter++) {
        // Call CUDA function with plain C arrays (using .data() from vectors)
        cuda_two_pass_reconstruction(
            active_coords.data(),  // double* - from vector
            active_vals.data(),    // double* - from vector
            ghost_indices.data(),  // int* - from vector
            xs, ys, zs,           // double* - plain arrays
            num_active,
            num_ghost,
            N, N, N,
            dx,
            2.0 * dx,             // tau1
            4.0 * dx,             // tau2
            1e-7,                 // regularization
            120,                  // max_neighbors
            200,                  // max_neighbors_bad
            0.2,                  // relerr_threshold
            1e-3,                 // phi_rel_clip
            reconstructed,        // double* - output
            exact_vals,           // double* - output
            rel_err,              // double* - output
            &num_bad              // int* - output
        );
        
        if ((iter + 1) % 10 == 0) {
            std::cout << "  Completed iteration " << (iter + 1) << "/" << num_iterations << "\n";
        }
    }
    
    auto t_cuda_end = high_resolution_clock::now();
    double cuda_time = duration_cast<milliseconds>(t_cuda_end - t_cuda_start).count();
    double avg_time_per_iter = cuda_time / num_iterations;
    
    // Compute statistics
    double mean_rel_err = 0.0;
    double max_rel_err = 0.0;
    for (int i = 0; i < num_ghost; i++) {
        mean_rel_err += rel_err[i];
        if (rel_err[i] > max_rel_err) max_rel_err = rel_err[i];
    }
    mean_rel_err /= num_ghost;
    
    auto t_end = high_resolution_clock::now();
    double total_time = duration_cast<milliseconds>(t_end - t_start).count();
    
    // Print results
    std::cout << "\n";
    std::cout << "CUDA Reconstruction Results (N=" << N << ")\n";
    std::cout << "==========================================\n";
    std::cout << "Grid size: " << N << "^3 = " << (N*N*N) << " cells\n";
    std::cout << "Active cells: " << num_active << "\n";
    std::cout << "Ghost cells: " << num_ghost << "\n";
    std::cout << "\nTiming:\n";
    std::cout << "  Setup time: " << setup_time << " ms\n";
    std::cout << "  CUDA time (total): " << cuda_time << " ms\n";
    std::cout << "  Iterations: " << num_iterations << "\n";
    std::cout << "  Avg time per iteration: " << avg_time_per_iter << " ms\n";
    std::cout << "  Total time: " << total_time << " ms\n";
    std::cout << "\nAccuracy (from last iteration):\n";
    std::cout << "  Mean rel error: " << mean_rel_err << "\n";
    std::cout << "  Max rel error: " << max_rel_err << "\n";
    std::cout << "  Bad cells: " << num_bad << "\n";
    
    // Cleanup
    delete[] xs;
    delete[] ys;
    delete[] zs;
    delete[] reconstructed;
    delete[] exact_vals;
    delete[] rel_err;
    
    return 0;
}
