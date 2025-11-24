/**
 * Optimized example: Minimal host-device copies for repeated reconstructions
 * 
 * This demonstrates the most efficient usage pattern:
 * 1. Allocate memory once (cuda_reconstruction_init)
 * 2. Copy static domain data once (cuda_reconstruction_set_domain)
 * 3. Call reconstruction many times, only copying active_vals each time
 * 4. Free memory once when done
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include "poly_cuda.h"

using namespace std::chrono;

// Exact solution for testing
double phi_exact(double x, double y, double z) {
    return exp(x - 0.5 + 2.0 * (y - 0.3) + 3.0 * (z - 0.6));
}

int main() {
    int N = 160;
    double L = 1.6;
    double dx = L / N;
    double R = 0.8;
    
    std::cout << "Optimized Reconstruction Example (N=" << N << ")\n";
    std::cout << "================================================\n\n";
    
    // Build grid coordinates
    std::vector<double> xs(N), ys(N), zs(N);
    for (int i = 0; i < N; i++) {
        xs[i] = -0.8 + i * dx;
        ys[i] = -0.8 + i * dx;
        zs[i] = -0.8 + i * dx;
    }
    
    // Identify active and ghost cells
    std::vector<double> active_coords;
    std::vector<double> active_vals;
    std::vector<int> ghost_indices;
    std::vector<bool> is_active(N * N * N, false);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double x = xs[i], y = ys[j], z = zs[k];
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
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                if (is_active[i * N * N + j * N + k]) continue;
                
                bool has_active_neighbor = false;
                for (int di = -1; di <= 1 && !has_active_neighbor; di++) {
                    for (int dj = -1; dj <= 1 && !has_active_neighbor; dj++) {
                        for (int dk = -1; dk <= 1 && !has_active_neighbor; dk++) {
                            int ni = i + di, nj = j + dj, nk = k + dk;
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
    
    std::cout << "Active cells: " << num_active << "\n";
    std::cout << "Ghost cells: " << num_ghost << "\n\n";
    
    // ========================================================================
    // STEP 1: Allocate device memory ONCE
    // ========================================================================
    std::cout << "Step 1: Allocating device memory...\n";
    auto t_init = high_resolution_clock::now();
    
    PolyReconstructionContext ctx = cuda_reconstruction_init(
        num_active, num_ghost, N, N, N
    );
    
    auto t_init_end = high_resolution_clock::now();
    double init_time = duration_cast<milliseconds>(t_init_end - t_init).count();
    std::cout << "  Time: " << init_time << " ms\n\n";
    
    // ========================================================================
    // STEP 2: Copy static domain data ONCE (grid coords, indices)
    // ========================================================================
    std::cout << "Step 2: Copying static domain data to device...\n";
    auto t_domain = high_resolution_clock::now();
    
    cuda_reconstruction_set_domain(
        ctx,
        active_coords.data(),
        ghost_indices.data(),
        xs.data(), ys.data(), zs.data()
    );
    
    auto t_domain_end = high_resolution_clock::now();
    double domain_time = duration_cast<milliseconds>(t_domain_end - t_domain).count();
    std::cout << "  Time: " << domain_time << " ms\n\n";
    
    // ========================================================================
    // STEP 3: Run reconstruction MANY TIMES
    //         Only copying active_vals each time (minimal overhead!)
    // ========================================================================
    std::cout << "Step 3: Running reconstruction 10 times...\n";
    
    int num_iterations = 10;
    std::vector<double> reconstructed(num_ghost);
    std::vector<double> exact_vals(num_ghost);
    std::vector<double> rel_err(num_ghost);
    int num_bad = 0;
    
    auto t_recon_start = high_resolution_clock::now();
    
    for (int iter = 0; iter < num_iterations; iter++) {
        // In a real application, active_vals would change each iteration
        // Here we just use the same values for demonstration
        
        cuda_reconstruction_execute(
            ctx,
            active_vals.data(),  // ONLY thing copied to device!
            dx,
            2.0 * dx,   // tau1
            4.0 * dx,   // tau2
            1e-7,       // reg_lambda
            120,        // max_neighbors
            200,        // max_neighbors_bad
            0.2,        // relerr_threshold
            1e-3,       // phi_rel_clip
            reconstructed.data(),
            exact_vals.data(),
            rel_err.data(),
            &num_bad
        );
    }
    
    auto t_recon_end = high_resolution_clock::now();
    double recon_time = duration_cast<milliseconds>(t_recon_end - t_recon_start).count();
    double avg_time = recon_time / num_iterations;
    
    std::cout << "  Total time: " << recon_time << " ms\n";
    std::cout << "  Avg per iteration: " << avg_time << " ms\n";
    std::cout << "  Throughput: " << (1000.0 / avg_time) << " reconstructions/sec\n\n";
    
    // Compute statistics
    double mean_err = 0.0, max_err = 0.0;
    for (int i = 0; i < num_ghost; i++) {
        mean_err += rel_err[i];
        max_err = std::max(max_err, rel_err[i]);
    }
    mean_err /= num_ghost;
    
    std::cout << "Accuracy:\n";
    std::cout << "  Mean rel error: " << mean_err << "\n";
    std::cout << "  Max rel error: " << max_err << "\n";
    std::cout << "  Bad cells: " << num_bad << "\n\n";
    
    // ========================================================================
    // STEP 4: Free device memory ONCE when done
    // ========================================================================
    std::cout << "Step 4: Freeing device memory...\n";
    cuda_reconstruction_free(ctx);
    
    std::cout << "\nOptimization Summary:\n";
    std::cout << "=====================\n";
    std::cout << "One-time setup: " << (init_time + domain_time) << " ms\n";
    std::cout << "Per-iteration cost: " << avg_time << " ms\n";
    std::cout << "  (Only " << (num_active * sizeof(double) / 1e6) << " MB copied per iteration)\n";
    
    return 0;
}
