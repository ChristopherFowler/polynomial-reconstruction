/**
 * Standalone C++ version of profile_poly3d.py
 * Uses CUDA code for polynomial reconstruction
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "poly_cuda.h"

using namespace std;
using namespace std::chrono;

// Exact solution function
double phi_exact(double x, double y, double z) {
    double r = sqrt((x - 0.53) * (x - 0.53) + y * y + (z - 0.48) * (z - 0.48));
    return exp(x - 0.5 + 2.0 * (y - 0.3) + 3.0 * (z - 0.6));
}

// Helper struct for timing results
struct ProfileResult {
    int N;
    double total_time;
    double classify_time;
    double cuda_time;
    int num_active;
    int num_ghost;
};

// Cell classification using sphere test and neighbor expansion
void classify_cells(
    const vector<double>& xs, const vector<double>& ys, const vector<double>& zs,
    double R,
    vector<double>& active_coords,
    vector<int>& ghost_indices,
    int& num_active, int& num_ghost
) {
    int Nx = xs.size();
    int Ny = ys.size();
    int Nz = zs.size();
    
    // Create 3D mask for active and ghost cells
    vector<vector<vector<bool>>> active_mask(Nx, 
        vector<vector<bool>>(Ny, vector<bool>(Nz, false)));
    
    // Step 1: Mark active cells (inside sphere)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                double x = xs[i];
                double y = ys[j];
                double z = zs[k];
                double r = sqrt(x*x + y*y + z*z);
                if (r < R) {
                    active_mask[i][j][k] = true;
                }
            }
        }
    }
    
    // Step 2: Create dilated mask (active + 1 cell in all directions)
    vector<vector<vector<bool>>> dilated_mask(Nx, 
        vector<vector<bool>>(Ny, vector<bool>(Nz, false)));
    
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                if (active_mask[i][j][k]) {
                    // Dilate by 1 in all directions
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            for (int dk = -1; dk <= 1; dk++) {
                                int ni = i + di;
                                int nj = j + dj;
                                int nk = k + dk;
                                if (ni >= 0 && ni < Nx && 
                                    nj >= 0 && nj < Ny && 
                                    nk >= 0 && nk < Nz) {
                                    dilated_mask[ni][nj][nk] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Step 3: Extract active coordinates and ghost indices
    active_coords.clear();
    ghost_indices.clear();
    
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                if (active_mask[i][j][k]) {
                    // Active cell - store coordinates
                    active_coords.push_back(xs[i]);
                    active_coords.push_back(ys[j]);
                    active_coords.push_back(zs[k]);
                } else if (dilated_mask[i][j][k]) {
                    // Ghost cell - store indices
                    ghost_indices.push_back(i);
                    ghost_indices.push_back(j);
                    ghost_indices.push_back(k);
                }
            }
        }
    }
    
    num_active = active_coords.size() / 3;
    num_ghost = ghost_indices.size() / 3;
}

ProfileResult profile_run(int N) {
    cout << "\n" << string(60, '=') << "\n";
    cout << "Profiling N=" << N << "\n";
    cout << string(60, '=') << "\n";
    
    auto t_start = high_resolution_clock::now();
    
    // Grid setup
    double L = 1.6;
    double dx = L / N;
    vector<double> xs(N), ys(N), zs(N);
    
    for (int i = 0; i < N; i++) {
        xs[i] = -0.8 + i * dx;
        ys[i] = -0.8 + i * dx;
        zs[i] = -0.8 + i * dx;
    }
    double R = 0.8;
    
    auto t_grid = high_resolution_clock::now();
    auto dt_grid = duration_cast<microseconds>(t_grid - t_start).count() / 1000.0;
    cout << "Grid setup: " << fixed << setprecision(2) << dt_grid << " ms\n";
    
    // Classify active vs ghost
    vector<double> active_coords;
    vector<int> ghost_indices;
    int num_active, num_ghost;
    
    classify_cells(xs, ys, zs, R, active_coords, ghost_indices, num_active, num_ghost);
    
    auto t_classify = high_resolution_clock::now();
    auto dt_classify = duration_cast<microseconds>(t_classify - t_grid).count() / 1000.0;
    cout << "Cell classification: " << dt_classify << " ms\n";
    cout << "  Active cells: " << num_active << "\n";
    cout << "  Ghost cells: " << num_ghost << "\n";
    
    // Compute active values
    vector<double> active_vals(num_active);
    for (int i = 0; i < num_active; i++) {
        double x = active_coords[i * 3 + 0];
        double y = active_coords[i * 3 + 1];
        double z = active_coords[i * 3 + 2];
        active_vals[i] = phi_exact(x, y, z);
    }
    
    auto t_vals = high_resolution_clock::now();
    auto dt_vals = duration_cast<microseconds>(t_vals - t_classify).count() / 1000.0;
    cout << "Computing active values: " << dt_vals << " ms\n";
    
    // CUDA reconstruction using optimized API
    double tau1 = 2.0 * dx;
    double tau2 = 4.0 * dx;
    
    vector<double> reconstructed(num_ghost);
    vector<double> exact_vals(num_ghost);
    vector<double> rel_err(num_ghost);
    int num_bad = 0;
    
    // Initialize context (allocate device memory)
    auto t_init = high_resolution_clock::now();
    PolyReconstructionContext ctx = cuda_reconstruction_init(num_active, num_ghost, N, N, N);
    auto t_init_end = high_resolution_clock::now();
    auto dt_init = duration_cast<microseconds>(t_init_end - t_init).count() / 1000.0;
    
    // Set domain (copy static data)
    auto t_domain = high_resolution_clock::now();
    cuda_reconstruction_set_domain(ctx, active_coords.data(), ghost_indices.data(),
                                    xs.data(), ys.data(), zs.data());
    auto t_domain_end = high_resolution_clock::now();
    auto dt_domain = duration_cast<microseconds>(t_domain_end - t_domain).count() / 1000.0;
    
    // Execute reconstruction (only copies active_vals)
    auto t_recon = high_resolution_clock::now();
    cuda_reconstruction_execute(
        ctx,
        active_vals.data(),
        dx,
        tau1, tau2,
        1e-7,
        120,    // max_neighbors
        200,    // max_neighbors_bad
        0.2,    // relerr_threshold
        1e-3,   // phi_rel_clip
        reconstructed.data(),
        exact_vals.data(),
        rel_err.data(),
        &num_bad
    );
    auto t_recon_end = high_resolution_clock::now();
    auto dt_recon = duration_cast<microseconds>(t_recon_end - t_recon).count() / 1000.0;
    
    // Free context
    cuda_reconstruction_free(ctx);
    
    auto t_cuda = high_resolution_clock::now();
    auto dt_cuda = duration_cast<microseconds>(t_cuda - t_vals).count() / 1000.0;
    cout << "CUDA reconstruction breakdown:\n";
    cout << "  Memory allocation: " << dt_init << " ms\n";
    cout << "  Domain setup: " << dt_domain << " ms\n";
    cout << "  Reconstruction: " << dt_recon << " ms\n";
    cout << "  Total CUDA: " << dt_cuda << " ms\n";
    
    // Statistics
    double mean_rel_err = 0.0;
    double max_rel_err = 0.0;
    for (int i = 0; i < num_ghost; i++) {
        mean_rel_err += rel_err[i];
        max_rel_err = max(max_rel_err, rel_err[i]);
    }
    mean_rel_err /= num_ghost;
    
    auto t_stats = high_resolution_clock::now();
    auto dt_stats = duration_cast<microseconds>(t_stats - t_cuda).count() / 1000.0;
    cout << "Statistics: " << dt_stats << " ms\n";
    
    auto dt_total = duration_cast<microseconds>(t_stats - t_start).count() / 1000.0;
    cout << "\nTotal time: " << dt_total << " ms\n";
    cout << "  Mean rel error: " << scientific << setprecision(3) << mean_rel_err << "\n";
    cout << "  Max rel error: " << max_rel_err << "\n";
    cout << "  Bad cells: " << num_bad << "\n";
    
    ProfileResult result;
    result.N = N;
    result.total_time = dt_total / 1000.0;  // Convert to seconds
    result.classify_time = dt_classify / 1000.0;
    result.cuda_time = dt_cuda / 1000.0;
    result.num_active = num_active;
    result.num_ghost = num_ghost;
    
    return result;
}

int main() {
    vector<int> N_values = {10, 20, 40, 80, 160};
    vector<ProfileResult> results;
    
    for (int N : N_values) {
        ProfileResult result = profile_run(N);
        results.push_back(result);
    }
    
    // Summary table
    cout << "\n" << string(60, '=') << "\n";
    cout << "SUMMARY\n";
    cout << string(60, '=') << "\n";
    cout << setw(5) << "N" 
         << setw(8) << "Active" 
         << setw(8) << "Ghost" 
         << setw(10) << "Classify" 
         << setw(10) << "CUDA" 
         << setw(10) << "Total" << "\n";
    cout << setw(5) << "" 
         << setw(8) << "cells" 
         << setw(8) << "cells" 
         << setw(10) << "(ms)" 
         << setw(10) << "(ms)" 
         << setw(10) << "(ms)" << "\n";
    cout << string(60, '-') << "\n";
    
    cout << fixed << setprecision(2);
    for (const auto& r : results) {
        cout << setw(5) << r.N
             << setw(8) << r.num_active
             << setw(8) << r.num_ghost
             << setw(10) << r.classify_time * 1000
             << setw(10) << r.cuda_time * 1000
             << setw(10) << r.total_time * 1000 << "\n";
    }
    
    // Bottleneck analysis
    cout << "\nBottleneck Analysis:\n";
    for (const auto& r : results) {
        double classify_pct = 100.0 * r.classify_time / r.total_time;
        double cuda_pct = 100.0 * r.cuda_time / r.total_time;
        cout << "  N=" << setw(2) << r.N 
             << ": Classification " << setprecision(1) << classify_pct 
             << "%, CUDA " << cuda_pct << "%\n";
    }
    
    return 0;
}
