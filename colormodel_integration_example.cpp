/**
 * ColorModel Integration Example
 * 
 * This shows how to integrate polynomial reconstruction into ColorModel.cpp
 * to fill inactive sites in Phi_TwoHalo using the optimized API.
 * 
 * Key pattern:
 * 1. Initialize once during setup
 * 2. Execute many times per timestep (fast!)
 * 3. Cleanup once when done
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "poly_cuda.h"

// Simulated ColorModel data structures
class ColorModelExample {
public:
    // Domain dimensions
    int Nx_TwoHalo, Ny_TwoHalo, Nz_TwoHalo;
    int N_TwoHalo;
    double dx;
    int rank;
    
    // Device arrays (simulated with pointers)
    double* Phi_TwoHalo;
    double* Inactive_ID_TwoHalo;
    
    // Reconstruction context and cached data
    PolyReconstructionContext m_recon_ctx;
    bool m_recon_initialized;
    std::vector<double> m_active_coords;
    std::vector<int> m_ghost_indices;
    std::vector<double> m_xs_twohalo, m_ys_twohalo, m_zs_twohalo;
    int m_num_active_recon, m_num_ghost_recon;
    
    // Cached mask to identify active sites quickly
    std::vector<bool> m_is_active_site;
    
    ColorModelExample() : m_recon_initialized(false), rank(0) {}
    
    ~ColorModelExample() {
        CleanupReconstruction();
    }
    
    // ========================================================================
    // INITIALIZATION: Call once during ColorModel setup
    // ========================================================================
    void InitializeReconstruction() {
        if (rank == 0) {
            std::cout << "\n=== Initializing Polynomial Reconstruction ===\n";
        }
        
        // Build grid coordinates for TwoHalo domain
        m_xs_twohalo.resize(Nx_TwoHalo);
        m_ys_twohalo.resize(Ny_TwoHalo);
        m_zs_twohalo.resize(Nz_TwoHalo);
        
        // Assuming domain origin is at (-L/2, -L/2, -L/2)
        double origin_x = -(Nx_TwoHalo * dx) / 2.0;
        double origin_y = -(Ny_TwoHalo * dx) / 2.0;
        double origin_z = -(Nz_TwoHalo * dx) / 2.0;
        
        if (rank == 0) {
            std::cout << "  Grid spacing dx: " << dx << "\n";
            std::cout << "  Domain origin: (" << origin_x << ", " << origin_y << ", " << origin_z << ")\n";
        }
        
        for (int i = 0; i < Nx_TwoHalo; i++) {
            m_xs_twohalo[i] = origin_x + i * dx;
        }
        for (int j = 0; j < Ny_TwoHalo; j++) {
            m_ys_twohalo[j] = origin_y + j * dx;
        }
        for (int k = 0; k < Nz_TwoHalo; k++) {
            m_zs_twohalo[k] = origin_z + k * dx;
        }
        
        // Copy Inactive_ID_TwoHalo from device to identify active/inactive sites
        std::vector<double> h_inactive_id(N_TwoHalo);
        // In real code: ScaLBL_CopyToHost(h_inactive_id.data(), Inactive_ID_TwoHalo, N_TwoHalo * sizeof(double));
        // For this example, simulate it:
        for (int i = 0; i < N_TwoHalo; i++) {
            // Simulate: -1 = inactive, anything else = active
            // You'll need to adjust this based on your actual marking scheme
            h_inactive_id[i] = (i % 5 == 0) ? -1.0 : 1.0;  // Dummy data
        }
        
        // Build lists of active coordinates and ghost indices
        m_active_coords.clear();
        m_ghost_indices.clear();
        m_is_active_site.resize(N_TwoHalo);
        
        for (int i = 0; i < Nx_TwoHalo; i++) {
            for (int j = 0; j < Ny_TwoHalo; j++) {
                for (int k = 0; k < Nz_TwoHalo; k++) {
                    int idx = i * Ny_TwoHalo * Nz_TwoHalo + j * Nz_TwoHalo + k;
                    
                    // Check if this is an active site
                    // Adjust this condition based on how Inactive_ID marks sites
                    // Common patterns: != -1, > 0, >= 0, etc.
                    bool is_active = (h_inactive_id[idx] != -1.0);
                    m_is_active_site[idx] = is_active;
                    
                    if (is_active) {
                        // Active site - store coordinates
                        m_active_coords.push_back(m_xs_twohalo[i]);
                        m_active_coords.push_back(m_ys_twohalo[j]);
                        m_active_coords.push_back(m_zs_twohalo[k]);
                    } else {
                        // Inactive/ghost site - store indices
                        m_ghost_indices.push_back(i);
                        m_ghost_indices.push_back(j);
                        m_ghost_indices.push_back(k);
                    }
                }
            }
        }
        
        m_num_active_recon = m_active_coords.size() / 3;
        m_num_ghost_recon = m_ghost_indices.size() / 3;
        
        if (rank == 0) {
            std::cout << "  Domain: " << Nx_TwoHalo << "x" << Ny_TwoHalo << "x" << Nz_TwoHalo << "\n";
            std::cout << "  Active cells: " << m_num_active_recon << "\n";
            std::cout << "  Ghost cells: " << m_num_ghost_recon << "\n";
        }
        
        // Check if we have enough active cells
        if (m_num_active_recon < 10) {
            std::cerr << "ERROR: Too few active cells for reconstruction!\n";
            return;
        }
        
        if (m_num_ghost_recon == 0) {
            if (rank == 0) {
                std::cout << "  No ghost cells to reconstruct - skipping initialization.\n";
            }
            return;
        }
        
        // Initialize CUDA context (allocate device memory)
        m_recon_ctx = cuda_reconstruction_init(
            m_num_active_recon,
            m_num_ghost_recon,
            Nx_TwoHalo, Ny_TwoHalo, Nz_TwoHalo
        );
        
        // Set static domain data (grid coords, indices) - ONLY ONCE!
        cuda_reconstruction_set_domain(
            m_recon_ctx,
            m_active_coords.data(),
            m_ghost_indices.data(),
            m_xs_twohalo.data(),
            m_ys_twohalo.data(),
            m_zs_twohalo.data()
        );
        
        m_recon_initialized = true;
        
        if (rank == 0) {
            std::cout << "  Reconstruction context initialized successfully.\n";
            std::cout << "==============================================\n\n";
        }
    }
    
    // ========================================================================
    // EXECUTE: Call every timestep at line 2970 in ColorModel.cpp
    // ========================================================================
    void ReconstructInactiveSites(int timestep) {
        if (!m_recon_initialized || m_num_ghost_recon == 0) {
            return;  // Skip if not initialized or no ghost cells
        }
        
        // Copy Phi_TwoHalo from device to host
        std::vector<double> h_phi_twohalo(N_TwoHalo);
        // In real code: ScaLBL_CopyToHost(h_phi_twohalo.data(), Phi_TwoHalo, N_TwoHalo * sizeof(double));
        // For this example, simulate it:
        for (int i = 0; i < N_TwoHalo; i++) {
            h_phi_twohalo[i] = std::sin(i * 0.01);  // Dummy data
        }
        
        // Extract active values from Phi_TwoHalo using cached mask
        std::vector<double> active_vals;
        active_vals.reserve(m_num_active_recon);
        
        for (int idx = 0; idx < N_TwoHalo; idx++) {
            if (m_is_active_site[idx]) {
                active_vals.push_back(h_phi_twohalo[idx]);
            }
        }
        
        // Sanity check
        if (active_vals.size() != m_num_active_recon) {
            std::cerr << "ERROR: Active cell count mismatch!\n";
            return;
        }
        
        // Allocate output arrays
        std::vector<double> reconstructed(m_num_ghost_recon);
        std::vector<double> exact_vals(m_num_ghost_recon);  // Not used but required
        std::vector<double> rel_err(m_num_ghost_recon);
        int num_bad = 0;
        
        // Execute polynomial reconstruction
        // This is FAST because it only copies active_vals to device!
        cuda_reconstruction_execute(
            m_recon_ctx,
            active_vals.data(),     // Only data that changes!
            dx,
            2.0 * dx,               // tau1
            4.0 * dx,               // tau2
            1e-7,                   // reg_lambda
            120,                    // max_neighbors
            200,                    // max_neighbors_bad
            0.2,                    // relerr_threshold
            1e-3,                   // phi_rel_clip
            reconstructed.data(),
            exact_vals.data(),
            rel_err.data(),
            &num_bad
        );
        
        // Fill reconstructed values back into h_phi_twohalo at ghost sites
        for (int g = 0; g < m_num_ghost_recon; g++) {
            int i = m_ghost_indices[g * 3 + 0];
            int j = m_ghost_indices[g * 3 + 1];
            int k = m_ghost_indices[g * 3 + 2];
            int idx = i * Ny_TwoHalo * Nz_TwoHalo + j * Nz_TwoHalo + k;
            
            h_phi_twohalo[idx] = reconstructed[g];
        }
        
        // Copy updated Phi_TwoHalo back to device
        // In real code: ScaLBL_CopyToDevice(Phi_TwoHalo, h_phi_twohalo.data(), N_TwoHalo * sizeof(double));
        
        // Optional: Print diagnostics periodically
        if (rank == 0 && timestep % 100 == 0) {
            // Compute statistics
            double mean_err = 0.0, max_err = 0.0;
            for (int i = 0; i < m_num_ghost_recon; i++) {
                mean_err += rel_err[i];
                max_err = std::max(max_err, rel_err[i]);
            }
            mean_err /= m_num_ghost_recon;
            
            std::cout << "Timestep " << timestep 
                      << ": Reconstruction complete (mean err: " << mean_err
                      << ", bad cells: " << num_bad << ")\n";
        }
    }
    
    // ========================================================================
    // CLEANUP: Call in destructor or when done
    // ========================================================================
    void CleanupReconstruction() {
        if (m_recon_initialized) {
            cuda_reconstruction_free(m_recon_ctx);
            m_recon_initialized = false;
            
            if (rank == 0) {
                std::cout << "Polynomial reconstruction context freed.\n";
            }
        }
    }
};

// ============================================================================
// MAIN: Demonstrates the usage pattern
// ============================================================================
int main() {
    std::cout << "ColorModel Polynomial Reconstruction Integration Example\n";
    std::cout << "========================================================\n\n";
    
    // Create a ColorModel-like object
    ColorModelExample model;
    
    // Set up domain (simulated)
    model.Nx_TwoHalo = 40;  // Use smaller grid for quick demo
    model.Ny_TwoHalo = 40;
    model.Nz_TwoHalo = 40;
    model.N_TwoHalo = model.Nx_TwoHalo * model.Ny_TwoHalo * model.Nz_TwoHalo;
    model.dx = 0.04;  // 1.6 / 40
    model.rank = 0;
    
    // Step 1: Initialize reconstruction (call once during ColorModel setup)
    std::cout << "Step 1: Initializing reconstruction...\n";
    model.InitializeReconstruction();
    
    // Step 2: Run multiple timesteps (simulates ColorModel::Run() loop)
    std::cout << "\nStep 2: Running timesteps with reconstruction...\n";
    int num_timesteps = 10;
    
    for (int ts = 1; ts <= num_timesteps; ts++) {
        // Simulate timestep work...
        // ...
        
        // At line 2970 in ColorModel.cpp, after ScaLBL_Shift_Fill:
        // This is where you would call reconstruction
        model.ReconstructInactiveSites(ts);
        
        // Continue with rest of timestep...
    }
    
    std::cout << "\nStep 3: Cleanup...\n";
    // Step 3: Cleanup happens automatically in destructor
    // But you can also call it explicitly if needed
    model.CleanupReconstruction();
    
    std::cout << "\n========================================================\n";
    std::cout << "Integration Example Complete!\n";
    std::cout << "\nKey Integration Points in ColorModel.cpp:\n";
    std::cout << "1. Add member variables for context and cached data\n";
    std::cout << "2. Call InitializeReconstruction() in constructor/Initialize()\n";
    std::cout << "3. Call ReconstructInactiveSites() at line 2970\n";
    std::cout << "4. Call CleanupReconstruction() in destructor\n";
    std::cout << "5. Include poly_cuda.h and link against cusolver\n";
    
    return 0;
}
