# ColorModel Integration Guide

This guide shows how to integrate polynomial reconstruction into `ColorModel.cpp`.

## Quick Start

Run the integration example:
```bash
make colormodel
```

## Integration Steps

### 1. Add Member Variables to ColorModel Class

In your ColorModel header (or at the top of the class definition):

```cpp
#include "poly_cuda.h"

class ColorModel {
private:
    // Polynomial reconstruction context
    PolyReconstructionContext m_recon_ctx;
    bool m_recon_initialized;
    
    // Cached domain structure
    std::vector<double> m_active_coords;
    std::vector<int> m_ghost_indices;
    std::vector<double> m_xs_twohalo, m_ys_twohalo, m_zs_twohalo;
    std::vector<bool> m_is_active_site;
    int m_num_active_recon, m_num_ghost_recon;
    
    // Methods
    void InitializeReconstruction();
    void ReconstructInactiveSites(int timestep);
    void CleanupReconstruction();
```

### 2. Initialize During Setup

Call this ONCE in your `ColorModel::Initialize()` or constructor (after domain is set up):

```cpp
void ColorModel::Initialize() {
    // ... existing initialization code ...
    
    // Initialize polynomial reconstruction
    InitializeReconstruction();
}
```

See `colormodel_integration_example.cpp` for the full implementation of `InitializeReconstruction()`.

### 3. Execute at Line 2970

Insert this code at line 2970 in `ColorModel.cpp`, right after `ScaLBL_Shift_Fill`:

```cpp
ScaLBL_Shift_Fill(Phi_TwoHalo, Phi, Nx, Ny, Nz);

// ============================================================================
// POLYNOMIAL RECONSTRUCTION: Fill inactive sites
// ============================================================================
if (m_recon_initialized && m_num_ghost_recon > 0) {
    // DEBUG: Print reconstruction parameters
    if (rank == 0 && timestep % 100 == 0) {
        printf("\n=== DEBUG: Reconstruction at timestep %d ===\n", timestep);
        printf("  Grid spacing dx: %.6e\n", dx);
        printf("  Domain size: %dx%dx%d\n", Nx_TwoHalo, Ny_TwoHalo, Nz_TwoHalo);
        printf("  Total cells: %d\n", N_TwoHalo);
        printf("  Active cells: %d\n", m_num_active_recon);
        printf("  Ghost cells: %d\n", m_num_ghost_recon);
        printf("  tau1: %.6e\n", 2.0 * dx);
        printf("  tau2: %.6e\n", 4.0 * dx);
    }
    
    // Copy Phi_TwoHalo from device
    std::vector<double> h_phi_twohalo(N_TwoHalo);
    ScaLBL_CopyToHost(h_phi_twohalo.data(), Phi_TwoHalo, N_TwoHalo * sizeof(double));
    
    // Extract active values (use cached mask for speed)
    std::vector<double> active_vals;
    active_vals.reserve(m_num_active_recon);
    for (int idx = 0; idx < N_TwoHalo; idx++) {
        if (m_is_active_site[idx]) {
            active_vals.push_back(h_phi_twohalo[idx]);
        }
    }
    
    // DEBUG: Check active values statistics
    if (rank == 0 && timestep % 100 == 0) {
        double min_val = active_vals[0], max_val = active_vals[0], sum_val = 0.0;
        int num_zero = 0, num_nan = 0, num_inf = 0;
        for (size_t i = 0; i < active_vals.size(); i++) {
            double val = active_vals[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum_val += val;
            if (val == 0.0) num_zero++;
            if (std::isnan(val)) num_nan++;
            if (std::isinf(val)) num_inf++;
        }
        printf("  Active values: min=%.6e, max=%.6e, mean=%.6e\n", 
               min_val, max_val, sum_val / active_vals.size());
        printf("  Active values: zeros=%d, NaNs=%d, Infs=%d\n", num_zero, num_nan, num_inf);
        
        // Print first few active coordinates and values
        printf("  First 5 active points:\n");
        int count = 0;
        for (int idx = 0; idx < N_TwoHalo && count < 5; idx++) {
            if (m_is_active_site[idx]) {
                int i = idx / (Ny_TwoHalo * Nz_TwoHalo);
                int j = (idx / Nz_TwoHalo) % Ny_TwoHalo;
                int k = idx % Nz_TwoHalo;
                printf("    [%d,%d,%d] idx=%d coord=(%.4f,%.4f,%.4f) val=%.6e\n",
                       i, j, k, idx, 
                       m_xs_twohalo[i], m_ys_twohalo[j], m_zs_twohalo[k],
                       active_vals[count]);
                count++;
            }
        }
        
        // Print first few ghost coordinates
        printf("  First 5 ghost points:\n");
        for (int g = 0; g < std::min(5, m_num_ghost_recon); g++) {
            int i = m_ghost_indices[g * 3 + 0];
            int j = m_ghost_indices[g * 3 + 1];
            int k = m_ghost_indices[g * 3 + 2];
            int idx = i * Ny_TwoHalo * Nz_TwoHalo + j * Nz_TwoHalo + k;
            printf("    [%d,%d,%d] idx=%d coord=(%.4f,%.4f,%.4f)\n",
                   i, j, k, idx,
                   m_xs_twohalo[i], m_ys_twohalo[j], m_zs_twohalo[k]);
        }
    }
    
    // Allocate outputs
    std::vector<double> reconstructed(m_num_ghost_recon);
    std::vector<double> exact_vals(m_num_ghost_recon);
    std::vector<double> rel_err(m_num_ghost_recon);
    int num_bad = 0;
    
    // Execute reconstruction (FAST - only copies active_vals!)
    cuda_reconstruction_execute(
        m_recon_ctx,
        active_vals.data(),
        dx,
        2.0 * dx,  // tau1
        4.0 * dx,  // tau2
        1e-7,      // reg_lambda
        120,       // max_neighbors
        200,       // max_neighbors_bad
        0.2,       // relerr_threshold
        1e-3,      // phi_rel_clip
        reconstructed.data(),
        exact_vals.data(),
        rel_err.data(),
        &num_bad
    );
    
    // DEBUG: Check reconstructed values
    if (rank == 0 && timestep % 100 == 0) {
        double min_rec = reconstructed[0], max_rec = reconstructed[0], sum_rec = 0.0;
        double min_err = rel_err[0], max_err = rel_err[0], sum_err = 0.0;
        int num_rec_zero = 0, num_rec_nan = 0, num_rec_inf = 0;
        int num_high_err = 0;
        
        for (int i = 0; i < m_num_ghost_recon; i++) {
            double val = reconstructed[i];
            double err = rel_err[i];
            min_rec = std::min(min_rec, val);
            max_rec = std::max(max_rec, val);
            sum_rec += val;
            min_err = std::min(min_err, err);
            max_err = std::max(max_err, err);
            sum_err += err;
            if (val == 0.0) num_rec_zero++;
            if (std::isnan(val)) num_rec_nan++;
            if (std::isinf(val)) num_rec_inf++;
            if (err > 0.2) num_high_err++;
        }
        
        double mean_rec = sum_rec / m_num_ghost_recon;
        double mean_err = sum_err / m_num_ghost_recon;
        
        printf("  Reconstructed values: min=%.6e, max=%.6e, mean=%.6e\n", 
               min_rec, max_rec, mean_rec);
        printf("  Reconstructed values: zeros=%d, NaNs=%d, Infs=%d\n", 
               num_rec_zero, num_rec_nan, num_rec_inf);
        printf("  Relative errors: min=%.6e, max=%.6e, mean=%.6e\n", 
               min_err, max_err, mean_err);
        printf("  High errors (>0.2): %d out of %d (%.1f%%)\n", 
               num_high_err, m_num_ghost_recon, 100.0 * num_high_err / m_num_ghost_recon);
        printf("  Bad cells: %d (%.1f%%)\n", 
               num_bad, 100.0 * num_bad / m_num_ghost_recon);
        
        // Print details of first few reconstructed points
        printf("  First 5 reconstructed points:\n");
        for (int g = 0; g < std::min(5, m_num_ghost_recon); g++) {
            int i = m_ghost_indices[g * 3 + 0];
            int j = m_ghost_indices[g * 3 + 1];
            int k = m_ghost_indices[g * 3 + 2];
            printf("    [%d,%d,%d] rec_val=%.6e, rel_err=%.6e%s\n",
                   i, j, k, reconstructed[g], rel_err[g],
                   (rel_err[g] > 0.2) ? " HIGH_ERROR" : "");
        }
        
        // Print details of worst errors
        if (num_high_err > 0) {
            printf("  Points with highest errors:\n");
            std::vector<std::pair<double, int>> err_idx;
            for (int i = 0; i < m_num_ghost_recon; i++) {
                err_idx.push_back(std::make_pair(rel_err[i], i));
            }
            std::sort(err_idx.begin(), err_idx.end(), std::greater<std::pair<double, int>>());
            
            for (int n = 0; n < std::min(5, (int)err_idx.size()); n++) {
                int g = err_idx[n].second;
                int i = m_ghost_indices[g * 3 + 0];
                int j = m_ghost_indices[g * 3 + 1];
                int k = m_ghost_indices[g * 3 + 2];
                printf("    [%d,%d,%d] coord=(%.4f,%.4f,%.4f) rec_val=%.6e, rel_err=%.6e\n",
                       i, j, k,
                       m_xs_twohalo[i], m_ys_twohalo[j], m_zs_twohalo[k],
                       reconstructed[g], rel_err[g]);
            }
        }
        printf("===========================================\n\n");
    }
    
    // Fill reconstructed values back
    for (int g = 0; g < m_num_ghost_recon; g++) {
        int i = m_ghost_indices[g * 3 + 0];
        int j = m_ghost_indices[g * 3 + 1];
        int k = m_ghost_indices[g * 3 + 2];
        int idx = i * Ny_TwoHalo * Nz_TwoHalo + j * Nz_TwoHalo + k;
        h_phi_twohalo[idx] = reconstructed[g];
    }
    
    // Copy back to device
    ScaLBL_CopyToDevice(Phi_TwoHalo, h_phi_twohalo.data(), N_TwoHalo * sizeof(double));
}
// ============================================================================

ScaLBL_Comm_Regular_TwoHalo->SendHalo(Phi_TwoHalo);
ScaLBL_Comm_Regular_TwoHalo->RecvHalo(Phi_TwoHalo);
```

### 4. Cleanup in Destructor

```cpp
ColorModel::~ColorModel() {
    // ... existing cleanup ...
    
    CleanupReconstruction();
}
```

### 5. Build System

Update your CMakeLists.txt:

```cmake
# Add CUDA files
set(CUDA_SOURCES
    ${CUDA_SOURCES}
    path/to/poly_cuda.cu
)

# Add include directory
include_directories(path/to/poly_cuda_headers)

# Link against cuSOLVER
target_link_libraries(ColorModel cusolver)
```

## Performance

For a 160³ grid with ~2M active cells and ~120K ghost cells:

- **One-time setup**: ~10 seconds (domain initialization)
- **Per reconstruction call**: ~170 ms (only copies 17 MB)

**Speedup for 1000 timesteps:**
- Old way (if done naively): ~3 hours
- New way: ~3 minutes
- **Speedup: ~60x**

## Key Optimization

The optimized API separates:
1. **Static data** (grid coords, indices) - copied ONCE during init
2. **Dynamic data** (active_vals) - copied EVERY call (~17 MB)

This minimizes host-device transfer overhead!

## Files

- `colormodel_integration_example.cpp` - Complete working example
- `poly_cuda.h` - Header with API
- `poly_cuda.cu` - CUDA implementation
- `optimized_example.cpp` - Performance demonstration

## Questions?

Check the example files or refer to the implementation in `colormodel_integration_example.cpp`.
