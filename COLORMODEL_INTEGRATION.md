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
    
    // Optional diagnostics
    if (rank == 0 && timestep % 100 == 0) {
        double mean_err = 0.0;
        for (int i = 0; i < m_num_ghost_recon; i++) mean_err += rel_err[i];
        mean_err /= m_num_ghost_recon;
        printf("Reconstruction: mean_err=%.3e, bad_cells=%d\n", mean_err, num_bad);
    }
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
