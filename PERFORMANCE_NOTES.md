# Performance Analysis: poly3d_6.py with Box-Based Neighbor Search

## Current Implementation

Changed from KNN (find K nearest neighbors) to **box-based search** (find all neighbors within a box of radius R lattice sites).

### Algorithm:
- **Pass 1**: Search within 3 lattice sites (7×7×7 box, ~343 cells max)
- **Pass 2**: Search within 4 lattice sites (9×9×9 box, ~729 cells max)

### Performance Results:

```
    N   Active    Ghost   Classify       CUDA      Total
         cells    cells       (ms)       (ms)       (ms)
------------------------------------------------------------
   10      360      464       6.78     389.76     396.85
   20     3544     1880      74.63     205.18     281.27
   40    30976     7568     711.89     957.83    1683.15
```

## Bottlenecks

### 1. **CPU Cell Classification: O(N³)** [MAJOR at high N]
- N=40: **712 ms (42% of total time)**
- Triple nested loop over entire grid
- Easy to parallelize on GPU

### 2. **Box Search Still O(M×N)** [MAJOR]
- Still loops through ALL active cells for each ghost cell
- Box test is fast, but checking all N cells is expensive
- For N=40: 7,568 ghost × 30,976 active = 234M comparisons

### 3. **Per-Thread Linear Solve: O(M×K×m²)**
- Each thread solves 10×10 system with Cholesky
- Not the bottleneck currently

## Why Box Search Didn't Speed Up Much

The box-based approach is **conceptually correct** but still has O(M×N) complexity because:

1. Active cells are stored as a **flat array of coordinates**
2. No spatial indexing structure
3. Must check every active cell to see if it's in the box
4. Early exit helps slightly but doesn't change complexity

## Required Optimization: Spatial Indexing

To achieve true O(M×log(N)) or O(M×1), we need:

### Option A: GPU Spatial Hash Grid
- Pre-compute hash grid of active cells
- Each grid cell stores list of active cells
- Lookup is O(1) per box cell
- **Estimated speedup: 10-100×** for CUDA portion

### Option B: GPU KD-Tree (RAPIDS cuML)
- Build KD-tree on active cell coordinates
- Query K nearest neighbors in O(log(N))
- Requires external library

### Option C: Sorted + Binary Search
- Sort active cells by Z-curve (3D space-filling curve)
- Binary search to find range
- Pure CUDA, no external libs

## Recommended Next Steps

1. **Move classification to GPU** (trivial parallelization)
   - Eliminates 712ms (42%) for N=40
   
2. **Implement spatial hash grid**
   - Pass grid structure from Python
   - Hash active cells into grid bins
   - Lookup only relevant bins
   - **Target: 10-50× speedup for CUDA kernel**

3. **Combined speedup estimate**:
   - N=40 current: 1683 ms
   - After GPU classification: ~970 ms (eliminate 712ms)
   - After spatial hash: ~100-200 ms (10× CUDA speedup)
   - **Total speedup: 8-17×**

## Current Status

✓ Box-based search implemented (correctness verified)
✓ Slightly better than KNN (avoids sorting)
✗ Still O(M×N) complexity
→ **Need spatial indexing for real speedup**
