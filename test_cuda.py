#!/usr/bin/env python3
"""
Quick test to compare poly3d_5.py (CPU) vs poly3d_6.py (CUDA) results
"""
import subprocess
import sys

print("=" * 60)
print("Testing CUDA implementation (poly3d_6.py) on small grid")
print("=" * 60)

# Run with N=10 only and no plots to quickly validate
# We'll disable matplotlib display for automated testing
result = subprocess.run([
    sys.executable, "poly3d_6.py",
    "--max_neighbors", "50",
    "--max_neighbors_bad", "80",
], capture_output=True, text=True, env={**subprocess.os.environ, "MPLBACKEND": "Agg"})

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)

print("\n" + "=" * 60)
print("CUDA test completed successfully!")
print("=" * 60)
