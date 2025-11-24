from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import subprocess

# Find CUDA installation
def find_cuda():
    """Locate CUDA installation directory"""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    
    if cuda_home is None:
        # Try common CUDA locations
        possible_paths = [
            '/usr/local/cuda',
            '/usr/local/cuda-12.0',
            '/usr/local/cuda-11.8',
            '/usr/local/cuda-11.7',
            '/opt/cuda',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                cuda_home = path
                break
    
    if cuda_home is None:
        raise EnvironmentError(
            'CUDA installation not found. Please set CUDA_HOME environment variable.'
        )
    
    return cuda_home

cuda_home = find_cuda()
print(f"Using CUDA from: {cuda_home}")

# Compile CUDA code to object file
cuda_source = 'poly_cuda.cu'
cuda_object = 'poly_cuda.o'

nvcc_cmd = [
    f'{cuda_home}/bin/nvcc',
    '-c', cuda_source,
    '-o', cuda_object,
    '-O3',
    '--compiler-options', '-fPIC',
    '-arch=sm_89',  # Using sm_89 for broad compatibility (Ada/Hopper) - adjust for sm_90 if needed
]

print(f"Compiling CUDA code: {' '.join(nvcc_cmd)}")
result = subprocess.run(nvcc_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("CUDA compilation failed:")
    print(result.stderr)
    raise RuntimeError("CUDA compilation failed")
print("CUDA compilation successful")

# Define Cython extension
extensions = [
    Extension(
        name="poly_cuda_wrapper",
        sources=["poly_cuda_wrapper.pyx"],
        extra_objects=[cuda_object],
        include_dirs=[
            np.get_include(),
            cuda_home + '/include',
        ],
        library_dirs=[
            cuda_home + '/lib64',
        ],
        libraries=['cudart', 'cusolver'],
        language='c++',
        extra_compile_args=['-std=c++11'],
    )
]

setup(
    name='poly_cuda_wrapper',
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)
