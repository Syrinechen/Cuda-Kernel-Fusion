from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "kernel_wrapper",
        sources=["kernel_wrapper.pyx", "path/to/your/cuda_kernel.cu"],
        library_dirs=['/usr/local/cuda/lib64'],
        libraries=['cudart'],
        language='c++',
        extra_compile_args={
            'gcc': [],
            'nvcc': ['-arch=sm_52']
        },
        include_dirs=[np.get_include(), '/usr/local/cuda/include']
    )
]

setup(
    name='CUDA Matrix Sigmoid Multiplication',
    ext_modules=cythonize(ext_modules)
)
