from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='tensorstore_helper',
      ext_modules=[cpp_extension.CUDAExtension('tensorstore_helper',
                      ['tensorstore_helper.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

