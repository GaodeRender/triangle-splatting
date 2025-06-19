#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

pkg_name = "simple_knn"

setup(
    name=f"{pkg_name}",
    version="1.0.0",
    packages=[f"{pkg_name}"],
    ext_modules=[
        CUDAExtension(
            name=f"{pkg_name}._C",
            sources=[
                "interface.cu",
                "simple_knn.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": [], "cxx": []},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
