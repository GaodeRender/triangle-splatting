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
from os.path import abspath, dirname, join

glm_path = join(dirname(dirname(abspath(__file__))), "glm")
pkg_name = "custom_gaussian_rasterization"

setup(
    name=f"{pkg_name}",
    version="1.0.0",
    packages=[f"{pkg_name}"],
    ext_modules=[
        CUDAExtension(
            name=f"{pkg_name}._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": ["-I" + glm_path]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
