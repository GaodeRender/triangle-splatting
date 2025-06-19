from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

pkg_name = "diff_triangle_rasterization_3D"

setup(
    name=f"{pkg_name}",
    version="1.0.0",
    packages=[f"{pkg_name}"],
    ext_modules=[
        CUDAExtension(
            name=f"{pkg_name}._C",
            sources=[
                "src/rasterizer.cu",
                "src/forward.cu",
                "src/backward.cu",
                "src/extension_interface.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": []},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
