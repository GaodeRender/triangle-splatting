from setuptools import setup, find_packages
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

install_requires = [
    f"custom-gaussian-rasterization @ file://localhost/{root_dir}/submodules/custom-gaussian-rasterization",
    f"simple-knn @ file://localhost/{root_dir}/submodules/simple-knn",
    f"diff-triangle-rasterization-2D @ file://localhost/{root_dir}/submodules/diff-triangle-rasterization-2D",
    f"diff-triangle-rasterization-3D @ file://localhost/{root_dir}/submodules/diff-triangle-rasterization-3D",
]
# install_requires += [x.strip() for x in open("requirements.txt").read().splitlines() if x.strip() and not x.strip().startswith("#")]
 
setup(
    name="diff_recon",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    author="Kaifeng Sheng",
    author_email="shengkaifeng.skf@alibaba-inc.com",
    description="Differentiable Reconstruction",
    python_requires=">=3.12",
)
