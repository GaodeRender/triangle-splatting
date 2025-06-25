# 2D Triangle Splatting for Direct Differentiable Mesh Training

This project implements an interactive web viewer for **2D Triangle Splatting**, a novel technique for direct differentiable mesh training.

## Authors

- **Kaifeng Sheng*** (Amap, Alibaba)
- **Zheng Zhou*** (Amap, Alibaba)
- **Yingliang Peng** (Amap, Alibaba)
- **Qianwei Wang** (Amap, Alibaba)

*Equal Contribution

## Overview

2D Triangle Splatting is a cutting-edge approach for training 3D meshes in a differentiable manner. This technique enables direct optimization of triangle-based mesh representations, offering advantages in terms of computational efficiency and geometric accuracy.

## Features

- **Interactive 3D Viewer**: Explore different mesh representations with real-time controls
- **Multiple Mesh Formats**: Support for GLB, PLY, and OBJ file formats
- **Comparison Views**: Side-by-side comparison of different reconstruction methods
- **Visualization Controls**: 
  - Wireframe rendering toggle
  - Auto-rotation controls
  - Backface culling options
  - Adjustable lighting
  - Face count display

## Supported Models

The viewer showcases results from different reconstruction methods:

- **2DTS (2D Triangle Splatting)**: Our proposed method in GLB format
- **2DGS (2D Gaussian Splatting)**: Baseline comparison in PLY format  
- **Nvdiffrec**: Traditional differentiable rendering approach in OBJ format

## Scenes

- **Ship**: Maritime vessel reconstruction
- **Ficus**: Botanical subject reconstruction

## Technology Stack

- **Three.js**: 3D graphics and rendering
- **WebGL**: Hardware-accelerated graphics
- **Modern JavaScript**: ES6+ modules and features
- **Responsive CSS**: Modern web styling

## Usage

1. Open `index.html` in a modern web browser
2. Use the dropdown menus to select different mesh representations
3. Toggle various visualization options using the checkboxes
4. Adjust lighting with the slider controls
5. Compare results across different reconstruction methods

## File Structure

```
├── index.html          # Main webpage
├── script.js           # JavaScript application logic
├── style.css           # Styling and layout
├── README.md           # This documentation
└── assets/             # 3D model assets
    ├── ship/           # Ship scene assets
    │   ├── 2dgs.ply
    │   ├── 2dts.glb
    │   ├── nvdiffrec.obj
    │   ├── nvdiffrec.mtl
    │   └── textures...
    └── ficus/          # Ficus scene assets
        ├── 2dgs.ply
        ├── 2dts.glb
        ├── nvdiffrec.obj
        ├── nvdiffrec.mtl
        └── textures...
```

## Links

- **Paper**: [arXiv:2506.18575](https://arxiv.org/abs/2506.18575)
- **Code**: [GitHub Repository](https://github.com/GaodeRender/diff_recon)

## License

This project is associated with Amap, Alibaba. Please refer to the official repository for licensing information.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{sheng2025triangle,
  title={2D Triangle Splatting for Direct Differentiable Mesh Training},
  author={Sheng, Kaifeng and Zhou, Zheng and Peng, Yingliang and Wang, Qianwei},
  journal={arXiv preprint arXiv:2506.18575},
  year={2025}
}
```
