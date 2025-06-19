#include <torch/extension.h>
#include "src/extension_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rasterize_triangles", &rasterizeTrianglesForward);
  // m.def("rasterize_triangles_rich_info", &rasterizeTrianglesRichInfoForward);
  m.def("rasterize_triangles_backward", &rasterizeTrianglesBackward);
}