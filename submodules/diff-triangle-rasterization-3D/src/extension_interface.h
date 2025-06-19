#pragma once

#include <torch/extension.h>
#include <cstdio>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeTrianglesForward(
	// camera info
	const int image_width,
	const int image_height,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor &campos,
	// geometry info
	const int sh_degree,
	const float gamma,
	const float scale_modifier,
	const float background_depth,
	const torch::Tensor &background,
	const torch::Tensor &vertex,
	const torch::Tensor &shs,
	const torch::Tensor &feature,
	const torch::Tensor &opacity,
	// control flags
	const bool back_culling,
	const bool rich_info,
	const bool debug = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeTrianglesBackward(
	// camera info
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor &campos,
	// geometry info
	const int sh_degree,
	const float gamma,
	const float scale_modifier,
	const float background_depth,
	const torch::Tensor &background,
	const torch::Tensor &vertex,
	const torch::Tensor &shs,
	const torch::Tensor &feature,
	const torch::Tensor &opacity,
	// saved forward outputs
	const int num_rendered,
	const torch::Tensor &radii,
	const torch::Tensor &geometryBuffer,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	// loss input
	const torch::Tensor &dL_dout_feature,
	const torch::Tensor &dL_dout_depth,
	const torch::Tensor &dL_dout_normal,
	// control flags
	const bool rich_info,
	const bool debug = false);
