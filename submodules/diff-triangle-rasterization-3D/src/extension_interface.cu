#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include <c10/cuda/CUDAGuard.h>

#include "extension_interface.h"
#include "rasterizer.h"
#include "config.h"

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeTrianglesForward(
	const int image_width,
	const int image_height,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor &campos,
	const int sh_degree,
	const float gamma,
	const float scale_modifier,
	const float background_depth,
	const torch::Tensor &background,
	const torch::Tensor &vertex,
	const torch::Tensor &shs,
	const torch::Tensor &feature,
	const torch::Tensor &opacity,
	const bool back_culling,
	const bool rich_info,
	const bool debug)
{
	const int P = vertex.size(0);
	const int H = image_height;
	const int W = image_width;
	const bool use_shs = feature.ndimension() <= 1 || (feature.size(0) == 0 && shs.size(0) > 0);
	const int C = use_shs ? 3 : feature.size(1);
	int M = 0;
	if (shs.size(0) != 0)
	{
		M = shs.size(1);
	}

	// Check inputs
	if (vertex.ndimension() != 3 || vertex.size(1) != 3 || vertex.size(2) != 3)
	{
		AT_ERROR("vertex must have dimensions (num_points, 3, 3)");
	}
	if (!use_shs && feature.ndimension() != 2)
	{
		AT_ERROR("feature must have dimensions (num_points, num_channels)");
	}
	if (use_shs && shs.ndimension() != 3)
	{
		AT_ERROR("shs must have dimensions (num_points, (1 + sh_degree) ** 2, 3)");
	}
	if (C > MAX_CHANNELS)
	{
		AT_ERROR("feature's num_channels can't be larger than MAX_CHANNELS");
	}
	if (C != background.size(0))
	{
		AT_ERROR("background must have the same number of channels as feature");
	}
	if (gamma < 0.0f)
	{
		AT_ERROR("gamma must be larger than 0");
	}

	at::cuda::OptionalCUDAGuard device_guard(vertex.device());

	Params::CameraInfo cameraInfo = {
		W, H, tan_fovx, tan_fovy,
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>()};

	Params::GeometryInfo geometryInfo = {
		P, sh_degree, M, C, use_shs, gamma, scale_modifier, background_depth,
		background.contiguous().data_ptr<float>(),
		vertex.contiguous().data_ptr<float>(),
		shs.contiguous().data_ptr<float>(),
		feature.contiguous().data_ptr<float>(),
		opacity.contiguous().data_ptr<float>()};

	auto int_opts = vertex.options().dtype(torch::kInt32);
	auto float_opts = vertex.options().dtype(torch::kFloat32);
	auto byte_opts = vertex.options().dtype(torch::kByte);

	torch::Tensor out_feature = torch::zeros({C, H, W}, float_opts);
	torch::Tensor radii = torch::zeros({P}, int_opts);

	torch::Tensor depth = torch::empty({0}, float_opts);
	torch::Tensor normal = torch::empty({0}, float_opts);
	torch::Tensor contrib_sum = torch::empty({0}, float_opts);
	torch::Tensor contrib_max = torch::empty({0}, float_opts);
	if (rich_info)
	{
		depth = torch::full({H, W}, 0.0, float_opts);
		normal = torch::full({3, H, W}, 0.0, float_opts);
		contrib_sum = torch::full({P}, 0.0, float_opts);
		contrib_max = torch::full({P}, 0.0, float_opts);
	}

	Params::ForwardOutput forwardOutput = {
		0, // num_rendered
		out_feature.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>(),
		depth.contiguous().data_ptr<float>(),
		normal.contiguous().data_ptr<float>(),
		contrib_sum.contiguous().data_ptr<float>(),
		contrib_max.contiguous().data_ptr<float>(),
		torch::empty({0}, byte_opts),  // geometryBuffer
		torch::empty({0}, byte_opts),  // binningBuffer
		torch::empty({0}, byte_opts)}; // imageBuffer

	if (P != 0)
	{
		Rasterizer::forward(
			cameraInfo,
			geometryInfo,
			forwardOutput,
			back_culling,
			rich_info,
			debug);
	}

	return std::make_tuple(
		forwardOutput.num_rendered,
		out_feature,
		radii,
		depth,
		normal,
		contrib_sum,
		contrib_max,
		forwardOutput.geometryBuffer,
		forwardOutput.binningBuffer,
		forwardOutput.imageBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeTrianglesBackward(
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const torch::Tensor &campos,
	const int sh_degree,
	const float gamma,
	const float scale_modifier,
	const float background_depth,
	const torch::Tensor &background,
	const torch::Tensor &vertex,
	const torch::Tensor &shs,
	const torch::Tensor &feature,
	const torch::Tensor &opacity,
	const int num_rendered,
	const torch::Tensor &radii,
	const torch::Tensor &geometryBuffer,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const torch::Tensor &dL_dout_feature,
	const torch::Tensor &dL_dout_depth,
	const torch::Tensor &dL_dout_normal,
	const bool rich_info,
	const bool debug)
{
	const int P = vertex.size(0);
	const int H = dL_dout_feature.size(1);
	const int W = dL_dout_feature.size(2);
	const bool use_shs = feature.ndimension() <= 1 || (feature.size(0) == 0 && shs.size(0) > 0);
	const int C = use_shs ? 3 : feature.size(1);
	int M = 0;
	if (shs.size(0) != 0)
	{
		M = shs.size(1);
	}

	at::cuda::OptionalCUDAGuard device_guard(vertex.device());

	Params::CameraInfo cameraInfo = {
		W, H, tan_fovx, tan_fovy,
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>()};

	Params::GeometryInfo geometryInfo = {
		P, sh_degree, M, C, use_shs, gamma, scale_modifier, background_depth,
		background.contiguous().data_ptr<float>(),
		vertex.contiguous().data_ptr<float>(),
		shs.contiguous().data_ptr<float>(),
		feature.contiguous().data_ptr<float>(),
		opacity.contiguous().data_ptr<float>()};

	Params::BackwardInput backwardInput = {
		num_rendered,
		radii.contiguous().data_ptr<int>(),
		geometryBuffer,
		binningBuffer,
		imageBuffer};

	Params::LossInput lossInput = {
		dL_dout_feature.contiguous().data_ptr<float>(),
		dL_dout_depth.contiguous().data_ptr<float>(),
		dL_dout_normal.contiguous().data_ptr<float>()};

	torch::Tensor dL_dvertex = torch::zeros({P, 3, 3}, vertex.options());
	torch::Tensor dL_dcenter2D = torch::zeros({P, 2}, vertex.options());
	torch::Tensor dL_dshs = torch::zeros({P, M, 3}, vertex.options());
	torch::Tensor dL_dfeature = torch::zeros({P, C}, vertex.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, vertex.options());

	Params::BackwardOutput backwardOutput = {
		dL_dvertex.contiguous().data_ptr<float>(),
		dL_dcenter2D.contiguous().data_ptr<float>(),
		dL_dshs.contiguous().data_ptr<float>(),
		dL_dfeature.contiguous().data_ptr<float>(),
		dL_dopacity.contiguous().data_ptr<float>()};

	if (P != 0)
	{
		Rasterizer::backward(
			cameraInfo,
			geometryInfo,
			backwardInput,
			lossInput,
			backwardOutput,
			rich_info,
			debug);
	}

	return std::make_tuple(
		dL_dvertex,
		dL_dcenter2D,
		dL_dshs,
		dL_dfeature,
		dL_dopacity);
}
