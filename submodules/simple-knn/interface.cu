/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include "cuda_runtime.h"

#include "interface.h"
#include "simple_knn.h"

torch::Tensor distCUDA2(const torch::Tensor &points)
{
	if (points.ndimension() != 2 || points.size(1) != 3)
	{
		AT_ERROR("points must have dimensions (num_points, 3)");
	}

	int init_device;
	cudaGetDevice(&init_device);
	cudaSetDevice(points.device().index());

	const int P = points.size(0);
	auto float_opts = points.options().dtype(torch::kFloat32);
	torch::Tensor means = torch::full({P}, 0.0, float_opts);

	SimpleKNN::knn(P, (float3 *)points.contiguous().data_ptr<float>(), means.contiguous().data_ptr<float>());

	cudaSetDevice(init_device);
	return means;
}

torch::Tensor nearestNeighbor(const torch::Tensor &points, int batch_size)
{
	if (batch_size <= 0)
	{
		AT_ERROR("batch_size must be greater than 0");
	}
	if (points.ndimension() != 2 || points.size(1) != 3 || points.size(0) % batch_size != 0)
	{
		AT_ERROR("points must have dimensions (num_points, 3) and num_points % batch_size == 0, where batch_size = ", batch_size);
	}

	int init_device;
	cudaGetDevice(&init_device);
	cudaSetDevice(points.device().index());

	const int P = points.size(0);
	auto uint32_opts = points.options().dtype(torch::kUInt32);
	torch::Tensor indices_nearest = torch::full({P}, 0, uint32_opts);

	SimpleKNN::nearestNeighbor(P, batch_size, (float3 *)points.contiguous().data_ptr<float>(), indices_nearest.contiguous().data_ptr<uint32_t>());

	cudaSetDevice(init_device);
	return indices_nearest;
}
