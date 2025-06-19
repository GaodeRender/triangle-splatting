#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "rasterizer.h"
#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

__global__ void duplicateWithKeys(
	int P, dim3 grid,
	const uint32_t *tiles_touched,
	const uint2 *rect_min,
	const uint2 *rect_max,
	const float *depth,
	const uint32_t *offsets,
	uint64_t *point_list_keys_unsorted,
	uint32_t *point_list_values_unsorted)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (tiles_touched[idx] <= 0)
		return;

	uint2 cur_rect_min = rect_min[idx];
	uint2 cur_rect_max = rect_max[idx];
	float cur_depth = depth[idx];
	uint32_t cur_offset = (idx == 0) ? 0 : offsets[idx - 1];

	// For each tile that the bounding rect overlaps, emit a key/value pair.
	// The key is | tile ID | depth |, and the value is the ID of the triangle.
	// Sorting the values with this key yields triangle IDs in a list,
	// such that they are first sorted by tile and then by depth.
	for (int y = cur_rect_min.y; y < cur_rect_max.y; y++)
	{
		for (int x = cur_rect_min.x; x < cur_rect_max.x; x++)
		{
			uint64_t key = y * grid.x + x;
			key <<= 32;
			key |= *((uint32_t *)&cur_depth);
			point_list_keys_unsorted[cur_offset] = key;
			point_list_values_unsorted[cur_offset] = idx;
			cur_offset++;
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in the full sorted list.
// If yes, write start/end of this tile. Run once per instanced (duplicated) triangle ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint32_t cur_tile = point_list_keys[idx] >> 32;
	if (idx == 0)
		ranges[cur_tile].x = 0;
	else
	{
		uint32_t prev_tile = point_list_keys[idx - 1] >> 32;
		if (cur_tile != prev_tile)
		{
			ranges[prev_tile].y = idx;
			ranges[cur_tile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[cur_tile].y = L;
}

void Rasterizer::forward(
	const Params::CameraInfo &cameraInfo,
	const Params::GeometryInfo &geometryInfo,
	Params::ForwardOutput &forwardOutput,
	const bool back_culling,
	const bool rich_info,
	const bool debug)
{
	const int W = cameraInfo.width;
	const int H = cameraInfo.height;
	const int P = geometryInfo.P;

	const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	Params::GeometryState geometryState(forwardOutput.geometryBuffer, (size_t)P, true);

	FORWARD::preprocessCUDA<<<(P + 255) / 256, 256>>>(
		W, H, P, geometryInfo.D, geometryInfo.M, rich_info,
		geometryInfo.use_shs, grid, back_culling,
		cameraInfo.viewmatrix,
		cameraInfo.projmatrix,
		cameraInfo.campos,
		geometryInfo.vertex,
		geometryInfo.shs,
		forwardOutput.radii,
		geometryState.v1_view,
		geometryState.v2_view,
		geometryState.v3_view,
		geometryState.normal_view,
		geometryState.depth,
		geometryState.rgb,
		geometryState.clamped,
		geometryState.tiles_touched,
		geometryState.rect_min,
		geometryState.rect_max);
	CHECK_CUDA(debug);

#ifdef DEBUG
	// Debugging: print out triangle properties
	float3 v1, v2, v3;
	float3 v1_view, v2_view, v3_view;
	uint2 rect_min, rect_max;
	int radii;
	for (int i = 0; i < P; i++)
	{
		cudaMemcpy(&v1, geometryInfo.vertex + 9 * i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v2, geometryInfo.vertex + 9 * i + 3, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v3, geometryInfo.vertex + 9 * i + 6, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v1_view, geometryState.v1_view + i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v2_view, geometryState.v2_view + i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&v3_view, geometryState.v3_view + i, sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(&rect_min, geometryState.rect_min + i, sizeof(uint2), cudaMemcpyDeviceToHost);
		cudaMemcpy(&rect_max, geometryState.rect_max + i, sizeof(uint2), cudaMemcpyDeviceToHost);
		cudaMemcpy(&radii, forwardOutput.radii + i, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "P: " << i << ", ";
		std::cout << "v1: [" << v1.x << ", " << v1.y << ", " << v1.z << "], ";
		std::cout << "v2: [" << v2.x << ", " << v2.y << ", " << v2.z << "], ";
		std::cout << "v3: [" << v3.x << ", " << v3.y << ", " << v3.z << "], ";
		std::cout << "v1_view: [" << v1_view.x << ", " << v1_view.y << ", " << v1_view.z << "], ";
		std::cout << "v2_view: [" << v2_view.x << ", " << v2_view.y << ", " << v2_view.z << "], ";
		std::cout << "v3_view: [" << v3_view.x << ", " << v3_view.y << ", " << v3_view.z << "], ";
		std::cout << "rect_min: [" << rect_min.x << ", " << rect_min.y << "], ";
		std::cout << "rect_max: [" << rect_max.x << ", " << rect_max.y << "], ";
		std::cout << "radii: " << radii << std::endl;
	}
#endif

	// Compute prefix sum over full list of touched tile counts.
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geometryState.scanning_space, geometryState.scan_size, geometryState.tiles_touched, geometryState.point_offsets, P);
	CHECK_CUDA(debug);

	// Retrieve total number of triangle instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geometryState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
	CHECK_CUDA(debug);
	forwardOutput.num_rendered = num_rendered;

	Params::BinningState binningState(forwardOutput.binningBuffer, (size_t)num_rendered, true);

	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated triangle indices to be sorted
	duplicateWithKeys<<<(P + 255) / 256, 256>>>(
		P, grid,
		geometryState.tiles_touched,
		geometryState.rect_min,
		geometryState.rect_max,
		geometryState.depth,
		geometryState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted);
	CHECK_CUDA(debug);

	int bit = getHigherMsb(grid.x * grid.y);
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted,
		binningState.point_list_keys,
		binningState.point_list_unsorted,
		binningState.point_list,
		num_rendered, 0, 32 + bit);
	CHECK_CUDA(debug);

	Params::ImageState imageState(forwardOutput.imageBuffer, (size_t)(W * H), true);

	cudaMemset(imageState.ranges, 0, grid.x * grid.y * sizeof(uint2));
	CHECK_CUDA(debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
	{
		identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(num_rendered, binningState.point_list_keys, imageState.ranges);
		CHECK_CUDA(debug);
	}

#ifdef DEBUG
	// Debugging: print out ranges
	uint2 h_ranges;
	for (int i = 0; i < grid.x * grid.y; i++)
	{
		cudaMemcpy(&h_ranges, imageState.ranges + i, sizeof(uint2), cudaMemcpyDeviceToHost);
		std::cout << "Tile " << i << ", start: " << h_ranges.x << ", end: " << h_ranges.y << std::endl;
	}
#endif

	// Let each tile blend its range of Triangles independently in parallel
	const float *feature = geometryInfo.use_shs ? (float *)geometryState.rgb : geometryInfo.feature;
	FORWARD::renderCUDA<<<grid, block>>>(
		W, H, geometryInfo.C, geometryInfo.gamma, rich_info,
		cameraInfo.tan_fovx,
		cameraInfo.tan_fovy,
		imageState.ranges,
		binningState.point_list,
		geometryState.v1_view,
		geometryState.v2_view,
		geometryState.v3_view,
		geometryState.normal_view,
		feature,
		geometryInfo.opacity,
		geometryInfo.background_depth,
		geometryInfo.background,
		imageState.final_T,
		imageState.n_contrib,
		forwardOutput.out_feature,
		forwardOutput.depth,
		forwardOutput.normal,
		forwardOutput.contrib_sum,
		forwardOutput.contrib_max);
	CHECK_CUDA(debug);
}

void Rasterizer::backward(
	const Params::CameraInfo &cameraInfo,
	const Params::GeometryInfo &geometryInfo,
	const Params::BackwardInput &backwardInput,
	const Params::LossInput &lossInput,
	Params::BackwardOutput &backwardOutput,
	const bool rich_info,
	const bool debug)
{
	const int W = cameraInfo.width;
	const int H = cameraInfo.height;
	const int P = geometryInfo.P;

	const dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	Params::GeometryState geometryState(backwardInput.geometryBuffer, (size_t)P);
	Params::BinningState binningState(backwardInput.binningBuffer, (size_t)backwardInput.num_rendered);
	Params::ImageState imageState(backwardInput.imageBuffer, (size_t)(W * H));

	auto float_opts = backwardInput.geometryBuffer.options().dtype(torch::kFloat32);
	torch::Tensor dL_dv1_view = torch::zeros({P, 3}, float_opts);
	torch::Tensor dL_dv2_view = torch::zeros({P, 3}, float_opts);
	torch::Tensor dL_dv3_view = torch::zeros({P, 3}, float_opts);

	torch::Tensor dL_dnormal_view = torch::empty({0}, float_opts);
	if (rich_info)
	{
		dL_dnormal_view = torch::zeros({P, 3}, float_opts);
	}

	float3 *dL_dv1_view_ptr = (float3 *)dL_dv1_view.data_ptr<float>();
	float3 *dL_dv2_view_ptr = (float3 *)dL_dv2_view.data_ptr<float>();
	float3 *dL_dv3_view_ptr = (float3 *)dL_dv3_view.data_ptr<float>();
	float3 *dL_dnormal_view_ptr = (float3 *)dL_dnormal_view.data_ptr<float>();

	const float *feature = geometryInfo.use_shs ? (float *)geometryState.rgb : geometryInfo.feature;
	BACKWARD::renderCUDA<<<grid, block>>>(
		W, H, geometryInfo.C, geometryInfo.gamma, rich_info,
		cameraInfo.tan_fovx,
		cameraInfo.tan_fovy,
		imageState.ranges,
		binningState.point_list,
		geometryState.v1_view,
		geometryState.v2_view,
		geometryState.v3_view,
		geometryState.normal_view,
		feature,
		geometryInfo.opacity,
		geometryInfo.background_depth,
		geometryInfo.background,
		imageState.final_T,
		imageState.n_contrib,
		lossInput.dL_dout_feature,
		lossInput.dL_dout_depth,
		lossInput.dL_dout_normal,
		dL_dv1_view_ptr,
		dL_dv2_view_ptr,
		dL_dv3_view_ptr,
		dL_dnormal_view_ptr,
		backwardOutput.dL_dfeature,
		backwardOutput.dL_dopacity);
	CHECK_CUDA(debug);

	BACKWARD::preprocessCUDA<<<(P + 255) / 256, 256>>>(
		W, H, P, geometryInfo.D, geometryInfo.M, geometryInfo.use_shs, rich_info,
		cameraInfo.viewmatrix,
		cameraInfo.projmatrix,
		cameraInfo.campos,
		geometryInfo.vertex,
		geometryInfo.shs,
		backwardInput.radii,
		geometryState.v1_view,
		geometryState.v2_view,
		geometryState.v3_view,
		geometryState.clamped,
		dL_dv1_view_ptr,
		dL_dv2_view_ptr,
		dL_dv3_view_ptr,
		dL_dnormal_view_ptr,
		backwardOutput.dL_dfeature,
		backwardOutput.dL_dvertex,
		backwardOutput.dL_dcenter2D,
		backwardOutput.dL_dshs);
	CHECK_CUDA(debug);
}