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
#include <iostream>
#include <float.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include "device_launch_parameters.h"
// #define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "simple_knn.h"
#include "auxiliary.h"

struct CustomMin
{
	__device__ __forceinline__ float3 operator()(const float3 &a, const float3 &b) const
	{
		return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
	}
};

struct CustomMax
{
	__device__ __forceinline__ float3 operator()(const float3 &a, const float3 &b) const
	{
		return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
	}
};

__device__ __forceinline__ float dist2(const float3 &a, const float3 &b)
{
	float3 d = {a.x - b.x, a.y - b.y, a.z - b.z};
	return d.x * d.x + d.y * d.y + d.z * d.z;
}

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(const float3 &coord, const float3 &minn, const float3 &maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3 *points, float3 minn, float3 maxx, uint32_t *codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(int P, const float3 *points, const uint32_t *indices, MinMax *boxes)
{
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = {FLT_MAX, FLT_MAX, FLT_MAX};
		me.maxx = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
	}

	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax &box, const float3 &p)
{
	float3 diff = {0, 0, 0};
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template <int K>
__device__ void updateKBest(const float3 &ref, const float3 &point, float *knn)
{
	float dist = dist2(ref, point);
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

__global__ void boxMeanDist(int P, const float3 *points, const uint32_t *indices, const MinMax *boxes, float *dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

__global__ void boxNearestNeighbor(int P, int batch_size, const float3 *points, const uint32_t *indices, const MinMax *boxes, uint32_t *indices_nearest)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	uint32_t c_idx = indices[idx];
	float3 point = points[c_idx];
	float best = FLT_MAX;
	float dist;
	uint32_t nearest, ii;

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		ii = indices[i];
		if (ii / batch_size == c_idx / batch_size)
			continue;
		dist = dist2(point, points[ii]);
		if (dist < best)
		{
			best = dist;
			nearest = ii;
		}
	}

	float reject = best;
	best = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		dist = distBoxPoint(box, point);
		if (dist > reject || dist > best)
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			ii = indices[i];
			if (ii / batch_size == c_idx / batch_size)
				continue;
			dist = dist2(point, points[ii]);
			if (dist < best)
			{
				best = dist;
				nearest = ii;
			}
		}
	}

	indices_nearest[c_idx] = nearest;
}

void SimpleKNN::knn(int P, const float3 *points, float *meanDists)
{
	size_t temp_storage_bytes;
	float3 init = {0, 0, 0}, minn, maxx;

	// calculate points bounding box
	float3 *result;
	cudaMalloc(&result, sizeof(float3));
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	CHECK_CUDA();

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(result);
	CHECK_CUDA();

	// calculate morton codes
	thrust::device_vector<uint32_t> morton(P);
	coord2Morton<<<(P + 255) / 256, 256>>>(P, points, minn, maxx, morton.data().get());
	CHECK_CUDA();

	// sort morton codes
	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);
	thrust::device_vector<uint32_t> morton_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	// calculate subgroups bounding boxes
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax<<<num_boxes, BOX_SIZE>>>(P, points, indices_sorted.data().get(), boxes.data().get());
	CHECK_CUDA();

	// calculate knn distances
	boxMeanDist<<<num_boxes, BOX_SIZE>>>(P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);
	CHECK_CUDA();
}

void SimpleKNN::nearestNeighbor(int P, int batch_size, const float3 *points, uint32_t *indices_nearest)
{
	size_t temp_storage_bytes;
	float3 init = {0, 0, 0}, minn, maxx;

	// calculate points bounding box
	float3 *result;
	cudaMalloc(&result, sizeof(float3));
	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);
	CHECK_CUDA();

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(result);
	CHECK_CUDA();

	// calculate morton codes
	thrust::device_vector<uint32_t> morton(P);
	coord2Morton<<<(P + 255) / 256, 256>>>(P, points, minn, maxx, morton.data().get());
	CHECK_CUDA();

	// sort morton codes
	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);
	thrust::device_vector<uint32_t> morton_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	// calculate subgroups bounding boxes
	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax<<<num_boxes, BOX_SIZE>>>(P, points, indices_sorted.data().get(), boxes.data().get());
	CHECK_CUDA();

	// calculate nearest neighbors
	boxNearestNeighbor<<<num_boxes, BOX_SIZE>>>(P, batch_size, points, indices_sorted.data().get(), boxes.data().get(), indices_nearest);
	CHECK_CUDA();
}
