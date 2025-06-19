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
#pragma once

#include "cuda_runtime.h"

namespace SimpleKNN
{
	void knn(int P, const float3* points, float* meanDists);

	void nearestNeighbor(int P, int batch_size, const float3* points, uint32_t* indices_nearest);
};
