#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "param_struct.h"
#include "config.h"

namespace FORWARD
{
	__global__ void preprocessCUDA(
		int W, int H, int P, int D, int M, bool rich_info,
		bool use_shs, dim3 grid, bool back_culling,
		float tan_fovx, float tan_fovy,
		const float *__restrict__ viewmatrix,
		const float *__restrict__ projmatrix,
		const float *__restrict__ campos,
		const float *__restrict__ vertex,
		const float *__restrict__ shs,
		int *__restrict__ radii,
		float2 *__restrict__ s_v1_2D,
		float2 *__restrict__ s_v2_2D,
		float2 *__restrict__ s_v3_2D,
		float *__restrict__ s_area2,
		float3 *__restrict__ s_normal_view,
		float3 *__restrict__ s_v_depth,
		float *__restrict__ s_depth,
		float *__restrict__ s_rgb,
		bool *__restrict__ s_clamped,
		uint32_t *__restrict__ s_tiles_touched,
		uint2 *__restrict__ s_rect_min,
		uint2 *__restrict__ s_rect_max);

	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
		renderCUDA(
			int W, int H, int C, float gamma, bool rich_info,
			const uint2 *__restrict__ ranges,
			const uint32_t *__restrict__ point_list,
			const float2 *__restrict__ s_v1_2D,
			const float2 *__restrict__ s_v2_2D,
			const float2 *__restrict__ s_v3_2D,
			const float *__restrict__ s_area2,
			const float3 *__restrict__ s_normal_view,
			const float3 *__restrict__ s_v_depth,
			const float *__restrict__ feature,
			const float *__restrict__ opacity,
			const float background_depth,
			const float *__restrict__ background,
			float *__restrict__ final_T,
			uint32_t *__restrict__ n_contrib,
			float *__restrict__ out_feature,
			float *__restrict__ out_depth,
			float *__restrict__ out_normal,
			float *__restrict__ contrib_sum,
			float *__restrict__ contrib_max);
}
