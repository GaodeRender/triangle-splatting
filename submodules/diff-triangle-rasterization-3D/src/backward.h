#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "param_struct.h"
#include "config.h"

namespace BACKWARD
{
	__global__ void preprocessCUDA(
		int W, int H, int P, int D, int M, bool use_shs, bool rich_info,
		const float *__restrict__ viewmatrix,
		const float *__restrict__ projmatrix,
		const float *__restrict__ campos,
		const float *__restrict__ vertex,
		const float *__restrict__ shs,
		const int *__restrict__ radii,
		const float3 *__restrict__ s_v1_view,
		const float3 *__restrict__ s_v2_view,
		const float3 *__restrict__ s_v3_view,
		const bool *__restrict__ clamped,
		const float3 *__restrict__ dL_dv1_view_ptr,
		const float3 *__restrict__ dL_dv2_view_ptr,
		const float3 *__restrict__ dL_dv3_view_ptr,
		const float3 *__restrict__ dL_dnormal_view_ptr,
		const float *__restrict__ dL_dfeature,
		float *__restrict__ dL_dvertex,
		float *__restrict__ dL_dcenter2D,
		float *__restrict__ dL_dshs);

	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
		renderCUDA(
			int W, int H, int C, float gamma, bool rich_info,
			float tan_fovx, float tan_fovy,
			const uint2 *__restrict__ ranges,
			const uint32_t *__restrict__ point_list,
			const float3 *__restrict__ s_v1_view,
			const float3 *__restrict__ s_v2_view,
			const float3 *__restrict__ s_v3_view,
			const float3 *__restrict__ s_normal_view,
			const float *__restrict__ feature,
			const float *__restrict__ opacity,
			const float background_depth,
			const float *__restrict__ background,
			const float *__restrict__ final_T,
			const uint32_t *__restrict__ n_contrib,
			const float *__restrict__ dL_dout_feature,
			const float *__restrict__ dL_dout_depth,
			const float *__restrict__ dL_dout_normal,
			float3 *__restrict__ dL_dv1_view,
			float3 *__restrict__ dL_dv2_view,
			float3 *__restrict__ dL_dv3_view,
			float3 *__restrict__ dL_dnormal_view,
			float *__restrict__ dL_dfeature,
			float *__restrict__ dL_dopacity);
}
