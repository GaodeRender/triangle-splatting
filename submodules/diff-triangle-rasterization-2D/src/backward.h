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
		float tan_fovx, float tan_fovy,
		const float *__restrict__ viewmatrix,
		const float *__restrict__ projmatrix,
		const float *__restrict__ campos,
		const float *__restrict__ vertex,
		const float *__restrict__ shs,
		const int *__restrict__ radii,
		const bool *__restrict__ clamped,
		const float2 *__restrict__ dL_dv1_2D_ptr,
		const float2 *__restrict__ dL_dv2_2D_ptr,
		const float2 *__restrict__ dL_dv3_2D_ptr,
		const float3 *__restrict__ dL_dnormal_view_ptr,
		const float3 *__restrict__ dL_dv_depth_ptr,
		const float *__restrict__ dL_dfeature,
		float *__restrict__ dL_dvertex,
		float *__restrict__ dL_dcenter2D,
		float *__restrict__ dL_dshs);

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
			const float *__restrict__ final_T,
			const uint32_t *__restrict__ n_contrib,
			const float *__restrict__ dL_dout_feature,
			const float *__restrict__ dL_dout_depth,
			const float *__restrict__ dL_dout_normal,
			float2 *__restrict__ dL_dv1_2D,
			float2 *__restrict__ dL_dv2_2D,
			float2 *__restrict__ dL_dv3_2D,
			float3 *__restrict__ dL_dnormal_view,
			float3 *__restrict__ dL_dv_depth,
			float *__restrict__ dL_dfeature,
			float *__restrict__ dL_dopacity);
}
