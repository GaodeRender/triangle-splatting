#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "forward.h"
#include "auxiliary.h"

// Forward method for converting the input spherical harmonics coefficients to RGB colors.
__device__ float3 computeRGBFromSH(int idx, int deg, int max_coeffs, float3 pos, float3 campos, const float *shs, bool *clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	float3 dir = pos - campos;
	dir /= norm(dir);

	float3 *sh = ((float3 *)shs) + idx * max_coeffs;
	float3 rgb = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		rgb = rgb - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			rgb = rgb +
				  SH_C2[0] * xy * sh[4] +
				  SH_C2[1] * yz * sh[5] +
				  SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				  SH_C2[3] * xz * sh[7] +
				  SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				rgb = rgb +
					  SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					  SH_C3[1] * xy * z * sh[10] +
					  SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					  SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					  SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					  SH_C3[5] * z * (xx - yy) * sh[14] +
					  SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	rgb += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (rgb.x < 0);
	clamped[3 * idx + 1] = (rgb.y < 0);
	clamped[3 * idx + 2] = (rgb.z < 0);
	return max(rgb, 0.0f);
}

__global__ void FORWARD::preprocessCUDA(
	int W, int H, int P, int D, int M, bool rich_info,
	bool use_shs, dim3 grid, bool back_culling,
	const float *__restrict__ viewmatrix,
	const float *__restrict__ projmatrix,
	const float *__restrict__ campos,
	const float *__restrict__ vertex,
	const float *__restrict__ shs,
	int *__restrict__ radii,
	float3 *__restrict__ s_v1_view,
	float3 *__restrict__ s_v2_view,
	float3 *__restrict__ s_v3_view,
	float3 *__restrict__ s_normal_view,
	float *__restrict__ s_depth,
	float3 *__restrict__ s_rgb,
	bool *__restrict__ s_clamped,
	uint32_t *__restrict__ s_tiles_touched,
	uint2 *__restrict__ s_rect_min,
	uint2 *__restrict__ s_rect_max)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	radii[idx] = 0;
	s_tiles_touched[idx] = 0;

	const float3 v1 = {vertex[9 * idx], vertex[9 * idx + 1], vertex[9 * idx + 2]};
	const float3 v2 = {vertex[9 * idx + 3], vertex[9 * idx + 4], vertex[9 * idx + 5]};
	const float3 v3 = {vertex[9 * idx + 6], vertex[9 * idx + 7], vertex[9 * idx + 8]};
	const float3 v1_view = transformPoint4x3(v1, viewmatrix);
	const float3 v2_view = transformPoint4x3(v2, viewmatrix);
	const float3 v3_view = transformPoint4x3(v3, viewmatrix);

	const float3 center_view = (v1_view + v2_view + v3_view) / 3.0f;
	float3 normal_view = cross(v2_view - v1_view, v3_view - v1_view);

	if (norm(normal_view) < EPS) // Skip degenerate triangles
		return;
	if (back_culling && normal_view.z >= 0) // Backface culling
		return;

	// calculate coverage of the triangle in screen space
	const float dilation = 3.0f;
	const float3 center = (v1 + v2 + v3) / 3.0f;
	const float3 v1_dilated = center + dilation * (v1 - center);
	const float3 v2_dilated = center + dilation * (v2 - center);
	const float3 v3_dilated = center + dilation * (v3 - center);
	const float3 v1_dilated_proj = projectPoint(v1_dilated, projmatrix);
	const float3 v2_dilated_proj = projectPoint(v2_dilated, projmatrix);
	const float3 v3_dilated_proj = projectPoint(v3_dilated, projmatrix);

	if (v1_dilated_proj.z <= 0 || v2_dilated_proj.z <= 0 || v3_dilated_proj.z <= 0) // Near culling
		return;
	
	const float2 v1_dilated_2D = {projToPix(v1_dilated_proj.x, W), projToPix(v1_dilated_proj.y, H)};
	const float2 v2_dilated_2D = {projToPix(v2_dilated_proj.x, W), projToPix(v2_dilated_proj.y, H)};
	const float2 v3_dilated_2D = {projToPix(v3_dilated_proj.x, W), projToPix(v3_dilated_proj.y, H)};
	const float2 v_min = min(v1_dilated_2D, v2_dilated_2D, v3_dilated_2D);
	const float2 v_max = max(v1_dilated_2D, v2_dilated_2D, v3_dilated_2D);

	const uint2 rect_min = {min(grid.x, max((int)0, (int)(v_min.x / BLOCK_X))),
							min(grid.y, max((int)0, (int)(v_min.y / BLOCK_Y)))};
	const uint2 rect_max = {min(grid.x, max((int)0, (int)((v_max.x + BLOCK_X - 1) / BLOCK_X))),
							min(grid.y, max((int)0, (int)((v_max.y + BLOCK_Y - 1) / BLOCK_Y)))};
	if (rect_max.x <= rect_min.x || rect_max.y <= rect_min.y)
		return;

	// Compute RGB colors if SH coefficients are provided.
	if (use_shs)
	{
		s_rgb[idx] = computeRGBFromSH(idx, D, M, center, *(float3 *)campos, shs, s_clamped);
	}

	// Store data for the following steps.
	s_v1_view[idx] = v1_view;
	s_v2_view[idx] = v2_view;
	s_v3_view[idx] = v3_view;
	s_normal_view[idx] = normal_view;
	s_depth[idx] = center_view.z;
	s_tiles_touched[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	s_rect_min[idx] = rect_min;
	s_rect_max[idx] = rect_max;
	radii[idx] = max(ceil((v_max.x - v_min.x) * 0.5f), ceil((v_max.y - v_min.y) * 0.5f));
}

// Main rasterization method.
// Collaboratively works on one tile per block, each thread treats one pixel.
// Alternates between fetching and rasterizing data.
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	FORWARD::renderCUDA(
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
		float *__restrict__ final_T,
		uint32_t *__restrict__ n_contrib,
		float *__restrict__ out_feature,
		float *__restrict__ out_depth,
		float *__restrict__ out_normal,
		float *__restrict__ contrib_sum,
		float *__restrict__ contrib_max)
{
	// Identify current tile and pixel.
	auto block = cg::this_thread_block();
	dim3 group_index = block.group_index();
	dim3 thread_index = block.thread_index();
	auto tid = block.thread_rank();

	const uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};

	const bool inside = pix.x < W && pix.y < H; // Check if this thread is associated with a valid pixel or outside.
	bool done = !inside; // Done threads can help with fetching, but don't rasterize
	
	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_v1_view[BLOCK_SIZE];
	__shared__ float3 collected_v2_view[BLOCK_SIZE];
	__shared__ float3 collected_v3_view[BLOCK_SIZE];
	__shared__ float3 collected_normal_view[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];
	__shared__ float collected_feature[BLOCK_SIZE * MAX_CHANNELS];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;

	float accum_feature[MAX_CHANNELS] = {0};
	float3 accum_normal = {0, 0, 0};
	float accum_depth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if the entire block votes that it is done rasterizing
		const int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch data from global to shared
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[tid] = coll_id;
			collected_v1_view[tid] = s_v1_view[coll_id];
			collected_v2_view[tid] = s_v2_view[coll_id];
			collected_v3_view[tid] = s_v3_view[coll_id];
			collected_normal_view[tid] = s_normal_view[coll_id];

			collected_opacity[tid] = opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_feature[tid * C + i] = feature[coll_id * C + i];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			last_contributor = contributor;

			const float3 v1_view = collected_v1_view[j];
			const float3 v2_view = collected_v2_view[j];
			const float3 v3_view = collected_v3_view[j];
			const float3 normal_view = collected_normal_view[j];

			const float p_ray_dot_n = dot(p_ray, normal_view);
			if (abs(p_ray_dot_n) < EPS)
				continue;
			const float depth = dot(v1_view, normal_view) / p_ray_dot_n;
			const float3 p_view = depth * p_ray;

			const float3 p_v1 = v1_view - p_view;
			const float3 p_v2 = v2_view - p_view;
			const float3 p_v3 = v3_view - p_view;

			const float inv_n_dot_n = 1.0f / dot(normal_view, normal_view);
			const float a1 = dot(cross(p_v2, p_v3), normal_view) * inv_n_dot_n;
			const float a2 = dot(cross(p_v3, p_v1), normal_view) * inv_n_dot_n;
			const float a3 = 1.0f - a1 - a2;
			const float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);
			if (ecc < 0.0f || ecc > 10.0f)
				continue;

			const float power = -0.5f * pow(ecc, 2.0f * gamma);
			const float alpha = min(0.99f, collected_opacity[j] * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			
			const float contrib = alpha * T;
			T *= (1.0f - alpha);

			for (int ch = 0; ch < C; ch++)
				accum_feature[ch] += collected_feature[j * C + ch] * contrib;

			if (rich_info)
			{
				const int global_id = collected_id[j];
				atomicAdd(&contrib_sum[global_id], contrib);
				atomicMaxFloat(&contrib_max[global_id], contrib);

				accum_normal += normal_view * contrib;
				accum_depth += depth * contrib;
			}

			if (T <= 0.0001f)
				done = true;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < C; ch++)
			out_feature[ch * H * W + pix_id] = accum_feature[ch] + T * background[ch];
		
		if (rich_info)
		{
			out_depth[pix_id] = accum_depth + T * background_depth;
			out_normal[pix_id] = accum_normal.x;
			out_normal[H * W + pix_id] = accum_normal.y;
			out_normal[2 * H * W + pix_id] = accum_normal.z;
		}
	}
}
