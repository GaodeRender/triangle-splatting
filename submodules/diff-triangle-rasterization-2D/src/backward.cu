#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "backward.h"
#include "auxiliary.h"

// Backward method for converting the input spherical harmonics coefficients to RGB colors.
__device__ void computeRGBFromSHBackward(int idx, int deg, int max_coeffs, float3 pos, float3 campos, const float *shs, const bool *clamped,
										 const float3 *dL_dfeature, float3 *dL_dshs, float3 &dL_dpos)
{
	// Compute intermediate values, as it is done during forward
	float3 dir_orig = pos - campos;
	float3 dir = dir_orig / norm(dir_orig);

	float3 *sh = ((float3 *)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied, gradient becomes 0.
	float3 dL_dRGB = dL_dfeature[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	float3 dRGBdx = {0, 0, 0};
	float3 dRGBdy = {0, 0, 0};
	float3 dRGBdz = {0, 0, 0};
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location to write SH gradients to
	float3 *dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation.
	// View direction is influenced by the triangle center,
	// so SHs gradients must propagate back into 3D position.
	float3 dL_ddir = {dot(dL_dRGB, dRGBdx), dot(dL_dRGB, dRGBdy), dot(dL_dRGB, dRGBdz)};

	// Account for normalization of direction
	dL_dpos = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});
}

__device__ void projectPointBackward(const float3 &p, const float *projmatrix, const float3 &dL_dp_proj, float3 &dL_dp)
{
	const float4 p_hom = transformPoint4x4(p, projmatrix);
	const float p_w_inv = 1.0f / (abs(p_hom.w) + EPS);
	const float3 p_proj = {p_hom.x * p_w_inv, p_hom.y * p_w_inv, p_hom.z * p_w_inv};

	const float4 dL_dp_hom = abs(p_w_inv) * make_float4(dL_dp_proj.x, dL_dp_proj.y, dL_dp_proj.z, -dot(dL_dp_proj, p_proj));
	dL_dp = transformPoint4x4Transpose(dL_dp_hom, projmatrix);
}

__device__ void projectVecApproxBackward(const float3 &p_view, const float3 &vec_view, float tan_fovx, float tan_fovy, 
										 const float2 &dL_dvec_proj, float3 &dL_dp_view, float3 &dL_dvec_view)
{
	const float px_pz = p_view.x / p_view.z;
	const float py_pz = p_view.y / p_view.z;
	const float vx_pz = vec_view.x / p_view.z;
	const float vy_pz = vec_view.y / p_view.z;
	const float vz_pz = vec_view.z / p_view.z;
	const float2 dL_dvec = {dL_dvec_proj.x / (p_view.z * tan_fovx), dL_dvec_proj.y / (p_view.z * tan_fovy)};
	dL_dvec_view = {dL_dvec.x, dL_dvec.y, -dL_dvec.x * px_pz - dL_dvec.y * py_pz};
	dL_dp_view = {-dL_dvec.x * vz_pz, -dL_dvec.y * vz_pz, dL_dvec.x * (2.0f * vz_pz * px_pz - vx_pz) + dL_dvec.y * (2.0f * vz_pz * py_pz - vy_pz)};
}

__global__ void BACKWARD::preprocessCUDA(
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
	float *__restrict__ dL_dshs)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || radii[idx] <= 0)
		return;

	const float3 v1 = {vertex[9 * idx], vertex[9 * idx + 1], vertex[9 * idx + 2]};
	const float3 v2 = {vertex[9 * idx + 3], vertex[9 * idx + 4], vertex[9 * idx + 5]};
	const float3 v3 = {vertex[9 * idx + 6], vertex[9 * idx + 7], vertex[9 * idx + 8]};
	const float3 center = (v1 + v2 + v3) / 3.0f;

	const float3 center_view = transformPoint4x3(center, viewmatrix);
	const float limx = 1.3f * tan_fovx * center_view.z;
	const float limy = 1.3f * tan_fovy * center_view.z;
	const float3 center_view_clipped = {min(max(-limx, center_view.x), limx), min(max(-limy, center_view.y), limy), center_view.z};

	const float3 r1 = v1 - center;
	const float3 r2 = v2 - center;
	const float3 r3 = v3 - center;
	const float3 r1_view = transformVec4x3(r1, viewmatrix);
	const float3 r2_view = transformVec4x3(r2, viewmatrix);
	const float3 r3_view = transformVec4x3(r3, viewmatrix);
	const float2 r1_proj = projectVecApprox(center_view_clipped, r1_view, tan_fovx, tan_fovy);
	const float2 r2_proj = projectVecApprox(center_view_clipped, r2_view, tan_fovx, tan_fovy);
	const float2 r3_proj = projectVecApprox(center_view_clipped, r3_view, tan_fovx, tan_fovy);

	const float2 dL_dv1_2D = dL_dv1_2D_ptr[idx];
	const float2 dL_dv2_2D = dL_dv2_2D_ptr[idx];
	const float2 dL_dv3_2D = dL_dv3_2D_ptr[idx];
	const float2 dL_dcenter_2D = dL_dv1_2D + dL_dv2_2D + dL_dv3_2D;

	const float2 scaling = {0.5f * W, 0.5f * H};
	const float kernel_size = 0.5f;

	const float2 dL_dr1_proj = scaling * dL_dv1_2D + kernel_size * dnormvdv(r1_proj, dL_dv1_2D);
	const float2 dL_dr2_proj = scaling * dL_dv2_2D + kernel_size * dnormvdv(r2_proj, dL_dv2_2D);
	const float2 dL_dr3_proj = scaling * dL_dv3_2D + kernel_size * dnormvdv(r3_proj, dL_dv3_2D);
	const float2 dL_dcenter_proj = scaling * dL_dcenter_2D;

	float3 dL_dr1_view, dL_dr2_view, dL_dr3_view, dL_dcenter_view_clipped;
	float3 dL_dcenter_view = {0, 0, 0};
	projectVecApproxBackward(center_view_clipped, r1_view, tan_fovx, tan_fovy, dL_dr1_proj, dL_dcenter_view_clipped, dL_dr1_view);
	dL_dcenter_view += dL_dcenter_view_clipped;
	projectVecApproxBackward(center_view_clipped, r2_view, tan_fovx, tan_fovy, dL_dr2_proj, dL_dcenter_view_clipped, dL_dr2_view);
	dL_dcenter_view += dL_dcenter_view_clipped;
	projectVecApproxBackward(center_view_clipped, r3_view, tan_fovx, tan_fovy, dL_dr3_proj, dL_dcenter_view_clipped, dL_dr3_view);
	dL_dcenter_view += dL_dcenter_view_clipped;
	if (center_view.x < -limx || center_view.x > limx)
	{
		dL_dcenter_view.x = 0;
	}
	if (center_view.y < -limy || center_view.y > limy)
	{
		dL_dcenter_view.y = 0;
	}

	if (rich_info)
	{
		const float3 dL_dnormal_view = dL_dnormal_view_ptr[idx];
		const float3 dL_dv_depth = dL_dv_depth_ptr[idx];
		const float3 r1_view_cross_r2_view = cross(r1_view, r2_view);
		const float3 dL_dr1_view_cross_r2_view = dnormvdv(r1_view_cross_r2_view, dL_dnormal_view);
		dL_dr1_view += cross(r2_view, dL_dr1_view_cross_r2_view) + make_float3(0, 0, dL_dv_depth.x);
		dL_dr2_view += cross(dL_dr1_view_cross_r2_view, r1_view) + make_float3(0, 0, dL_dv_depth.y);
		dL_dr3_view += make_float3(0, 0, dL_dv_depth.z);
		dL_dcenter_view += make_float3(0, 0, dL_dv_depth.x + dL_dv_depth.y + dL_dv_depth.z);
	}

	float3 dL_dcenter;
	projectPointBackward(center, projmatrix, make_float3(dL_dcenter_proj.x, dL_dcenter_proj.y, 0), dL_dcenter);
	dL_dcenter += transformVec4x3Transpose(dL_dcenter_view, viewmatrix);

	const float3 dL_dr1 = transformVec4x3Transpose(dL_dr1_view, viewmatrix);
	const float3 dL_dr2 = transformVec4x3Transpose(dL_dr2_view, viewmatrix);
	const float3 dL_dr3 = transformVec4x3Transpose(dL_dr3_view, viewmatrix);

	// Compute gradient updates due to computing RGB from SHs
	if (use_shs)
	{
		float3 dL_dcenter_sh;
		computeRGBFromSHBackward(idx, D, M, center, *(float3 *)campos, shs, clamped,
								 (const float3 *)dL_dfeature, (float3 *)dL_dshs, dL_dcenter_sh);
		dL_dcenter += dL_dcenter_sh;
	}

	const float3 dL_dv1 = (2 * dL_dr1 - dL_dr2 - dL_dr3 + dL_dcenter) / 3.0f;
	const float3 dL_dv2 = (2 * dL_dr2 - dL_dr1 - dL_dr3 + dL_dcenter) / 3.0f;
	const float3 dL_dv3 = (2 * dL_dr3 - dL_dr1 - dL_dr2 + dL_dcenter) / 3.0f;

	dL_dvertex[9 * idx] = dL_dv1.x;
	dL_dvertex[9 * idx + 1] = dL_dv1.y;
	dL_dvertex[9 * idx + 2] = dL_dv1.z;
	dL_dvertex[9 * idx + 3] = dL_dv2.x;
	dL_dvertex[9 * idx + 4] = dL_dv2.y;
	dL_dvertex[9 * idx + 5] = dL_dv2.z;
	dL_dvertex[9 * idx + 6] = dL_dv3.x;
	dL_dvertex[9 * idx + 7] = dL_dv3.y;
	dL_dvertex[9 * idx + 8] = dL_dv3.z;

	dL_dcenter2D[2 * idx] = dL_dcenter_2D.x;
	dL_dcenter2D[2 * idx + 1] = dL_dcenter_2D.y;
}

__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	BACKWARD::renderCUDA(
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
		float *__restrict__ dL_dopacity)
{
	auto block = cg::this_thread_block();
	dim3 group_index = block.group_index();
	dim3 thread_index = block.thread_index();
	auto tid = block.thread_rank();

	const uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
	const uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_v1_2D[BLOCK_SIZE];
	__shared__ float2 collected_v2_2D[BLOCK_SIZE];
	__shared__ float2 collected_v3_2D[BLOCK_SIZE];
	__shared__ float collected_area2[BLOCK_SIZE];
	__shared__ float3 collected_normal_view[BLOCK_SIZE];
	__shared__ float3 collected_v_depth[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];
	__shared__ float collected_feature[BLOCK_SIZE * MAX_CHANNELS];

	const float T_final = inside ? final_T[pix_id] : 0;
	float T = T_final;
	const uint32_t last_contributor = inside ? n_contrib[pix_id] : 0;
	uint32_t contributor = toDo;

	float accum_feature[MAX_CHANNELS] = {0}; // Accumulated feature from back to front
	float3 accum_normal = {0, 0, 0};		 // Accumulated normal from back to front
	float accum_depth = background_depth;	 // Accumulated depth from back to front
	
	float dL_dfeature_pixel[MAX_CHANNELS] = {0};
	float3 dL_dnormal_pixel = {0, 0, 0};
	float dL_ddepth_pixel = 0;

	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			accum_feature[i] = background[i];
			dL_dfeature_pixel[i] = dL_dout_feature[i * H * W + pix_id];
		}
		if (rich_info)
		{
			dL_dnormal_pixel = make_float3(dL_dout_normal[pix_id], dL_dout_normal[W * H + pix_id], dL_dout_normal[2 * W * H + pix_id]);
			dL_ddepth_pixel = dL_dout_depth[pix_id];
		}
	}

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory,
		// start from the BACK and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_v1_2D[tid] = s_v1_2D[coll_id];
			collected_v2_2D[tid] = s_v2_2D[coll_id];
			collected_v3_2D[tid] = s_v3_2D[coll_id];
			collected_area2[tid] = s_area2[coll_id];

			if (rich_info)
			{
				collected_normal_view[tid] = s_normal_view[coll_id];
				collected_v_depth[tid] = s_v_depth[coll_id];
			}

			collected_opacity[tid] = opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_feature[tid * C + i] = feature[coll_id * C + i];
		}
		block.sync();

		// Iterate over triangles in the batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current triangle ID.
			// Skip if this one is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float area2 = collected_area2[j];
			const float2 v1_2D = collected_v1_2D[j];
			const float2 v2_2D = collected_v2_2D[j];
			const float2 v3_2D = collected_v3_2D[j];
			const float2 p_v1 = v1_2D - pixf;
			const float2 p_v2 = v2_2D - pixf;
			const float2 p_v3 = v3_2D - pixf;
			const float a1 = cross(p_v2, p_v3) / area2;
			const float a2 = cross(p_v3, p_v1) / area2;
			const float a3 = 1.0f - a1 - a2;
			const float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);
			if (ecc < 0.0f || ecc > 10.0f)
				continue;

			const float power = -0.5f * pow(ecc, 2.0f * gamma);
			const float op = collected_opacity[j];
			const float G = exp(power);
			const float alpha = min(0.99f, op * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T /= (1.0f - alpha);
			const float contrib = alpha * T;

			// Propagate gradients to per-triangle feature and alpha
			float dL_dcontrib = 0.0f;
			float3 dL_da = {0, 0, 0};
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				atomicAdd(&(dL_dfeature[global_id * C + ch]), dL_dfeature_pixel[ch] * contrib);

				const float feat = collected_feature[j * C + ch];
				dL_dcontrib += dL_dfeature_pixel[ch] * (feat - accum_feature[ch]);
				accum_feature[ch] = alpha * feat + (1.0f - alpha) * accum_feature[ch];
			}

			if (rich_info)
			{
				atomicAdd(&dL_dnormal_view[global_id].x, dL_dnormal_pixel.x * contrib);
				atomicAdd(&dL_dnormal_view[global_id].y, dL_dnormal_pixel.y * contrib);
				atomicAdd(&dL_dnormal_view[global_id].z, dL_dnormal_pixel.z * contrib);
				const float3 normal = collected_normal_view[j];
				dL_dcontrib += dot(dL_dnormal_pixel, normal - accum_normal);
				accum_normal = alpha * normal + (1.0f - alpha) * accum_normal;

				const float dL_ddepth = dL_ddepth_pixel * contrib;
				atomicAdd(&dL_dv_depth[global_id].x, dL_ddepth * a1);
				atomicAdd(&dL_dv_depth[global_id].y, dL_ddepth * a2);
				atomicAdd(&dL_dv_depth[global_id].z, dL_ddepth * a3);
				const float3 v_depth = collected_v_depth[j];
				dL_da += dL_ddepth * v_depth;
				const float depth = v_depth.x * a1 + v_depth.y * a2 + v_depth.z * a3;
				dL_dcontrib += dL_ddepth_pixel * (depth - accum_depth);
				accum_depth = alpha * depth + (1.0f - alpha) * accum_depth;
			}

			const float dL_dalpha = dL_dcontrib * T;

			// Helpful reusable temporary variables
			float dL_dpower = 0.0f;
			if (op * G < 0.99f) // if not clamped by min(0.99, op * G)
			{
				dL_dpower = dL_dalpha * alpha;
			}
			const float dL_decc = dL_dpower * 2 * gamma * power / (ecc + EPS);

			float3 decc_da = {0, 0, 0};
			if (a1 <= a2 && a1 <= a3)
			{
				decc_da.x = -3.0f;
			}
			else if (a2 <= a1 && a2 <= a3)
			{
				decc_da.y = -3.0f;
			}
			else
			{
				decc_da.z = -3.0f;
			}
			dL_da += dL_decc * decc_da;

			const float2 v1_v2 = v2_2D - v1_2D;
			const float2 v2_v3 = v3_2D - v2_2D;
			const float2 v3_v1 = v1_2D - v3_2D;
			const float area2_inv = 1.0f / area2;
			const float2 da1_dv1_2D = cross(v2_v3 * a1) * area2_inv;
			const float2 da1_dv2_2D = cross(v3_v1 * a1 + p_v3) * area2_inv;
			const float2 da1_dv3_2D = cross(v1_v2 * a1 - p_v2) * area2_inv;
			const float2 da2_dv1_2D = cross(v2_v3 * a2 - p_v3) * area2_inv;
			const float2 da2_dv2_2D = cross(v3_v1 * a2) * area2_inv;
			const float2 da2_dv3_2D = cross(v1_v2 * a2 + p_v1) * area2_inv;
			const float2 da3_dv1_2D = cross(v2_v3 * a3 + p_v2) * area2_inv;
			const float2 da3_dv2_2D = cross(v3_v1 * a3 - p_v1) * area2_inv;
			const float2 da3_dv3_2D = cross(v1_v2 * a3) * area2_inv;
			const float2 dL_dv1_2D_point = dL_da.x * da1_dv1_2D + dL_da.y * da2_dv1_2D + dL_da.z * da3_dv1_2D;
			const float2 dL_dv2_2D_point = dL_da.x * da1_dv2_2D + dL_da.y * da2_dv2_2D + dL_da.z * da3_dv2_2D;
			const float2 dL_dv3_2D_point = dL_da.x * da1_dv3_2D + dL_da.y * da2_dv3_2D + dL_da.z * da3_dv3_2D;

			// Update gradients w.r.t. triangle vertex positions
			atomicAdd(&dL_dv1_2D[global_id].x, dL_dv1_2D_point.x);
			atomicAdd(&dL_dv1_2D[global_id].y, dL_dv1_2D_point.y);
			atomicAdd(&dL_dv2_2D[global_id].x, dL_dv2_2D_point.x);
			atomicAdd(&dL_dv2_2D[global_id].y, dL_dv2_2D_point.y);
			atomicAdd(&dL_dv3_2D[global_id].x, dL_dv3_2D_point.x);
			atomicAdd(&dL_dv3_2D[global_id].y, dL_dv3_2D_point.y);

			// Update gradients w.r.t. triangle opacity
			atomicAdd(&(dL_dopacity[global_id]), dL_dalpha * G);
		}
	}
}
