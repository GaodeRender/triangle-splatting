#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "stdio.h"

#define EPS (float)(1e-8)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f};

enum TriangleEccCond
{
	R1R2,
	R2R3,
	R3R1
};

__forceinline__ __device__ float projToPix(float v, int S)
{
	return (v + 1.0f) * S * 0.5f - 0.5f;
}

__forceinline__ __device__ float pixToProj(float v, int S)
{
	return (2.0f * v - S + 1.0f) / (float)(S);
}

__forceinline__ __device__ float3 transformPoint4x3(const float3 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 &p, const float *matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x4Transpose(const float4 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3] * p.w,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7] * p.w,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11] * p.w};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 projectPoint(const float3 &p, const float *projmatrix)
{
	float4 p_hom = transformPoint4x4(p, projmatrix);
	float p_w_inv = 1.0f / (abs(p_hom.w) + EPS); // use abs to make sure p_proj.z for points behind the camera are negative
	float3 p_proj = {p_hom.x * p_w_inv, p_hom.y * p_w_inv, p_hom.z * p_w_inv};
	return p_proj;
}

__forceinline__ __device__ float2 projectVecApprox(const float3 &p_view, const float3 &vec_view, float tan_fovx, float tan_fovy)
{
	/*
		p_view and vec_view are view-space origin and vector.
		We will use a linear expansion of the projection function at p to project the vector.

		We will use the following formulas:
		---------------------------
		p_view = (x, y, z)
		vec_view = (dx, dy, dz)
		p_proj = (x/z/tan_fovx, y/z/tan_fovy)
		vec_proj = (d(x_proj), d(y_proj))
		d(x_proj) = d(x/z/tan_fovx) = dx/z/tan_fovx - dz*x/z^2/tan_fovx = (dx - dz*x/z)/z/tan_fovx
		d(y_proj) = d(y/z/tan_fovy) = dy/z/tan_fovy - dz*y/z^2/tan_fovy = (dy - dz*y/z)/z/tan_fovy
		---------------------------

		x/z and y/z should be clipped by tan_fovx and tan_fovy to avoid large distortions.
	*/
	float2 vec_proj = {(vec_view.x - vec_view.z * p_view.x / p_view.z) / (p_view.z * tan_fovx),
					   (vec_view.y - vec_view.z * p_view.y / p_view.z) / (p_view.z * tan_fovy)};
	return vec_proj;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float2 dnormvdv(float2 v, float2 dv)
{
	float sum2 = v.x * v.x + v.y * v.y;
	float normv = sqrt(sum2);
	float invsum32 = 1.0f / (normv * normv * normv);

	float2 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float normv = sqrt(sum2);
	float invsum32 = 1.0f / (normv * normv * normv);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float normv = sqrt(sum2);
	float invsum32 = 1.0f / (normv * normv * normv);

	float4 vdv = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float cross(const float2 &a, const float2 &b)
{
	return a.x * b.y - a.y * b.x;
}

__forceinline__ __device__ float3 cross(const float3 &a, const float3 &b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ float2 cross(const float2 &a)
{
	return make_float2(a.y, -a.x);
}

__forceinline__ __device__ float2 operator+(const float2 &a, const float2 &b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __device__ float2 operator+(const float2 &a, float b)
{
	return make_float2(a.x + b, a.y + b);
}

__forceinline__ __device__ float2 operator-(const float2 &a, const float2 &b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __device__ float2 operator*(const float2 &a, const float2 &b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}

__forceinline__ __device__ float2 operator*(float a, const float2 &b)
{
	return make_float2(a * b.x, a * b.y);
}

__forceinline__ __device__ float2 operator*(const float2 &a, float b)
{
	return make_float2(a.x * b, a.y * b);
}

__forceinline__ __device__ float2 operator/(const float2 &a, const float2 &b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}

__forceinline__ __device__ float2 operator/(const float2 &a, float b)
{
	return make_float2(a.x / b, a.y / b);
}

__forceinline__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 operator+(const float3 &a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __device__ void operator+=(float3 &a, const float3 &b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__forceinline__ __device__ void operator+=(float3 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

__forceinline__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator-(const float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float3 operator/(const float3 &a, const float3 &b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__forceinline__ __device__ void operator/=(float3 &a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

__forceinline__ __device__ float3 operator*(float a, const float3 &b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__forceinline__ __device__ float3 operator*(const float3 &a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ float3 operator/(const float3 &a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__forceinline__ __device__ float4 operator*(float a, const float4 &b)
{
	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__forceinline__ __device__ float4 operator/(const float4 &a, float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__forceinline__ __device__ float dot(const float2 &a, const float2 &b)
{
	return a.x * b.x + a.y * b.y;
}

__forceinline__ __device__ float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float dot(const float4 &a, const float4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __device__ float norm(const float2 &a)
{
	return sqrt(dot(a, a));
}

__forceinline__ __device__ float norm(const float3 &a)
{
	return sqrt(dot(a, a));
}

__forceinline__ __device__ float2 min(const float2 &a, const float2 &b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

__forceinline__ __device__ float2 min(const float2 &a, const float2 &b, const float2 &c)
{
	return min(min(a, b), c);
}

__forceinline__ __device__ float2 max(const float2 &a, const float2 &b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

__forceinline__ __device__ float2 max(const float2 &a, const float2 &b, const float2 &c)
{
	return max(max(a, b), c);
}

__forceinline__ __device__ float3 max(const float3 &a, float b)
{
	return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

__forceinline__ __device__ float atomicMaxFloat(float *addr, float value)
{
	float old;
	old = !signbit(value) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

#define CHECK_CUDA(debug)                                                                                              \
	if (debug)                                                                                                         \
	{                                                                                                                  \
		auto ret = cudaDeviceSynchronize();                                                                            \
		if (ret != cudaSuccess)                                                                                        \
		{                                                                                                              \
			std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
			throw std::runtime_error(cudaGetErrorString(ret));                                                         \
		}                                                                                                              \
	}
