#pragma once

#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace Params
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	class BaseDataBuffer{
	public:
		BaseDataBuffer() {}
		~BaseDataBuffer() {}
		
		virtual void fromChunk(char*& chunk, size_t data_size) = 0;

		void fromTensor(torch::Tensor t, size_t data_size, bool resize = false) {
			if (resize){
				size_t required_size = requiredSize(data_size);
				t.resize_({(long long) required_size});
				t.fill_(0);
			}
            char *chunk = reinterpret_cast<char *>(t.contiguous().data_ptr());
			fromChunk(chunk, data_size);
		}
		
		size_t requiredSize(size_t data_size) {
			char* size = nullptr;
			fromChunk(size, data_size);
			return ((size_t)size) + 128;
		}
	};

	class GeometryState: public BaseDataBuffer
	{
	public:
		float3* v1_view;
		float3* v2_view;
		float3* v3_view;
		float3* normal_view;
		float* depth;
		float3* rgb;
		bool* clamped;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;
        uint2* rect_min;
        uint2* rect_max;
		size_t scan_size;
		char* scanning_space;

        GeometryState(torch::Tensor t, size_t data_size, bool resize = false) {
			fromTensor(t, data_size, resize);
		}

		void fromChunk(char*& chunk, size_t P) {
			obtain(chunk, v1_view, P, 128);
			obtain(chunk, v2_view, P, 128);
			obtain(chunk, v3_view, P, 128);
			obtain(chunk, normal_view, P, 128);
			obtain(chunk, depth, P, 128);
			obtain(chunk, rgb, P, 128);
			obtain(chunk, clamped, P * 3, 128);
			obtain(chunk, point_offsets, P, 128);
			obtain(chunk, tiles_touched, P, 128);
            obtain(chunk, rect_min, P, 128);
            obtain(chunk, rect_max, P, 128);
			cub::DeviceScan::InclusiveSum(nullptr, scan_size, tiles_touched, tiles_touched, P);
			obtain(chunk, scanning_space, scan_size, 128);
		}
	};

	class ImageState: public BaseDataBuffer
	{
	public:
		uint2* ranges;
		uint32_t* n_contrib;
		float* final_T;

        ImageState(torch::Tensor t, size_t data_size, bool resize = false) {
			fromTensor(t, data_size, resize);
		}

		void fromChunk(char*& chunk, size_t N) {
			obtain(chunk, ranges, N, 128);
			obtain(chunk, n_contrib, N, 128);
			obtain(chunk, final_T, N, 128);
		}
	};

	class BinningState: public BaseDataBuffer
	{
	public:
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		size_t sorting_size;
		char* list_sorting_space;
        
        BinningState(torch::Tensor t, size_t data_size, bool resize = false) {
			fromTensor(t, data_size, resize);
		}

		void fromChunk(char*& chunk, size_t P) {
			obtain(chunk, point_list_keys_unsorted, P, 128);
			obtain(chunk, point_list_keys, P, 128);
			obtain(chunk, point_list_unsorted, P, 128);
			obtain(chunk, point_list, P, 128);
			cub::DeviceRadixSort::SortPairs(nullptr, sorting_size, point_list_keys_unsorted, point_list_keys, point_list_unsorted, point_list, P);
			obtain(chunk, list_sorting_space, sorting_size, 128);
		}
	};
	
	struct CameraInfo
	{
		const int width;
		const int height;
		const float tan_fovx;
		const float tan_fovy;

		const float *viewmatrix;
		const float *projmatrix;
		const float *campos;
	};

	struct GeometryInfo
	{
		const int P; // Number of points
		const int D; // Number of sh degrees
		const int M; // Number of sh coefficients
        const int C; // Number of feature channels
        const bool use_shs;
		const float gamma;
		const float scale_modifier;
		const float background_depth;

		const float *background;
		const float *vertex;
		const float *shs;
		const float *feature;
		const float *opacity;
	};

	struct ForwardOutput
	{
		int num_rendered;
		float *out_feature;
		int *radii;
		float *depth;
		float *normal;
		float *contrib_sum;
		float *contrib_max;
		torch::Tensor geometryBuffer;
		torch::Tensor binningBuffer;
		torch::Tensor imageBuffer;
	};

    struct BackwardInput
    {
        const int num_rendered;
        const int *radii;
        const torch::Tensor geometryBuffer;
        const torch::Tensor binningBuffer;
        const torch::Tensor imageBuffer;
    };
	
	struct LossInput
	{
		const float *dL_dout_feature;
		const float *dL_dout_depth;
		const float *dL_dout_normal;
	};
	
	struct BackwardOutput
	{
		float *dL_dvertex;
		float *dL_dcenter2D;
		float *dL_dshs;
		float *dL_dfeature;
		float *dL_dopacity;
	};
};