#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include "param_struct.h"

namespace Rasterizer
{
	void forward(
		const Params::CameraInfo &cameraInfo,
		const Params::GeometryInfo &geometryInfo,
		Params::ForwardOutput &forwardOutput,
		const bool back_culling,
		const bool rich_info,
		const bool debug);

	void backward(
		const Params::CameraInfo &cameraInfo,
		const Params::GeometryInfo &geometryInfo,
		const Params::BackwardInput &backwardInput,
		const Params::LossInput &lossInput,
		Params::BackwardOutput &backwardOutput,
		const bool rich_info,
		const bool debug);
};