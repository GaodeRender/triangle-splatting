#include "interface.h"
#include <torch/torch.h>
#include <iostream>

template <typename T>
void printCudaTensor(const torch::Tensor &tensor, int range_min=0, int range_max=10)
{
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Tensor of size: " << tensor.sizes() << std::endl;
    std::cout << "Tensor of type: " << tensor.dtype() << std::endl;
    std::cout << "Tensor on device: " << tensor.device() << std::endl;

    std::cout << "Tensor values: " << std::endl;
    T *tensor_data = tensor.data_ptr<T>();
    T h_data, h_min, h_max;
    for (int i = 0; i < tensor.numel(); i++)
    {
        cudaMemcpy(&h_data, tensor_data, sizeof(T), cudaMemcpyDeviceToHost);
        if (range_min <= i && i < range_max)
        {
            std::cout << h_data << ", ";
        }

        if (i == 0)
        {
            h_min = h_data;
            h_max = h_data;
        }
        if (h_data < h_min)
        {
            h_min = h_data;
        }
        if (h_data > h_max)
        {
            h_max = h_data;
        }
        tensor_data++;
    }
    std::cout << "..." << std::endl;
    std::cout << "Tensor min: " << h_min << std::endl;
    std::cout << "Tensor max: " << h_max << std::endl;

    // std::cout << "Tensor values: " << tensor << std::endl;
    // std::cout << "Tensor min: " << tensor.min() << std::endl;
    // std::cout << "Tensor max: " << tensor.max() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

int main()
{
    torch::Device device(torch::kCUDA, 0);
    torch::manual_seed(42);

    // torch::Tensor points = torch::rand({10, 3}).to(device);
    torch::Tensor points = torch::tensor({{0.0, 0.0, 0.1},
                                          {0.5, 0.0, 0.0},
                                          {0.0, 1.0, 0.0},
                                          {0.0, 3.0, 0.1},
                                          {0.5, 3.0, 0.0},
                                          {0.0, 4.0, 0.0},
                                          {3.0, 0.0, 0.1},
                                          {3.5, 0.0, 0.0},
                                          {3.0, 1.0, 0.0}}).to(device);
    std::cout << "point num: " << points.size(0) << std::endl;

    std::cout << "Calling distCUDA2" << std::endl;
    torch::Tensor distance = distCUDA2(points);
    printCudaTensor<float>(distance);
    std::cout << "mean distance: " << distance.mean().item<float>() << std::endl;

    std::cout << "Calling nearestNeighbor" << std::endl;
    torch::Tensor nearest_neighbor = nearestNeighbor(points, 3);
    printCudaTensor<uint32_t>(nearest_neighbor);

    std::cout << "Finished" << std::endl;
    return 0;
}