#include "src/extension_interface.h"
#include <torch/torch.h>
#include <iostream>

int main()
{
    torch::Device device(torch::kCUDA, 0);
    torch::manual_seed(42);

    int image_width = 2160;
    int image_height = 1440;
    float tan_fovx = 0.3148;
    float tan_fovy = 0.21;
    torch::Tensor viewmatrix = torch::tensor({{-1.0, 0.0, 0.0, 0.0},
                                              {0.0, 1.0, 0.0, 0.0},
                                              {0.0, 0.0, -1.0, 0.0},
                                              {0.0, 0.0, 1200.0, 1.0}}).to(device);
    torch::Tensor projmatrix = torch::tensor({{-3.1764, 0.0, 0.0, 0.0},
                                              {0.0, 4.7617, 0.0, 0.0},
                                              {0.0, 0.0, -1.0, -1.0},
                                              {0.0, 0.0, 1200.0, 1200.0}}).to(device);
    torch::Tensor campos = torch::tensor({0.0, 0.0, 1200.0}).to(device);
    int sh_degree = 3;
    float gamma = 1.0;
    float scale_modifier = 1.0;
    uint32_t N = 1000;
    float background_depth = 5000.0;
    torch::Tensor background = torch::zeros({3}).to(device);
    torch::Tensor vertex = torch::rand({N, 3, 3}).to(device) * torch::tensor({{1200.0, 600.0, 200.0}}).to(device) - torch::tensor({{600.0, 300.0, 0.0}}).to(device);
    // torch::Tensor shs = torch::zeros({0}).to(device);
    torch::Tensor shs = torch::rand({N, 16, 3}).to(device);
    // torch::Tensor feature = torch::rand({N, 3}).to(device);
    torch::Tensor feature = torch::zeros({0}).to(device);
    torch::Tensor opacity = torch::rand({N, 1}).to(device);
    bool back_culling = false;
    bool rich_info = true;
    bool debug = true;

    std::cout << "Calling rasterizeTrianglesForward" << std::endl;

    auto forward_result = rasterizeTrianglesForward(
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        viewmatrix,
        projmatrix,
        campos,
        sh_degree,
        gamma,
        scale_modifier,
        background_depth,
        background,
        vertex,
        shs,
        feature,
        opacity,
        back_culling,
        rich_info,
        debug);
    
    int num_rendered;
    torch::Tensor out_feature;
    torch::Tensor radii;
    torch::Tensor depth;
    torch::Tensor normal;
    torch::Tensor contrib_sum;
    torch::Tensor contrib_max;
    torch::Tensor geometryBuffer;
    torch::Tensor binningBuffer;
    torch::Tensor imageBuffer;
    std::tie(num_rendered, out_feature, radii, depth, normal, contrib_sum, contrib_max, geometryBuffer, binningBuffer, imageBuffer) = forward_result;

    std::cout << "num_rendered: " << num_rendered << std::endl;
    std::cout << "Calling rasterizeTrianglesBackward" << std::endl;
    
    torch::Tensor dL_dout_feature = torch::rand({3, image_height, image_width}).to(device);
    torch::Tensor dL_dout_depth = torch::rand({image_height, image_width}).to(device);
    torch::Tensor dL_dout_normal = torch::rand({3, image_height, image_width}).to(device);

    auto backward_result = rasterizeTrianglesBackward(
        tan_fovx,
        tan_fovy,
        viewmatrix,
        projmatrix,
        campos,
        sh_degree,
        gamma,
        scale_modifier,
        background_depth,
        background,
        vertex,
        shs,
        feature,
        opacity,
        num_rendered,
        radii,
        geometryBuffer,
        binningBuffer,
        imageBuffer,
        dL_dout_feature,
        dL_dout_depth,
        dL_dout_normal,
        rich_info,
        debug);
    
    torch::Tensor dL_dvertex;
    torch::Tensor dL_dcenter2D;
    torch::Tensor dL_dshs;
    torch::Tensor dL_dfeature;
    torch::Tensor dL_dopacity;
    std::tie(dL_dvertex, dL_dcenter2D, dL_dshs, dL_dfeature, dL_dopacity) = backward_result;

    float vertex_grad_mean = dL_dvertex.abs().mean().item<float>();
    float vertex_grad_norm = dL_dvertex.norm().item<float>();
    float vertex_grad_std = dL_dvertex.std().item<float>();
    std::cout << "vertex_grad_mean: " << vertex_grad_mean << std::endl;
    std::cout << "vertex_grad_norm: " << vertex_grad_norm << std::endl;
    std::cout << "vertex_grad_std: " << vertex_grad_std << std::endl;
    std::cout << "Finished" << std::endl;

    return 0;
}