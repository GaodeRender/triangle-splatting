#pragma once

#define BOX_SIZE 1024
// #define DEBUG

#ifdef DEBUG
#define CHECK_CUDA()                                                                                                   \
    {                                                                                                                  \
        auto ret = cudaDeviceSynchronize();                                                                            \
        if (ret != cudaSuccess)                                                                                        \
        {                                                                                                              \
            std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
            throw std::runtime_error(cudaGetErrorString(ret));                                                         \
        }                                                                                                              \
    }
#else
#define CHECK_CUDA()
#endif