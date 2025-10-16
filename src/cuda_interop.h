#pragma once

#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            return false; \
        } \
    } while(0)

#define CU_CHECK(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CUDA Driver Error: %s at %s:%d\n", \
                    errStr, __FILE__, __LINE__); \
            return false; \
        } \
    } while(0)

// Structure to hold Vulkan-CUDA interop state
struct VulkanCudaInterop {
    // Vulkan objects
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkDeviceMemory imageMemory;
    VkImage image;
    VkImageView imageView;
    uint32_t width;
    uint32_t height;

    // CUDA objects
    CUdevice cuDevice;
    CUcontext cuContext;
    CUexternalMemory cuExtMem;
    CUdeviceptr cuDevicePtr;
    CUstream cuStream;

    // State
    bool initialized;
    int memoryFd;  // File descriptor for external memory
};

// Initialize CUDA context
bool cudaInteropInit(VulkanCudaInterop* interop, VkDevice device, VkPhysicalDevice physicalDevice);

// Create Vulkan image with external memory that can be shared with CUDA
bool cudaInteropCreateImage(VulkanCudaInterop* interop, uint32_t width, uint32_t height);

// Import the Vulkan image memory into CUDA
bool cudaInteropImportMemory(VulkanCudaInterop* interop);

// Convert NV12 to RGBA on GPU using CUDA kernel
// nv12Data: pointer to NV12 data (Y plane followed by UV plane)
// rgbaOutput: pointer to output RGBA buffer (the imported Vulkan image memory)
bool cudaConvertNV12ToRGBA(VulkanCudaInterop* interop, const uint8_t* nv12Data, size_t nv12Size);

// Cleanup
void cudaInteropDestroy(VulkanCudaInterop* interop);
