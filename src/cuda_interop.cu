#include "cuda_interop.h"
#include <iostream>
#include <unistd.h>
#include <cstring>

// CUDA kernel for NV12 to RGBA conversion
__global__ void nv12ToRgbaKernel(const uint8_t* yPlane, const uint8_t* uvPlane,
                                  uint8_t* rgbaOutput, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Get Y value
    int yIndex = y * width + x;
    float Y = yPlane[yIndex];

    // Get UV values (NV12: UV plane has interleaved U/V pairs)
    // For each 2x2 block of Y pixels, there's one U/V pair
    // UV plane has same width as Y plane (because of U/V interleaving)
    // but half the height
    int uvY = y / 2;
    int uvX = (x & ~1);  // Round x down to even number (same as (x/2)*2)
    int uvIndex = uvY * width + uvX;
    float U = uvPlane[uvIndex];
    float V = uvPlane[uvIndex + 1];

    // YUV to RGB conversion (BT.709)
    Y = (Y - 16.0f) * 1.164f;
    U = U - 128.0f;
    V = V - 128.0f;

    float R = Y + 1.793f * V;
    float G = Y - 0.213f * U - 0.533f * V;
    float B = Y + 2.112f * U;

    // Clamp to [0, 255]
    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    // Write RGBA output
    int rgbaIndex = (y * width + x) * 4;
    rgbaOutput[rgbaIndex + 0] = (uint8_t)R;
    rgbaOutput[rgbaIndex + 1] = (uint8_t)G;
    rgbaOutput[rgbaIndex + 2] = (uint8_t)B;
    rgbaOutput[rgbaIndex + 3] = 255;  // Alpha
}

bool cudaInteropInit(VulkanCudaInterop* interop, VkDevice device, VkPhysicalDevice physicalDevice) {
    std::cout << "Initializing CUDA interop..." << std::endl;

    memset(interop, 0, sizeof(VulkanCudaInterop));
    interop->device = device;
    interop->physicalDevice = physicalDevice;
    interop->memoryFd = -1;

    // Initialize CUDA driver API
    CU_CHECK(cuInit(0));

    // Get CUDA device
    CU_CHECK(cuDeviceGet(&interop->cuDevice, 0));

    // Create CUDA context
    CU_CHECK(cuCtxCreate(&interop->cuContext, 0, interop->cuDevice));

    // Create CUDA stream for async operations
    CU_CHECK(cuStreamCreate(&interop->cuStream, CU_STREAM_DEFAULT));

    interop->initialized = true;
    std::cout << "CUDA interop initialized successfully" << std::endl;

    return true;
}

bool cudaInteropCreateImage(VulkanCudaInterop* interop, uint32_t width, uint32_t height) {
    std::cout << "Creating interop-capable Vulkan image (" << width << "x" << height << ")..." << std::endl;

    interop->width = width;
    interop->height = height;

    // Create image with external memory export
    VkExternalMemoryImageCreateInfo externalMemInfo = {};
    externalMemInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalMemInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.pNext = &externalMemInfo;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;  // LINEAR for CUDA compatibility
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(interop->device, &imageInfo, nullptr, &interop->image) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create Vulkan image with external memory" << std::endl;
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(interop->device, interop->image, &memRequirements);

    // Find memory type that is device-local
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(interop->physicalDevice, &memProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX) {
        std::cerr << "ERROR: Failed to find suitable memory type" << std::endl;
        return false;
    }

    // Allocate memory with external memory export
    VkExportMemoryAllocateInfo exportInfo = {};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(interop->device, &allocInfo, nullptr, &interop->imageMemory) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to allocate Vulkan memory with external export" << std::endl;
        return false;
    }

    // Bind image memory
    if (vkBindImageMemory(interop->device, interop->image, interop->imageMemory, 0) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to bind image memory" << std::endl;
        return false;
    }

    // Create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = interop->image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(interop->device, &viewInfo, nullptr, &interop->imageView) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create image view" << std::endl;
        return false;
    }

    // Create sampler for texture sampling in shaders
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(interop->device, &samplerInfo, nullptr, &interop->sampler) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create sampler" << std::endl;
        return false;
    }

    std::cout << "Vulkan image created successfully" << std::endl;
    return true;
}

bool cudaInteropImportMemory(VulkanCudaInterop* interop) {
    std::cout << "Importing Vulkan memory into CUDA..." << std::endl;

    // Get file descriptor from Vulkan memory
    VkMemoryGetFdInfoKHR getFdInfo = {};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = interop->imageMemory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    // Get the function pointer for vkGetMemoryFdKHR
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(interop->device, "vkGetMemoryFdKHR");

    if (!vkGetMemoryFdKHR) {
        std::cerr << "ERROR: vkGetMemoryFdKHR not available" << std::endl;
        return false;
    }

    if (vkGetMemoryFdKHR(interop->device, &getFdInfo, &interop->memoryFd) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to get memory file descriptor" << std::endl;
        return false;
    }

    std::cout << "Got memory FD: " << interop->memoryFd << std::endl;

    // Import into CUDA
    size_t memorySize = interop->width * interop->height * 4;  // RGBA

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memHandleDesc = {};
    memHandleDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    memHandleDesc.handle.fd = interop->memoryFd;
    memHandleDesc.size = memorySize;

    CU_CHECK(cuImportExternalMemory(&interop->cuExtMem, &memHandleDesc));

    // Map the external memory to a CUDA device pointer
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = memorySize;

    CU_CHECK(cuExternalMemoryGetMappedBuffer(&interop->cuDevicePtr, interop->cuExtMem, &bufferDesc));

    std::cout << "CUDA memory imported successfully (ptr: " << std::hex << interop->cuDevicePtr << std::dec << ")" << std::endl;

    return true;
}

bool cudaConvertNV12ToRGBA(VulkanCudaInterop* interop, const uint8_t* nv12Data, size_t nv12Size) {
    // Allocate temporary GPU buffers for NV12 data
    uint8_t* d_nv12;
    CUDA_CHECK(cudaMalloc(&d_nv12, nv12Size));

    // Copy NV12 data to GPU
    CUDA_CHECK(cudaMemcpy(d_nv12, nv12Data, nv12Size, cudaMemcpyHostToDevice));

    // Split NV12 data into Y and UV planes
    int yPlaneSize = interop->width * interop->height;
    uint8_t* d_yPlane = d_nv12;
    uint8_t* d_uvPlane = d_nv12 + yPlaneSize;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((interop->width + blockSize.x - 1) / blockSize.x,
                  (interop->height + blockSize.y - 1) / blockSize.y);

    nv12ToRgbaKernel<<<gridSize, blockSize>>>(
        d_yPlane, d_uvPlane,
        (uint8_t*)interop->cuDevicePtr,
        interop->width, interop->height
    );

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    // Free temporary buffer
    CUDA_CHECK(cudaFree(d_nv12));

    return true;
}

void cudaInteropDestroy(VulkanCudaInterop* interop) {
    if (!interop->initialized) return;

    std::cout << "Cleaning up CUDA interop..." << std::endl;

    if (interop->cuDevicePtr) {
        // The mapped buffer is automatically freed when external memory is destroyed
        interop->cuDevicePtr = 0;
    }

    if (interop->cuExtMem) {
        cuDestroyExternalMemory(interop->cuExtMem);
        interop->cuExtMem = nullptr;
    }

    if (interop->cuStream) {
        cuStreamDestroy(interop->cuStream);
        interop->cuStream = nullptr;
    }

    if (interop->cuContext) {
        cuCtxDestroy(interop->cuContext);
        interop->cuContext = nullptr;
    }

    if (interop->memoryFd >= 0) {
        close(interop->memoryFd);
        interop->memoryFd = -1;
    }

    if (interop->imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(interop->device, interop->imageView, nullptr);
        interop->imageView = VK_NULL_HANDLE;
    }

    if (interop->image != VK_NULL_HANDLE) {
        vkDestroyImage(interop->device, interop->image, nullptr);
        interop->image = VK_NULL_HANDLE;
    }

    if (interop->imageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(interop->device, interop->imageMemory, nullptr);
        interop->imageMemory = VK_NULL_HANDLE;
    }

    interop->initialized = false;
    std::cout << "CUDA interop cleanup complete" << std::endl;
}
