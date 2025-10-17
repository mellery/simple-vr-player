#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cmath>

// FFmpeg includes
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

// CUDA interop
#include "cuda_interop.h"

// Math utilities for 3D rendering
#include "math_utils.h"

// Helper macro for OpenXR error checking
#define XR_CHECK(result, msg) \
    if (XR_FAILED(result)) { \
        std::cerr << "ERROR: " << msg << " (result: " << result << ")" << std::endl; \
        return false; \
    }

#define VK_CHECK(result, msg) \
    if (result != VK_SUCCESS) { \
        std::cerr << "ERROR: " << msg << " (result: " << result << ")" << std::endl; \
        return false; \
    }

class VideoDecoder {
public:
    VideoDecoder() = default;
    ~VideoDecoder() { close(); }

    bool open(const char* filename);
    bool readFrame(uint8_t* nv12Buffer, int* width, int* height, size_t* nv12Size);  // Now outputs NV12
    void close();

    int getWidth() const { return codecCtx ? codecCtx->width : 0; }
    int getHeight() const { return codecCtx ? codecCtx->height : 0; }
    double getDuration() const;

private:
    AVFormatContext* formatCtx = nullptr;
    AVCodecContext* codecCtx = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* frameRGB = nullptr;
    AVFrame* swFrame = nullptr;  // Software frame for hw->sw transfer
    SwsContext* swsCtx = nullptr;
    AVPacket* packet = nullptr;
    AVBufferRef* hwDeviceCtx = nullptr;  // Hardware device context
    int videoStreamIndex = -1;
    bool useHardwareDecode = false;  // Track if hardware decode is active
};

bool VideoDecoder::open(const char* filename) {
    std::cout << "Opening video file: " << filename << std::endl;

    // Open video file
    int ret = avformat_open_input(&formatCtx, filename, nullptr, nullptr);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "ERROR: Could not open video file: " << errbuf << std::endl;
        return false;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(formatCtx, nullptr) < 0) {
        std::cerr << "ERROR: Could not find stream information" << std::endl;
        return false;
    }

    // Find the first video stream
    videoStreamIndex = -1;
    for (unsigned i = 0; i < formatCtx->nb_streams; i++) {
        if (formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            break;
        }
    }

    if (videoStreamIndex == -1) {
        std::cerr << "ERROR: Could not find video stream" << std::endl;
        return false;
    }

    // Get codec parameters
    AVCodecParameters* codecParams = formatCtx->streams[videoStreamIndex]->codecpar;

    // Try hardware decoder first (h264_cuvid for NVIDIA)
    const AVCodec* codec = nullptr;
    if (codecParams->codec_id == AV_CODEC_ID_H264) {
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        if (codec) {
            std::cout << "Found h264_cuvid hardware decoder, attempting to use NVDEC" << std::endl;
            useHardwareDecode = true;
        } else {
            std::cout << "h264_cuvid not available, falling back to software decode" << std::endl;
        }
    }

    // Fallback to software decoder
    if (!codec) {
        codec = avcodec_find_decoder(codecParams->codec_id);
        if (!codec) {
            std::cerr << "ERROR: Unsupported codec" << std::endl;
            return false;
        }
        useHardwareDecode = false;
    }

    // Allocate codec context
    codecCtx = avcodec_alloc_context3(codec);
    if (!codecCtx) {
        std::cerr << "ERROR: Could not allocate codec context" << std::endl;
        return false;
    }

    // Copy codec parameters to context
    if (avcodec_parameters_to_context(codecCtx, codecParams) < 0) {
        std::cerr << "ERROR: Could not copy codec parameters" << std::endl;
        return false;
    }

    // Set up hardware acceleration if using NVDEC
    if (useHardwareDecode) {
        int ret = av_hwdevice_ctx_create(&hwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
        if (ret < 0) {
            std::cerr << "WARNING: Failed to create CUDA device context, falling back to software decode" << std::endl;
            useHardwareDecode = false;

            // Recreate codec context with software decoder
            avcodec_free_context(&codecCtx);
            codec = avcodec_find_decoder(codecParams->codec_id);
            codecCtx = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(codecCtx, codecParams);
        } else {
            codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx);
            std::cout << "Hardware acceleration enabled (CUDA/NVDEC)" << std::endl;
        }
    }

    // Open codec
    if (avcodec_open2(codecCtx, codec, nullptr) < 0) {
        std::cerr << "ERROR: Could not open codec" << std::endl;
        return false;
    }

    // Allocate frames
    frame = av_frame_alloc();
    frameRGB = av_frame_alloc();
    packet = av_packet_alloc();

    if (!frame || !frameRGB || !packet) {
        std::cerr << "ERROR: Could not allocate frames" << std::endl;
        return false;
    }

    // Allocate software frame for hardware decode transfer
    if (useHardwareDecode) {
        swFrame = av_frame_alloc();
        if (!swFrame) {
            std::cerr << "ERROR: Could not allocate software frame" << std::endl;
            return false;
        }
    }

    std::cout << "Video opened successfully:" << std::endl;
    std::cout << "  Resolution: " << codecCtx->width << "x" << codecCtx->height << std::endl;
    std::cout << "  Codec: " << codec->name << std::endl;
    std::cout << "  Decoder: " << (useHardwareDecode ? "NVDEC (hardware)" : "software") << std::endl;
    std::cout << "  Duration: " << getDuration() << " seconds" << std::endl;

    return true;
}

bool VideoDecoder::readFrame(uint8_t* nv12Buffer, int* width, int* height, size_t* nv12Size) {
    while (av_read_frame(formatCtx, packet) >= 0) {
        if (packet->stream_index == videoStreamIndex) {
            // Send packet to decoder
            int ret = avcodec_send_packet(codecCtx, packet);
            if (ret < 0) {
                av_packet_unref(packet);
                continue;
            }

            // Receive decoded frame
            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_packet_unref(packet);
                continue;
            } else if (ret < 0) {
                av_packet_unref(packet);
                return false;
            }

            // If hardware decode, transfer frame from GPU to CPU
            AVFrame* sourceFrame = frame;
            if (useHardwareDecode && frame->format == AV_PIX_FMT_CUDA) {
                // Transfer hardware frame to software frame (NV12 format)
                swFrame->format = AV_PIX_FMT_NV12;
                ret = av_hwframe_transfer_data(swFrame, frame, 0);
                if (ret < 0) {
                    std::cerr << "ERROR: Failed to transfer frame from GPU to CPU" << std::endl;
                    av_packet_unref(packet);
                    return false;
                }
                sourceFrame = swFrame;
            } else {
                // Software decode - convert to NV12 if needed
                if (sourceFrame->format != AV_PIX_FMT_NV12) {
                    if (!swsCtx) {
                        swsCtx = sws_getContext(
                            codecCtx->width, codecCtx->height,
                            (AVPixelFormat)sourceFrame->format,
                            codecCtx->width, codecCtx->height, AV_PIX_FMT_NV12,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                    }
                    if (swsCtx) {
                        swFrame->format = AV_PIX_FMT_NV12;
                        swFrame->width = codecCtx->width;
                        swFrame->height = codecCtx->height;
                        av_frame_get_buffer(swFrame, 0);

                        sws_scale(swsCtx, sourceFrame->data, sourceFrame->linesize, 0, codecCtx->height,
                                 swFrame->data, swFrame->linesize);
                        sourceFrame = swFrame;
                    }
                }
            }

            // Copy NV12 data (Y plane + UV plane) accounting for FFmpeg linesize/stride
            int yPlaneSize = codecCtx->width * codecCtx->height;
            int uvPlaneSize = yPlaneSize / 2;  // UV plane is half size
            *nv12Size = yPlaneSize + uvPlaneSize;

            // Debug: Print linesize info on first frame
            static bool firstFrame = true;
            if (firstFrame) {
                std::cout << "NV12 Debug Info:" << std::endl;
                std::cout << "  Frame size: " << codecCtx->width << "x" << codecCtx->height << std::endl;
                std::cout << "  Y linesize: " << sourceFrame->linesize[0] << " (expected: " << codecCtx->width << ")" << std::endl;
                std::cout << "  UV linesize: " << sourceFrame->linesize[1] << " (expected: " << codecCtx->width << ")" << std::endl;
                std::cout << "  Format: " << av_get_pix_fmt_name((AVPixelFormat)sourceFrame->format) << std::endl;
                firstFrame = false;
            }

            // Copy Y plane row by row (accounting for stride/padding)
            uint8_t* yDst = nv12Buffer;
            const uint8_t* ySrc = sourceFrame->data[0];
            int yStride = sourceFrame->linesize[0];
            for (int row = 0; row < codecCtx->height; row++) {
                memcpy(yDst, ySrc, codecCtx->width);
                yDst += codecCtx->width;
                ySrc += yStride;
            }

            // Copy UV plane row by row (accounting for stride/padding)
            uint8_t* uvDst = nv12Buffer + yPlaneSize;
            const uint8_t* uvSrc = sourceFrame->data[1];
            int uvStride = sourceFrame->linesize[1];
            int uvHeight = codecCtx->height / 2;
            for (int row = 0; row < uvHeight; row++) {
                memcpy(uvDst, uvSrc, codecCtx->width);  // UV width is same as Y width (interleaved U/V)
                uvDst += codecCtx->width;
                uvSrc += uvStride;
            }

            *width = codecCtx->width;
            *height = codecCtx->height;

            av_packet_unref(packet);
            return true;
        }
        av_packet_unref(packet);
    }

    return false;  // End of file
}

double VideoDecoder::getDuration() const {
    if (!formatCtx || videoStreamIndex < 0)
        return 0.0;

    AVStream* stream = formatCtx->streams[videoStreamIndex];
    return (double)stream->duration * av_q2d(stream->time_base);
}

void VideoDecoder::close() {
    if (swsCtx) {
        sws_freeContext(swsCtx);
        swsCtx = nullptr;
    }

    if (swFrame) {
        av_frame_free(&swFrame);
    }

    if (frameRGB) {
        av_frame_free(&frameRGB);
    }

    if (frame) {
        av_frame_free(&frame);
    }

    if (packet) {
        av_packet_free(&packet);
    }

    if (codecCtx) {
        avcodec_free_context(&codecCtx);
    }

    if (hwDeviceCtx) {
        av_buffer_unref(&hwDeviceCtx);
    }

    if (formatCtx) {
        avformat_close_input(&formatCtx);
    }

    useHardwareDecode = false;
}

class DynamicTexture {
public:
    DynamicTexture() = default;
    ~DynamicTexture() { destroy(); }

    bool create(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool cmdPool, uint32_t width, uint32_t height);
    void update(VkDevice device, VkQueue queue, const uint8_t* rgbaData, uint32_t width, uint32_t height);  // Now expects RGBA
    void waitForUpload(VkDevice device);  // Wait for async upload to complete
    void destroy();

    VkImage getImage() const { return image; }
    VkImageView getImageView() const { return imageView; }
    VkSampler getSampler() const { return sampler; }

private:
    VkDevice device = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    size_t stagingBufferSize = 0;

    VkFence uploadFence = VK_NULL_HANDLE;  // Fence for async uploads
    VkCommandBuffer uploadCmdBuffer = VK_NULL_HANDLE;  // Pre-allocated command buffer for uploads

    uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

uint32_t DynamicTexture::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    std::cerr << "ERROR: Failed to find suitable memory type" << std::endl;
    return 0;
}

bool DynamicTexture::create(VkDevice dev, VkPhysicalDevice physicalDevice, VkCommandPool cmdPool, uint32_t width, uint32_t height) {
    device = dev;
    size_t imageSize = width * height * 4;  // RGBA (4 bytes per pixel for better GPU support)

    // Create staging buffer
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create staging buffer" << std::endl;
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to allocate staging buffer memory" << std::endl;
        return false;
    }

    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);
    stagingBufferSize = imageSize;

    // Create image (use RGBA format for better GPU compatibility)
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create image" << std::endl;
        return false;
    }

    vkGetImageMemoryRequirements(device, image, &memRequirements);

    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to allocate image memory" << std::endl;
        return false;
    }

    vkBindImageMemory(device, image, imageMemory, 0);

    // Create image view
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create image view" << std::endl;
        return false;
    }

    // Create sampler
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
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

    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create sampler" << std::endl;
        return false;
    }

    // Create fence for async uploads
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled

    if (vkCreateFence(device, &fenceInfo, nullptr, &uploadFence) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create upload fence" << std::endl;
        return false;
    }

    // Pre-allocate command buffer for uploads (avoids per-frame allocation)
    VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &uploadCmdBuffer) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to allocate upload command buffer" << std::endl;
        return false;
    }

    return true;
}

void DynamicTexture::update(VkDevice device, VkQueue queue, const uint8_t* rgbaData, uint32_t width, uint32_t height) {
    size_t imageSize = width * height * 4;

    // Direct RGBA copy to staging buffer (no conversion needed!)
    void* data;
    vkMapMemory(device, stagingMemory, 0, imageSize, 0, &data);
    memcpy(data, rgbaData, imageSize);  // Simple memcpy - much faster!
    vkUnmapMemory(device, stagingMemory);

    // Wait for previous upload to complete before resetting command buffer
    vkWaitForFences(device, 1, &uploadFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &uploadFence);

    // Reset and begin command buffer (reuse pre-allocated buffer)
    vkResetCommandBuffer(uploadCmdBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(uploadCmdBuffer, &beginInfo);

    // Transition image to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(uploadCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(uploadCmdBuffer, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition image to SHADER_READ_ONLY_OPTIMAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(uploadCmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(uploadCmdBuffer);

    // Submit with fence (ASYNC - doesn't wait!)
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &uploadCmdBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, uploadFence);  // Signal fence when done

    // DON'T wait here - let GPU work async!
    // Command buffer will be reused next frame (no free needed!)
}

void DynamicTexture::waitForUpload(VkDevice device) {
    // Wait for async upload to complete
    if (uploadFence != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &uploadFence, VK_TRUE, UINT64_MAX);
    }
}

void DynamicTexture::destroy() {
    if (device == VK_NULL_HANDLE) return;

    // Wait for any pending uploads before destroying
    if (uploadFence != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &uploadFence, VK_TRUE, UINT64_MAX);
        vkDestroyFence(device, uploadFence, nullptr);
    }

    // Command buffer will be freed when command pool is destroyed
    // (no manual free needed for pre-allocated buffer)

    if (sampler != VK_NULL_HANDLE) vkDestroySampler(device, sampler, nullptr);
    if (imageView != VK_NULL_HANDLE) vkDestroyImageView(device, imageView, nullptr);
    if (image != VK_NULL_HANDLE) vkDestroyImage(device, image, nullptr);
    if (imageMemory != VK_NULL_HANDLE) vkFreeMemory(device, imageMemory, nullptr);
    if (stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, stagingBuffer, nullptr);
    if (stagingMemory != VK_NULL_HANDLE) vkFreeMemory(device, stagingMemory, nullptr);

    device = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    imageMemory = VK_NULL_HANDLE;
    imageView = VK_NULL_HANDLE;
    sampler = VK_NULL_HANDLE;
    stagingBuffer = VK_NULL_HANDLE;
    stagingMemory = VK_NULL_HANDLE;
    uploadFence = VK_NULL_HANDLE;
    uploadCmdBuffer = VK_NULL_HANDLE;
}

class SimpleVRPlayer {
public:
    // Rendering mode enum
    enum VideoMode {
        MODE_FLAT,      // Flat quad (default)
        MODE_SPHERE_180, // 180° hemisphere
        MODE_SPHERE_360  // 360° full sphere
    };

    bool initialize();
    bool loadVideo(const char* filename);
    void setSBSMode(bool enabled) { sbsMode = enabled; }
    void setDebugMode(bool enabled) { debugMode = enabled; }
    void setVideoMode(VideoMode mode) { videoMode = mode; }
    void run();
    void shutdown();

private:
    // OpenXR objects
    XrInstance xrInstance = XR_NULL_HANDLE;
    XrSystemId xrSystemId = XR_NULL_SYSTEM_ID;
    XrSession xrSession = XR_NULL_HANDLE;
    XrSpace xrPlaySpace = XR_NULL_HANDLE;

    std::vector<XrSwapchain> xrSwapchains;
    std::vector<uint32_t> swapchainLengths;
    std::vector<std::vector<XrSwapchainImageVulkanKHR>> swapchainImages;

    uint32_t viewCount = 0;
    std::vector<XrViewConfigurationView> viewConfigs;
    std::vector<XrView> views;
    std::vector<XrCompositionLayerProjectionView> projectionViews;

    // Vulkan objects
    VkInstance vkInstance = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice vkDevice = VK_NULL_HANDLE;
    VkQueue vkQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;

    // Vulkan rendering resources
    VkCommandPool vkCommandPool = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;

    // Graphics pipeline resources for 3D rendering
    VkRenderPass vkRenderPass = VK_NULL_HANDLE;
    VkPipelineLayout vkPipelineLayout = VK_NULL_HANDLE;
    VkPipeline vkGraphicsPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout vkDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool vkDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet vkDescriptorSet = VK_NULL_HANDLE;

    // Quad geometry
    VkBuffer vkVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vkVertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer vkIndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vkIndexBufferMemory = VK_NULL_HANDLE;
    uint32_t indexCount = 0;

    // Framebuffers and image views (one per eye per swapchain image)
    std::vector<std::vector<VkImageView>> vkSwapchainImageViews;  // [eyeIndex][imageIndex]
    std::vector<std::vector<VkFramebuffer>> vkFramebuffers;  // [eyeIndex][imageIndex]

    // Video playback resources
    VideoDecoder videoDecoder;
    VulkanCudaInterop cudaInterop = {};
    std::vector<uint8_t> frameBuffer;
    bool videoLoaded = false;
    bool sbsMode = false;  // Side-by-side 3D mode
    bool debugMode = false;  // Debug/performance mode
    VideoMode videoMode = MODE_FLAT;

    // Performance tracking
    std::chrono::steady_clock::time_point lastFPSPrint;
    uint64_t frameCount = 0;
    double totalDecodeTime = 0.0;
    double totalUploadTime = 0.0;
    double totalRenderTime = 0.0;
    double totalRenderEyeTime = 0.0;  // NEW: Track renderEye time
    double totalOpenXRTime = 0.0;     // NEW: Track OpenXR overhead

    // Helper functions
    bool createXrInstance();
    bool createXrSystem();
    bool createVulkanInstance();
    bool createVulkanDevice();
    bool createXrSession();
    bool createXrSpace();
    bool createSwapchains();
    bool createRenderingResources();

    // 3D rendering helpers
    std::vector<char> loadShaderFile(const char* filename);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    bool createRenderPass();
    bool createGraphicsPipeline();
    bool createDescriptorSetLayout();
    bool createDescriptorPool();
    bool createDescriptorSet();
    void updateDescriptorSet();  // Update descriptor to point to video texture
    bool createQuadGeometry();
    bool createSphereGeometry(int segments, float radius, float angleHorizontal, float angleVertical);
    bool createSphereGeometryPerEye(int eyeIndex, int segments, float radius, float angleHorizontal, float angleVertical);
    bool createFramebuffers();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void renderEye(uint32_t eyeIndex, uint32_t imageIndex, VkImage image, uint32_t width, uint32_t height);
    void processEvents(bool* exitRequested, bool* sessionRunning);
    void renderFrame();
};

bool SimpleVRPlayer::createXrInstance() {
    std::cout << "Creating OpenXR instance..." << std::endl;

    const char* extensions[] = {
        XR_KHR_VULKAN_ENABLE_EXTENSION_NAME,
    };

    XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
    strncpy(createInfo.applicationInfo.applicationName, "Simple VR Player", XR_MAX_APPLICATION_NAME_SIZE);
    createInfo.applicationInfo.applicationVersion = 1;
    strncpy(createInfo.applicationInfo.engineName, "Custom", XR_MAX_ENGINE_NAME_SIZE);
    createInfo.applicationInfo.engineVersion = 1;
    createInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    createInfo.enabledExtensionCount = 1;
    createInfo.enabledExtensionNames = extensions;

    XrResult result = xrCreateInstance(&createInfo, &xrInstance);
    XR_CHECK(result, "Failed to create OpenXR instance");

    std::cout << "OpenXR instance created successfully" << std::endl;
    return true;
}

bool SimpleVRPlayer::createXrSystem() {
    std::cout << "Getting OpenXR system..." << std::endl;

    XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
    systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    XrResult result = xrGetSystem(xrInstance, &systemInfo, &xrSystemId);
    XR_CHECK(result, "Failed to get OpenXR system");

    std::cout << "OpenXR system ID: " << xrSystemId << std::endl;

    // Get view configuration
    result = xrEnumerateViewConfigurationViews(xrInstance, xrSystemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr);
    XR_CHECK(result, "Failed to get view count");

    viewConfigs.resize(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    result = xrEnumerateViewConfigurationViews(xrInstance, xrSystemId,
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, viewConfigs.data());
    XR_CHECK(result, "Failed to enumerate view configurations");

    std::cout << "View count: " << viewCount << std::endl;
    for (uint32_t i = 0; i < viewCount; i++) {
        std::cout << "  View " << i << ": "
                  << viewConfigs[i].recommendedImageRectWidth << "x"
                  << viewConfigs[i].recommendedImageRectHeight << std::endl;
    }

    return true;
}

bool SimpleVRPlayer::createVulkanInstance() {
    std::cout << "Creating Vulkan instance..." << std::endl;

    // Get required Vulkan instance extensions from OpenXR
    PFN_xrGetVulkanInstanceExtensionsKHR pfnGetVulkanInstanceExtensionsKHR;
    xrGetInstanceProcAddr(xrInstance, "xrGetVulkanInstanceExtensionsKHR",
        (PFN_xrVoidFunction*)&pfnGetVulkanInstanceExtensionsKHR);

    uint32_t extCount = 0;
    pfnGetVulkanInstanceExtensionsKHR(xrInstance, xrSystemId, 0, &extCount, nullptr);

    std::vector<char> extBuffer(extCount);
    pfnGetVulkanInstanceExtensionsKHR(xrInstance, xrSystemId, extCount, &extCount, extBuffer.data());

    // Parse space-separated extension names
    std::vector<const char*> extensions;
    char* token = strtok(extBuffer.data(), " ");
    while (token != nullptr) {
        extensions.push_back(token);
        token = strtok(nullptr, " ");
    }

    std::cout << "Required Vulkan instance extensions: " << extensions.size() << std::endl;

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "Simple VR Player";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Custom";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkResult result = vkCreateInstance(&createInfo, nullptr, &vkInstance);
    VK_CHECK(result, "Failed to create Vulkan instance");

    std::cout << "Vulkan instance created successfully" << std::endl;
    return true;
}

bool SimpleVRPlayer::createVulkanDevice() {
    std::cout << "Creating Vulkan device..." << std::endl;

    // Get physical device from OpenXR
    PFN_xrGetVulkanGraphicsDeviceKHR pfnGetVulkanGraphicsDeviceKHR;
    xrGetInstanceProcAddr(xrInstance, "xrGetVulkanGraphicsDeviceKHR",
        (PFN_xrVoidFunction*)&pfnGetVulkanGraphicsDeviceKHR);

    XrResult xrResult = pfnGetVulkanGraphicsDeviceKHR(xrInstance, xrSystemId, vkInstance, &vkPhysicalDevice);
    XR_CHECK(xrResult, "Failed to get Vulkan physical device");

    // Find queue family with graphics support
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queueFamilyIndex = i;
            break;
        }
    }

    // Get required device extensions from OpenXR
    PFN_xrGetVulkanDeviceExtensionsKHR pfnGetVulkanDeviceExtensionsKHR;
    xrGetInstanceProcAddr(xrInstance, "xrGetVulkanDeviceExtensionsKHR",
        (PFN_xrVoidFunction*)&pfnGetVulkanDeviceExtensionsKHR);

    uint32_t extCount = 0;
    pfnGetVulkanDeviceExtensionsKHR(xrInstance, xrSystemId, 0, &extCount, nullptr);

    std::vector<char> extBuffer(extCount);
    pfnGetVulkanDeviceExtensionsKHR(xrInstance, xrSystemId, extCount, &extCount, extBuffer.data());

    std::vector<const char*> deviceExtensions;
    char* token = strtok(extBuffer.data(), " ");
    while (token != nullptr) {
        deviceExtensions.push_back(token);
        token = strtok(nullptr, " ");
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = deviceExtensions.size();
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    VkResult result = vkCreateDevice(vkPhysicalDevice, &createInfo, nullptr, &vkDevice);
    VK_CHECK(result, "Failed to create Vulkan device");

    vkGetDeviceQueue(vkDevice, queueFamilyIndex, 0, &vkQueue);

    std::cout << "Vulkan device created successfully" << std::endl;
    return true;
}

bool SimpleVRPlayer::createXrSession() {
    std::cout << "Creating OpenXR session..." << std::endl;

    // Check Vulkan graphics requirements (required before session creation)
    PFN_xrGetVulkanGraphicsRequirementsKHR pfnGetVulkanGraphicsRequirementsKHR;
    xrGetInstanceProcAddr(xrInstance, "xrGetVulkanGraphicsRequirementsKHR",
        (PFN_xrVoidFunction*)&pfnGetVulkanGraphicsRequirementsKHR);

    XrGraphicsRequirementsVulkanKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
    XrResult xrResult = pfnGetVulkanGraphicsRequirementsKHR(xrInstance, xrSystemId, &graphicsRequirements);
    XR_CHECK(xrResult, "Failed to get Vulkan graphics requirements");

    XrGraphicsBindingVulkanKHR graphicsBinding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    graphicsBinding.instance = vkInstance;
    graphicsBinding.physicalDevice = vkPhysicalDevice;
    graphicsBinding.device = vkDevice;
    graphicsBinding.queueFamilyIndex = queueFamilyIndex;
    graphicsBinding.queueIndex = 0;

    XrSessionCreateInfo createInfo{XR_TYPE_SESSION_CREATE_INFO};
    createInfo.next = &graphicsBinding;
    createInfo.systemId = xrSystemId;

    XrResult result = xrCreateSession(xrInstance, &createInfo, &xrSession);
    XR_CHECK(result, "Failed to create OpenXR session");

    std::cout << "OpenXR session created successfully" << std::endl;
    return true;
}

bool SimpleVRPlayer::createXrSpace() {
    std::cout << "Creating OpenXR reference space..." << std::endl;

    XrPosef identityPose{};
    identityPose.orientation.w = 1.0f;
    identityPose.position = {0.0f, 0.0f, 0.0f};

    XrReferenceSpaceCreateInfo createInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    createInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    createInfo.poseInReferenceSpace = identityPose;

    XrResult result = xrCreateReferenceSpace(xrSession, &createInfo, &xrPlaySpace);
    XR_CHECK(result, "Failed to create reference space");

    std::cout << "Reference space created successfully" << std::endl;
    return true;
}

bool SimpleVRPlayer::createSwapchains() {
    std::cout << "Creating swapchains..." << std::endl;

    // Enumerate swapchain formats
    uint32_t formatCount;
    xrEnumerateSwapchainFormats(xrSession, 0, &formatCount, nullptr);

    std::vector<int64_t> formats(formatCount);
    xrEnumerateSwapchainFormats(xrSession, formatCount, &formatCount, formats.data());

    // Choose SRGB format if available, otherwise use first format
    int64_t chosenFormat = formats[0];
    for (int64_t format : formats) {
        if (format == VK_FORMAT_R8G8B8A8_SRGB || format == VK_FORMAT_B8G8R8A8_SRGB) {
            chosenFormat = format;
            break;
        }
    }

    std::cout << "Using swapchain format: " << chosenFormat << std::endl;
    swapchainFormat = static_cast<VkFormat>(chosenFormat);

    // Create one swapchain per view (eye)
    xrSwapchains.resize(viewCount);
    swapchainLengths.resize(viewCount);
    swapchainImages.resize(viewCount);

    for (uint32_t i = 0; i < viewCount; i++) {
        XrSwapchainCreateInfo createInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        createInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.format = chosenFormat;
        createInfo.sampleCount = 1;
        createInfo.width = viewConfigs[i].recommendedImageRectWidth;
        createInfo.height = viewConfigs[i].recommendedImageRectHeight;
        createInfo.faceCount = 1;
        createInfo.arraySize = 1;
        createInfo.mipCount = 1;

        XrResult result = xrCreateSwapchain(xrSession, &createInfo, &xrSwapchains[i]);
        XR_CHECK(result, "Failed to create swapchain");

        // Get swapchain images
        result = xrEnumerateSwapchainImages(xrSwapchains[i], 0, &swapchainLengths[i], nullptr);
        XR_CHECK(result, "Failed to get swapchain image count");

        swapchainImages[i].resize(swapchainLengths[i], {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        result = xrEnumerateSwapchainImages(xrSwapchains[i], swapchainLengths[i], &swapchainLengths[i],
            reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchainImages[i].data()));
        XR_CHECK(result, "Failed to enumerate swapchain images");

        std::cout << "  Swapchain " << i << ": " << swapchainLengths[i] << " images" << std::endl;
    }

    // Initialize view and projection view arrays
    views.resize(viewCount, {XR_TYPE_VIEW});
    projectionViews.resize(viewCount, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW});

    for (uint32_t i = 0; i < viewCount; i++) {
        projectionViews[i].subImage.swapchain = xrSwapchains[i];
        projectionViews[i].subImage.imageArrayIndex = 0;
        projectionViews[i].subImage.imageRect.offset = {0, 0};
        projectionViews[i].subImage.imageRect.extent = {
            (int32_t)viewConfigs[i].recommendedImageRectWidth,
            (int32_t)viewConfigs[i].recommendedImageRectHeight
        };
        std::cout << "DEBUG: Eye " << i << " projectionView:" << std::endl;
        std::cout << "  swapchain=" << projectionViews[i].subImage.swapchain << std::endl;
        std::cout << "  imageRect offset=(" << projectionViews[i].subImage.imageRect.offset.x << ", "
                  << projectionViews[i].subImage.imageRect.offset.y << ")" << std::endl;
        std::cout << "  imageRect extent=(" << projectionViews[i].subImage.imageRect.extent.width << ", "
                  << projectionViews[i].subImage.imageRect.extent.height << ")" << std::endl;
    }

    std::cout << "Swapchains created successfully" << std::endl;
    return true;
}

// Helper function to load compiled SPIR-V shader from file
std::vector<char> SimpleVRPlayer::loadShaderFile(const char* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open shader file: " << filename << std::endl;
        return {};
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule SimpleVRPlayer::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(vkDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "ERROR: Failed to create shader module" << std::endl;
        return VK_NULL_HANDLE;
    }

    return shaderModule;
}

uint32_t SimpleVRPlayer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    std::cerr << "ERROR: Failed to find suitable memory type" << std::endl;
    return 0;
}

bool SimpleVRPlayer::createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VkResult result = vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &vkRenderPass);
    VK_CHECK(result, "Failed to create render pass");

    return true;
}

bool SimpleVRPlayer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &samplerLayoutBinding;

    VkResult result = vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &vkDescriptorSetLayout);
    VK_CHECK(result, "Failed to create descriptor set layout");

    return true;
}

bool SimpleVRPlayer::createGraphicsPipeline() {
    // Load shaders
    auto vertShaderCode = loadShaderFile("../shaders/quad.vert.spv");
    auto fragShaderCode = loadShaderFile("../shaders/quad.frag.spv");

    if (vertShaderCode.empty() || fragShaderCode.empty()) {
        std::cerr << "ERROR: Failed to load shader files" << std::endl;
        return false;
    }

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    if (vertShaderModule == VK_NULL_HANDLE || fragShaderModule == VK_NULL_HANDLE) {
        return false;
    }

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // Vertex input
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(float) * 5;  // 3 floats for position, 2 for UV
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescriptions[2] = {};
    // Position
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = 0;
    // TexCoord
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = sizeof(float) * 3;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Viewport and scissor (dynamic)
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;  // Disable culling for debugging
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // Dynamic state
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // Push constants for MVP matrix + UV offset/scale
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(float) * 16 + sizeof(float) * 4;  // mat4 (64 bytes) + vec2 + vec2 (16 bytes) = 80 bytes

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &vkDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    VkResult result = vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, nullptr, &vkPipelineLayout);
    VK_CHECK(result, "Failed to create pipeline layout");

    // Graphics pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = vkPipelineLayout;
    pipelineInfo.renderPass = vkRenderPass;
    pipelineInfo.subpass = 0;

    result = vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vkGraphicsPipeline);
    VK_CHECK(result, "Failed to create graphics pipeline");

    // Cleanup shader modules
    vkDestroyShaderModule(vkDevice, fragShaderModule, nullptr);
    vkDestroyShaderModule(vkDevice, vertShaderModule, nullptr);

    return true;
}

bool SimpleVRPlayer::createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkResult result = vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &vkDescriptorPool);
    VK_CHECK(result, "Failed to create descriptor pool");

    return true;
}

bool SimpleVRPlayer::createDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = vkDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &vkDescriptorSetLayout;

    VkResult result = vkAllocateDescriptorSets(vkDevice, &allocInfo, &vkDescriptorSet);
    VK_CHECK(result, "Failed to allocate descriptor set");

    return true;
}

bool SimpleVRPlayer::createQuadGeometry() {
    // Quad vertices: position (x, y, z) and UV coordinates (u, v)
    // Position the quad 2 meters in front of the viewer, 2 meters wide and tall
    float quadVertices[] = {
        // Positions (X, Y, Z)     // UVs (U, V)
        -1.0f, -1.0f, -2.0f,       0.0f, 0.0f,  // Bottom-left (flipped V)
         1.0f, -1.0f, -2.0f,       1.0f, 0.0f,  // Bottom-right (flipped V)
         1.0f,  1.0f, -2.0f,       1.0f, 1.0f,  // Top-right (flipped V)
        -1.0f,  1.0f, -2.0f,       0.0f, 1.0f   // Top-left (flipped V)
    };

    uint16_t quadIndices[] = {
        0, 1, 2,  // First triangle
        2, 3, 0   // Second triangle
    };

    indexCount = 6;

    // Create vertex buffer
    VkDeviceSize bufferSize = sizeof(quadVertices);

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &vkVertexBuffer);
    VK_CHECK(result, "Failed to create vertex buffer");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vkDevice, vkVertexBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    result = vkAllocateMemory(vkDevice, &allocInfo, nullptr, &vkVertexBufferMemory);
    VK_CHECK(result, "Failed to allocate vertex buffer memory");

    vkBindBufferMemory(vkDevice, vkVertexBuffer, vkVertexBufferMemory, 0);

    void* data;
    vkMapMemory(vkDevice, vkVertexBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, quadVertices, (size_t)bufferSize);
    vkUnmapMemory(vkDevice, vkVertexBufferMemory);

    // Create index buffer
    bufferSize = sizeof(quadIndices);

    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    result = vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &vkIndexBuffer);
    VK_CHECK(result, "Failed to create index buffer");

    vkGetBufferMemoryRequirements(vkDevice, vkIndexBuffer, &memRequirements);

    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    result = vkAllocateMemory(vkDevice, &allocInfo, nullptr, &vkIndexBufferMemory);
    VK_CHECK(result, "Failed to allocate index buffer memory");

    vkBindBufferMemory(vkDevice, vkIndexBuffer, vkIndexBufferMemory, 0);

    vkMapMemory(vkDevice, vkIndexBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, quadIndices, (size_t)bufferSize);
    vkUnmapMemory(vkDevice, vkIndexBufferMemory);

    return true;
}

bool SimpleVRPlayer::createSphereGeometry(int segments, float radius, float angleHorizontal, float angleVertical) {
    std::vector<float> vertices;
    std::vector<uint16_t> indices;

    // Generate sphere vertices
    for (int lat = 0; lat <= segments; lat++) {
        float theta = (float)lat / segments * angleVertical;
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int lon = 0; lon <= segments; lon++) {
            // Center the horizontal span around the front (-Z axis)
            // For 180°, phi goes from -π/2 to +π/2 (centered at 0, facing -Z)
            float phi = ((float)lon / segments - 0.5f) * angleHorizontal;
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            // Position (X, Y, Z)
            // Standard sphere coordinates with viewer at origin facing -Z
            float x = radius * sinTheta * sinPhi;  // Left-Right
            float y = radius * cosTheta;            // Up-Down
            float z = -radius * sinTheta * cosPhi; // Front-Back (negative = in front)

            // UV coordinates (flip V vertically)
            float u = (float)lon / segments;
            float v = 1.0f - (float)lat / segments;  // Flip vertically

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            vertices.push_back(u);
            vertices.push_back(v);
        }
    }

    // Generate sphere indices (clockwise winding for inward-facing triangles)
    for (int lat = 0; lat < segments; lat++) {
        for (int lon = 0; lon < segments; lon++) {
            int first = lat * (segments + 1) + lon;
            int second = first + segments + 1;

            // Clockwise winding when viewed from inside (matches VK_FRONT_FACE_CLOCKWISE)
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(first + 1);
            indices.push_back(second);
            indices.push_back(second + 1);
        }
    }

    indexCount = indices.size();

    std::cout << "Sphere created: " << vertices.size()/5 << " vertices, "
              << indexCount << " indices" << std::endl;

    // DEBUG: Print some sample vertex positions to verify sphere geometry
    std::cout << "DEBUG: Sample sphere vertices (first row - top pole):" << std::endl;
    for (int i = 0; i < std::min(5, (int)(vertices.size() / 5)); i++) {
        int idx = i * 5;
        std::cout << "  Vertex " << i << ": pos=(" << vertices[idx] << ", " << vertices[idx+1] << ", " << vertices[idx+2]
                  << ") uv=(" << vertices[idx+3] << ", " << vertices[idx+4] << ")" << std::endl;
    }

    // Print middle row (should show actual hemisphere curvature)
    int segmentsPerRow = 65; // segments + 1
    int middleRow = 32; // Middle of 64 segments
    std::cout << "DEBUG: Sample sphere vertices (middle row - equator):" << std::endl;
    for (int i = 0; i < std::min(5, segmentsPerRow); i++) {
        int vertexIndex = middleRow * segmentsPerRow + i;
        int idx = vertexIndex * 5;
        std::cout << "  Vertex " << vertexIndex << ": pos=(" << vertices[idx] << ", " << vertices[idx+1] << ", " << vertices[idx+2]
                  << ") uv=(" << vertices[idx+3] << ", " << vertices[idx+4] << ")" << std::endl;
    }

    // Create vertex buffer
    VkDeviceSize vertexBufferSize = vertices.size() * sizeof(float);

    VkBufferCreateInfo vertexBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    vertexBufferInfo.size = vertexBufferSize;
    vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(vkDevice, &vertexBufferInfo, nullptr, &vkVertexBuffer);
    VK_CHECK(result, "Failed to create vertex buffer");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vkDevice, vkVertexBuffer, &memRequirements);

    VkMemoryAllocateInfo vertexAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    vertexAllocInfo.allocationSize = memRequirements.size;
    vertexAllocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    result = vkAllocateMemory(vkDevice, &vertexAllocInfo, nullptr, &vkVertexBufferMemory);
    VK_CHECK(result, "Failed to allocate vertex buffer memory");

    vkBindBufferMemory(vkDevice, vkVertexBuffer, vkVertexBufferMemory, 0);

    void* data;
    vkMapMemory(vkDevice, vkVertexBufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)vertexBufferSize);
    vkUnmapMemory(vkDevice, vkVertexBufferMemory);

    // Create index buffer
    VkDeviceSize indexBufferSize = indices.size() * sizeof(uint16_t);

    VkBufferCreateInfo indexBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    indexBufferInfo.size = indexBufferSize;
    indexBufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(vkDevice, &indexBufferInfo, nullptr, &vkIndexBuffer);
    VK_CHECK(result, "Failed to create index buffer");

    vkGetBufferMemoryRequirements(vkDevice, vkIndexBuffer, &memRequirements);

    VkMemoryAllocateInfo indexAllocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    indexAllocInfo.allocationSize = memRequirements.size;
    indexAllocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    result = vkAllocateMemory(vkDevice, &indexAllocInfo, nullptr, &vkIndexBufferMemory);
    VK_CHECK(result, "Failed to allocate index buffer memory");

    vkBindBufferMemory(vkDevice, vkIndexBuffer, vkIndexBufferMemory, 0);

    vkMapMemory(vkDevice, vkIndexBufferMemory, 0, indexBufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)indexBufferSize);
    vkUnmapMemory(vkDevice, vkIndexBufferMemory);

    return true;
}

void SimpleVRPlayer::updateDescriptorSet() {
    // Update descriptor set to bind the video texture from CUDA interop
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = cudaInterop.imageView;
    imageInfo.sampler = cudaInterop.sampler;

    VkWriteDescriptorSet descriptorWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    descriptorWrite.dstSet = vkDescriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(vkDevice, 1, &descriptorWrite, 0, nullptr);
}

bool SimpleVRPlayer::createFramebuffers() {
    std::cout << "Creating framebuffers..." << std::endl;

    vkSwapchainImageViews.resize(viewCount);
    vkFramebuffers.resize(viewCount);

    for (uint32_t eyeIndex = 0; eyeIndex < viewCount; eyeIndex++) {
        uint32_t imageCount = swapchainLengths[eyeIndex];
        vkSwapchainImageViews[eyeIndex].resize(imageCount);
        vkFramebuffers[eyeIndex].resize(imageCount);

        for (uint32_t i = 0; i < imageCount; i++) {
            // Create image view for swapchain image
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = swapchainImages[eyeIndex][i].image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = swapchainFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            VkResult result = vkCreateImageView(vkDevice, &viewInfo, nullptr, &vkSwapchainImageViews[eyeIndex][i]);
            VK_CHECK(result, "Failed to create swapchain image view");

            // Create framebuffer
            VkImageView attachments[] = {
                vkSwapchainImageViews[eyeIndex][i]
            };

            VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            framebufferInfo.renderPass = vkRenderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = viewConfigs[eyeIndex].recommendedImageRectWidth;
            framebufferInfo.height = viewConfigs[eyeIndex].recommendedImageRectHeight;
            framebufferInfo.layers = 1;

            result = vkCreateFramebuffer(vkDevice, &framebufferInfo, nullptr, &vkFramebuffers[eyeIndex][i]);
            VK_CHECK(result, "Failed to create framebuffer");
        }
    }

    std::cout << "Framebuffers created successfully" << std::endl;
    return true;
}

bool SimpleVRPlayer::createRenderingResources() {
    std::cout << "Creating rendering resources..." << std::endl;

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkResult result = vkCreateCommandPool(vkDevice, &poolInfo, nullptr, &vkCommandPool);
    VK_CHECK(result, "Failed to create command pool");

    // Create rendering pipeline
    if (!createDescriptorSetLayout()) return false;
    if (!createRenderPass()) return false;
    if (!createGraphicsPipeline()) return false;
    if (!createDescriptorPool()) return false;
    if (!createDescriptorSet()) return false;

    // Create geometry based on video mode
    if (videoMode == MODE_FLAT) {
        std::cout << "Creating FLAT quad geometry..." << std::endl;
        if (!createQuadGeometry()) return false;
    } else if (videoMode == MODE_SPHERE_180) {
        std::cout << "Creating 180° SPHERE geometry..." << std::endl;
        // 180° hemisphere: 64 segments, 1.5m radius for proper stereo separation, 180° horizontal, 180° vertical
        if (!createSphereGeometry(64, 1.5f, M_PI, M_PI)) return false;
    } else if (videoMode == MODE_SPHERE_360) {
        std::cout << "Creating 360° SPHERE geometry..." << std::endl;
        // 360° full sphere: 64 segments, 3m radius (very close for maximum immersion), 360° horizontal, 180° vertical
        if (!createSphereGeometry(64, 3.0f, 2.0f * M_PI, M_PI)) return false;
    }

    if (!createFramebuffers()) return false;

    std::cout << "Rendering resources created successfully" << std::endl;
    return true;
}

void SimpleVRPlayer::renderEye(uint32_t eyeIndex, uint32_t imageIndex, VkImage image, uint32_t width, uint32_t height) {
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = vkCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer;
    vkAllocateCommandBuffers(vkDevice, &allocInfo, &cmdBuffer);

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    if (videoLoaded) {
        // DEBUG: Confirm we're rendering video
        static bool printed = false;
        if (!printed) {
            std::cout << "DEBUG: Rendering video texture (videoLoaded=true)" << std::endl;
            std::cout << "DEBUG: cudaInterop.image = " << cudaInterop.image << std::endl;
            std::cout << "DEBUG: cudaInterop.imageView = " << cudaInterop.imageView << std::endl;
            std::cout << "DEBUG: cudaInterop.sampler = " << cudaInterop.sampler << std::endl;
            std::cout << "DEBUG: vkDescriptorSet = " << vkDescriptorSet << std::endl;
            printed = true;
        }

        // Transition video texture to SHADER_READ_ONLY_OPTIMAL
        VkImageMemoryBarrier srcBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.image = cudaInterop.image;
        srcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        srcBarrier.subresourceRange.baseMipLevel = 0;
        srcBarrier.subresourceRange.levelCount = 1;
        srcBarrier.subresourceRange.baseArrayLayer = 0;
        srcBarrier.subresourceRange.layerCount = 1;
        srcBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &srcBarrier);

        // Begin render pass
        VkRenderPassBeginInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        renderPassInfo.renderPass = vkRenderPass;
        renderPassInfo.framebuffer = vkFramebuffers[eyeIndex][imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = {width, height};

        VkClearValue clearValue{};
        clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};  // BLACK background
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearValue;

        vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Bind graphics pipeline
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkGraphicsPipeline);

        // Bind descriptor set (video texture)
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipelineLayout,
            0, 1, &vkDescriptorSet, 0, nullptr);

        // Set viewport and scissor
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)width;
        viewport.height = (float)height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = {width, height};
        vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

        // Compute MVP matrix from OpenXR view and projection
        using namespace MathUtils;
        Matrix4x4 projection = Matrix4x4::CreateProjectionFov(views[eyeIndex].fov, 0.1f, 100.0f);
        Matrix4x4 view = Matrix4x4::CreateViewMatrix(views[eyeIndex].pose);
        Matrix4x4 model = Matrix4x4::Identity();

        // DEBUG: Print view pose every 60 frames to see if tracking works
        static int frameCount = 0;
        if (eyeIndex == 0) frameCount++;
        if (frameCount % 60 == 0 && eyeIndex == 0) {
            std::cout << "Frame " << frameCount << " Eye " << eyeIndex << " pose: pos=("
                      << views[eyeIndex].pose.position.x << ", "
                      << views[eyeIndex].pose.position.y << ", "
                      << views[eyeIndex].pose.position.z << "), orient=("
                      << views[eyeIndex].pose.orientation.x << ", "
                      << views[eyeIndex].pose.orientation.y << ", "
                      << views[eyeIndex].pose.orientation.z << ", "
                      << views[eyeIndex].pose.orientation.w << ")" << std::endl;
        }

        // MVP = Projection * View * Model
        Matrix4x4 vp = Matrix4x4::Multiply(projection, view);
        Matrix4x4 mvp = Matrix4x4::Multiply(vp, model);

        // DEBUG: Print MVP matrix for first frame
        static bool printedMVP = false;
        if (!printedMVP && eyeIndex == 0) {
            std::cout << "DEBUG MVP matrix eye 0:" << std::endl;
            std::cout << "  Projection[0-3]: " << projection.m[0] << ", " << projection.m[1] << ", " << projection.m[2] << ", " << projection.m[3] << std::endl;
            std::cout << "  View[0-3]: " << view.m[0] << ", " << view.m[1] << ", " << view.m[2] << ", " << view.m[3] << std::endl;
            std::cout << "  Model[0-3]: " << model.m[0] << ", " << model.m[1] << ", " << model.m[2] << ", " << model.m[3] << std::endl;
            std::cout << "  MVP[0-3]: " << mvp.m[0] << ", " << mvp.m[1] << ", " << mvp.m[2] << ", " << mvp.m[3] << std::endl;
            printedMVP = true;
        }

        // Compute UV offset and scale for SBS mode
        float uvOffset[2], uvScale[2];
        if (sbsMode) {
            // Side-by-side mode: each eye sees half the texture
            uvScale[0] = 0.5f;  // Scale U to half
            uvScale[1] = 1.0f;  // Keep V at full height
            if (eyeIndex == 0) {
                // Left eye: left half (U from 0.0 to 0.5)
                uvOffset[0] = 0.0f;
                uvOffset[1] = 0.0f;
            } else {
                // Right eye: right half (U from 0.5 to 1.0)
                uvOffset[0] = 0.5f;
                uvOffset[1] = 0.0f;
            }

            static bool printedSBS = false;
            if (!printedSBS) {
                std::cout << "DEBUG SBS: Eye " << eyeIndex << " uvOffset=(" << uvOffset[0] << ", " << uvOffset[1]
                          << ") uvScale=(" << uvScale[0] << ", " << uvScale[1] << ")" << std::endl;
                if (eyeIndex == 1) printedSBS = true;
            }
        } else {
            // Mono mode: use full texture for both eyes
            uvScale[0] = 1.0f;
            uvScale[1] = 1.0f;
            uvOffset[0] = 0.0f;
            uvOffset[1] = 0.0f;
        }

        // Push constants: MVP matrix + UV offset + UV scale
        // NOTE: Explicit padding to ensure correct alignment
        struct PushConstants {
            float mvp[16];       // 64 bytes (0-63)
            float uvOffset[2];   // 8 bytes (64-71)
            float uvScale[2];    // 8 bytes (72-79)
        } __attribute__((packed)) pushConstants;

        // DEBUG: Print struct size and member offsets
        static bool printedLayout = false;
        if (!printedLayout) {
            std::cout << "DEBUG LAYOUT: sizeof(PushConstants) = " << sizeof(PushConstants) << std::endl;
            std::cout << "  offsetof(mvp) = " << offsetof(PushConstants, mvp) << std::endl;
            std::cout << "  offsetof(uvOffset) = " << offsetof(PushConstants, uvOffset) << std::endl;
            std::cout << "  offsetof(uvScale) = " << offsetof(PushConstants, uvScale) << std::endl;
            printedLayout = true;
        }

        memcpy(pushConstants.mvp, mvp.m, sizeof(mvp.m));
        pushConstants.uvOffset[0] = uvOffset[0];
        pushConstants.uvOffset[1] = uvOffset[1];
        pushConstants.uvScale[0] = uvScale[0];
        pushConstants.uvScale[1] = uvScale[1];

        // DEBUG: Print actual push constant values - ALWAYS print first few frames
        static int debugFrameCount = 0;
        if (debugFrameCount < 4) {
            std::cout << "DEBUG PUSH FRAME " << debugFrameCount << ": Eye " << eyeIndex
                      << " uvOffset=(" << pushConstants.uvOffset[0] << ", " << pushConstants.uvOffset[1]
                      << ") uvScale=(" << pushConstants.uvScale[0] << ", " << pushConstants.uvScale[1] << ")" << std::endl;
            if (eyeIndex == 1) debugFrameCount++;
        }

        vkCmdPushConstants(cmdBuffer, vkPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
            0, sizeof(PushConstants), &pushConstants);

        // Bind vertex and index buffers
        VkBuffer vertexBuffers[] = {vkVertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmdBuffer, vkIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

        // DEBUG: Confirm we're drawing and which framebuffer
        static bool drawPrinted = false;
        if (!drawPrinted) {
            std::cout << "DEBUG: Drawing with indexCount=" << indexCount;
            if (indexCount == 6) {
                std::cout << " (QUAD geometry)" << std::endl;
            } else {
                std::cout << " (SPHERE geometry)" << std::endl;
            }
            std::cout << "DEBUG: Eye " << eyeIndex << " rendering to framebuffer "
                      << vkFramebuffers[eyeIndex][imageIndex]
                      << " (image " << imageIndex << ")" << std::endl;
            if (eyeIndex == 1) drawPrinted = true;
        }

        // Draw the geometry
        vkCmdDrawIndexed(cmdBuffer, indexCount, 1, 0, 0, 0);

        // End render pass
        vkCmdEndRenderPass(cmdBuffer);

        // Transition video texture back to GENERAL for CUDA
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        srcBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;

        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &srcBarrier);
    } else {
        // Fallback: Clear with test pattern colors using render pass
        VkRenderPassBeginInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        renderPassInfo.renderPass = vkRenderPass;
        renderPassInfo.framebuffer = vkFramebuffers[eyeIndex][imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = {width, height};

        VkClearValue clearValue{};
        if (eyeIndex == 0) {
            // Left eye: Cyan
            clearValue.color = {{0.0f, 0.7f, 1.0f, 1.0f}};
        } else {
            // Right eye: Magenta
            clearValue.color = {{1.0f, 0.0f, 0.7f, 1.0f}};
        }
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearValue;

        vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdEndRenderPass(cmdBuffer);
    }

    // End command buffer
    vkEndCommandBuffer(cmdBuffer);

    // Submit command buffer
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    vkQueueSubmit(vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vkQueue);

    // Free command buffer
    vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &cmdBuffer);
}

void SimpleVRPlayer::processEvents(bool* exitRequested, bool* sessionRunning) {
    *exitRequested = false;

    XrEventDataBuffer eventData{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(xrInstance, &eventData) == XR_SUCCESS) {
        switch (eventData.type) {
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                XrEventDataSessionStateChanged* stateEvent =
                    reinterpret_cast<XrEventDataSessionStateChanged*>(&eventData);

                std::cout << "Session state changed to: " << stateEvent->state << std::endl;

                switch (stateEvent->state) {
                    case XR_SESSION_STATE_READY: {
                        XrSessionBeginInfo beginInfo{XR_TYPE_SESSION_BEGIN_INFO};
                        beginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                        xrBeginSession(xrSession, &beginInfo);
                        *sessionRunning = true;
                        break;
                    }
                    case XR_SESSION_STATE_STOPPING:
                        xrEndSession(xrSession);
                        *sessionRunning = false;
                        break;
                    case XR_SESSION_STATE_EXITING:
                    case XR_SESSION_STATE_LOSS_PENDING:
                        *exitRequested = true;
                        break;
                    default:
                        break;
                }
                break;
            }
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                *exitRequested = true;
                break;
            default:
                break;
        }

        eventData.type = XR_TYPE_EVENT_DATA_BUFFER;
    }
}

void SimpleVRPlayer::renderFrame() {
    auto frameStartTime = std::chrono::steady_clock::now();

    // Wait for the next frame
    auto waitFrameStart = std::chrono::steady_clock::now();
    XrFrameWaitInfo waitInfo{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frameState{XR_TYPE_FRAME_STATE};
    xrWaitFrame(xrSession, &waitInfo, &frameState);
    auto waitFrameEnd = std::chrono::steady_clock::now();
    double waitFrameTime = std::chrono::duration<double, std::milli>(waitFrameEnd - waitFrameStart).count();

    // Begin frame
    auto beginFrameStart = std::chrono::steady_clock::now();
    XrFrameBeginInfo beginInfo{XR_TYPE_FRAME_BEGIN_INFO};
    xrBeginFrame(xrSession, &beginInfo);
    auto beginFrameEnd = std::chrono::steady_clock::now();
    double beginFrameTime = std::chrono::duration<double, std::milli>(beginFrameEnd - beginFrameStart).count();

    // Render layers (must be declared outside if block to stay in scope)
    std::vector<XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerProjection projectionLayer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};

    double decodeTime = 0.0;
    double uploadTime = 0.0;
    double renderEyeTime = 0.0;
    double locateViewsTime = 0.0;
    double acquireSwapchainTime = 0.0;
    double openxrTime = 0.0;

    if (frameState.shouldRender) {
        // Decode next video frame if video is loaded
        if (videoLoaded) {
            auto decodeStart = std::chrono::steady_clock::now();

            int width, height;
            size_t nv12Size;
            if (videoDecoder.readFrame(frameBuffer.data(), &width, &height, &nv12Size)) {
                auto decodeEnd = std::chrono::steady_clock::now();
                decodeTime = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

                // Convert NV12 to RGBA on GPU using CUDA
                auto uploadStart = std::chrono::steady_clock::now();
                cudaConvertNV12ToRGBA(&cudaInterop, frameBuffer.data(), nv12Size);
                auto uploadEnd = std::chrono::steady_clock::now();
                uploadTime = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
            } else {
                // End of video - loop back to start
                std::cout << "End of video reached, looping..." << std::endl;
                videoDecoder.close();
                videoLoaded = false;
            }
        }

        // Locate views
        auto locateViewsStart = std::chrono::steady_clock::now();
        XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
        locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        locateInfo.displayTime = frameState.predictedDisplayTime;
        locateInfo.space = xrPlaySpace;

        XrViewState viewState{XR_TYPE_VIEW_STATE};
        uint32_t viewCountOutput;
        xrLocateViews(xrSession, &locateInfo, &viewState, viewCount, &viewCountOutput, views.data());
        auto locateViewsEnd = std::chrono::steady_clock::now();
        locateViewsTime = std::chrono::duration<double, std::milli>(locateViewsEnd - locateViewsStart).count();

        // Acquire and render to each swapchain
        auto acquireSwapchainStart = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < viewCount; i++) {
            uint32_t imageIndex;
            XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            xrAcquireSwapchainImage(xrSwapchains[i], &acquireInfo, &imageIndex);

            XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            waitInfo.timeout = XR_INFINITE_DURATION;
            xrWaitSwapchainImage(xrSwapchains[i], &waitInfo);

            // Render to the swapchain image
            auto renderEyeStart = std::chrono::steady_clock::now();  // NEW
            VkImage image = swapchainImages[i][imageIndex].image;

            // DEBUG: Print VkImage pointers to verify they're different
            static bool printedImages = false;
            if (!printedImages) {
                std::cout << "DEBUG RENDER: Eye " << i << " rendering to VkImage " << image << std::endl;
                if (i == 1) printedImages = true;
            }

            renderEye(i, imageIndex, image, viewConfigs[i].recommendedImageRectWidth, viewConfigs[i].recommendedImageRectHeight);
            auto renderEyeEnd = std::chrono::steady_clock::now();  // NEW
            renderEyeTime += std::chrono::duration<double, std::milli>(renderEyeEnd - renderEyeStart).count();  // NEW

            XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            xrReleaseSwapchainImage(xrSwapchains[i], &releaseInfo);

            // Update projection view
            projectionViews[i].pose = views[i].pose;
            projectionViews[i].fov = views[i].fov;
        }
        auto acquireSwapchainEnd = std::chrono::steady_clock::now();
        acquireSwapchainTime = std::chrono::duration<double, std::milli>(acquireSwapchainEnd - acquireSwapchainStart).count() - renderEyeTime;

        // Set up projection layer
        projectionLayer.space = xrPlaySpace;
        projectionLayer.viewCount = viewCount;
        projectionLayer.views = projectionViews.data();

        layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&projectionLayer));
    }

    // End frame
    auto endFrameStart = std::chrono::steady_clock::now();
    XrFrameEndInfo endInfo{XR_TYPE_FRAME_END_INFO};
    endInfo.displayTime = frameState.predictedDisplayTime;
    endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    endInfo.layerCount = layers.size();
    endInfo.layers = layers.data();

    xrEndFrame(xrSession, &endInfo);
    auto endFrameEnd = std::chrono::steady_clock::now();
    double endFrameTime = std::chrono::duration<double, std::milli>(endFrameEnd - endFrameStart).count();

    // Calculate total OpenXR time from detailed components
    openxrTime = waitFrameTime + beginFrameTime + locateViewsTime + acquireSwapchainTime + endFrameTime;

    // Performance tracking
    auto frameEndTime = std::chrono::steady_clock::now();
    double frameTime = std::chrono::duration<double, std::milli>(frameEndTime - frameStartTime).count();

    if (debugMode) {
        frameCount++;
        totalDecodeTime += decodeTime;
        totalUploadTime += uploadTime;
        totalRenderTime += frameTime;
        totalRenderEyeTime += renderEyeTime;  // NEW
        totalOpenXRTime += openxrTime;         // NEW

        // Print stats every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - lastFPSPrint).count();
        if (elapsed >= 1.0) {
            double avgFPS = frameCount / elapsed;
            double avgDecode = frameCount > 0 ? totalDecodeTime / frameCount : 0.0;
            double avgUpload = frameCount > 0 ? totalUploadTime / frameCount : 0.0;
            double avgFrame = frameCount > 0 ? totalRenderTime / frameCount : 0.0;
            double avgRenderEye = frameCount > 0 ? totalRenderEyeTime / frameCount : 0.0;  // NEW
            double avgOpenXR = frameCount > 0 ? totalOpenXRTime / frameCount : 0.0;  // NEW

            // Calculate accounted and unaccounted time
            double accountedTime = avgDecode + avgUpload + avgRenderEye + avgOpenXR;
            double unaccountedTime = avgFrame - accountedTime;

            // Print comprehensive timing breakdown
            std::cout << "[PERF] FPS: " << std::fixed << std::setprecision(1) << avgFPS
                      << " | Frame: " << std::setprecision(2) << avgFrame << "ms" << std::endl;
            std::cout << "       Decode: " << avgDecode << "ms"
                      << " | Upload: " << avgUpload << "ms"
                      << " | RenderEye: " << avgRenderEye << "ms"
                      << " | OpenXR: " << avgOpenXR << "ms" << std::endl;
            std::cout << "       Accounted: " << accountedTime << "ms"
                      << " | UNACCOUNTED: " << unaccountedTime << "ms ("
                      << std::setprecision(1) << (unaccountedTime / avgFrame * 100.0) << "%)" << std::endl;

            // Calculate and print detailed OpenXR breakdown (these are single-frame samples, not averages)
            std::cout << "       [OpenXR] WaitFrame: " << std::setprecision(2) << waitFrameTime << "ms"
                      << " | BeginFrame: " << beginFrameTime << "ms"
                      << " | LocateViews: " << locateViewsTime << "ms" << std::endl;
            std::cout << "       [OpenXR] AcquireSwap: " << acquireSwapchainTime << "ms"
                      << " | EndFrame: " << endFrameTime << "ms" << std::endl;

            // Print warning if xrWaitFrame is suspiciously long
            if (waitFrameTime > 10.0) {
                std::cout << "       ⚠️  WARNING: xrWaitFrame is " << waitFrameTime << "ms - likely vsync throttling!" << std::endl;
            }

            // Reset counters
            lastFPSPrint = now;
            frameCount = 0;
            totalDecodeTime = 0.0;
            totalUploadTime = 0.0;
            totalRenderTime = 0.0;
            totalRenderEyeTime = 0.0;  // NEW
            totalOpenXRTime = 0.0;     // NEW
        }
    }
}

bool SimpleVRPlayer::initialize() {
    if (!createXrInstance()) return false;
    if (!createXrSystem()) return false;
    if (!createVulkanInstance()) return false;
    if (!createVulkanDevice()) return false;
    if (!createXrSession()) return false;
    if (!createXrSpace()) return false;
    if (!createSwapchains()) return false;
    if (!createRenderingResources()) return false;

    std::cout << "\nInitialization complete!" << std::endl;
    return true;
}

bool SimpleVRPlayer::loadVideo(const char* filename) {
    std::cout << "\nLoading video file: " << filename << std::endl;

    // Open video file with decoder
    if (!videoDecoder.open(filename)) {
        std::cerr << "ERROR: Failed to open video file" << std::endl;
        return false;
    }

    // Create texture to match video dimensions
    int videoWidth = videoDecoder.getWidth();
    int videoHeight = videoDecoder.getHeight();

    // Initialize CUDA interop
    if (!cudaInteropInit(&cudaInterop, vkDevice, vkPhysicalDevice)) {
        std::cerr << "ERROR: Failed to initialize CUDA interop" << std::endl;
        return false;
    }

    if (!cudaInteropCreateImage(&cudaInterop, videoWidth, videoHeight)) {
        std::cerr << "ERROR: Failed to create interop image" << std::endl;
        return false;
    }

    if (!cudaInteropImportMemory(&cudaInterop)) {
        std::cerr << "ERROR: Failed to import Vulkan memory into CUDA" << std::endl;
        return false;
    }

    // Update descriptor set to bind the video texture
    updateDescriptorSet();
    std::cout << "Descriptor set updated with video texture" << std::endl;

    // Allocate frame buffer for NV12 frames (1.5 bytes per pixel)
    frameBuffer.resize(videoWidth * videoHeight * 3 / 2);  // NV12

    // Decode and convert first frame
    int width, height;
    size_t nv12Size;
    if (videoDecoder.readFrame(frameBuffer.data(), &width, &height, &nv12Size)) {
        cudaConvertNV12ToRGBA(&cudaInterop, frameBuffer.data(), nv12Size);
        std::cout << "First frame loaded and converted via CUDA" << std::endl;

        // Transition image from UNDEFINED to GENERAL layout for CUDA/Vulkan interop
        VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.commandPool = vkCommandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmdBuffer;
        vkAllocateCommandBuffers(vkDevice, &allocInfo, &cmdBuffer);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuffer, &beginInfo);

        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = cudaInterop.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        vkEndCommandBuffer(cmdBuffer);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        vkQueueSubmit(vkQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(vkQueue);

        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &cmdBuffer);
        std::cout << "Image transitioned to GENERAL layout" << std::endl;
    } else {
        std::cerr << "ERROR: Failed to decode first frame" << std::endl;
        return false;
    }

    videoLoaded = true;
    std::cout << "Video loaded successfully!" << std::endl;
    return true;
}

void SimpleVRPlayer::run() {
    std::cout << "\nStarting main loop..." << std::endl;
    std::cout << "Press Ctrl+C to exit" << std::endl;

    bool exitRequested = false;
    bool sessionRunning = false;

    // Initialize performance tracking
    lastFPSPrint = std::chrono::steady_clock::now();

    while (!exitRequested) {
        processEvents(&exitRequested, &sessionRunning);

        if (sessionRunning) {
            renderFrame();
        }
    }

    std::cout << "Exiting main loop" << std::endl;
}

void SimpleVRPlayer::shutdown() {
    std::cout << "\nShutting down..." << std::endl;

    // Wait for device to be idle before cleanup
    if (vkDevice != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vkDevice);
    }

    // Cleanup CUDA interop (must be done before destroying Vulkan device)
    cudaInteropDestroy(&cudaInterop);

    // Destroy framebuffers and image views
    for (auto& framebufferList : vkFramebuffers) {
        for (auto framebuffer : framebufferList) {
            if (framebuffer != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(vkDevice, framebuffer, nullptr);
            }
        }
    }
    for (auto& imageViewList : vkSwapchainImageViews) {
        for (auto imageView : imageViewList) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(vkDevice, imageView, nullptr);
            }
        }
    }

    // Destroy geometry buffers
    if (vkVertexBuffer != VK_NULL_HANDLE) vkDestroyBuffer(vkDevice, vkVertexBuffer, nullptr);
    if (vkVertexBufferMemory != VK_NULL_HANDLE) vkFreeMemory(vkDevice, vkVertexBufferMemory, nullptr);
    if (vkIndexBuffer != VK_NULL_HANDLE) vkDestroyBuffer(vkDevice, vkIndexBuffer, nullptr);
    if (vkIndexBufferMemory != VK_NULL_HANDLE) vkFreeMemory(vkDevice, vkIndexBufferMemory, nullptr);

    // Destroy descriptor sets and pools
    if (vkDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(vkDevice, vkDescriptorPool, nullptr);
    if (vkDescriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout, nullptr);

    // Destroy pipeline and layout
    if (vkGraphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(vkDevice, vkGraphicsPipeline, nullptr);
    if (vkPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, nullptr);
    if (vkRenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(vkDevice, vkRenderPass, nullptr);

    // Destroy command pool
    if (vkCommandPool != VK_NULL_HANDLE) vkDestroyCommandPool(vkDevice, vkCommandPool, nullptr);

    // Destroy swapchains
    for (auto swapchain : xrSwapchains) {
        if (swapchain != XR_NULL_HANDLE) {
            xrDestroySwapchain(swapchain);
        }
    }

    // Destroy OpenXR objects
    if (xrPlaySpace != XR_NULL_HANDLE) xrDestroySpace(xrPlaySpace);
    if (xrSession != XR_NULL_HANDLE) xrDestroySession(xrSession);
    if (xrInstance != XR_NULL_HANDLE) xrDestroyInstance(xrInstance);

    // Destroy Vulkan objects
    if (vkDevice != VK_NULL_HANDLE) vkDestroyDevice(vkDevice, nullptr);
    if (vkInstance != VK_NULL_HANDLE) vkDestroyInstance(vkInstance, nullptr);

    std::cout << "Shutdown complete" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Simple VR Player ===" << std::endl;
    std::cout << "Phase 6: 180°/360° Sphere VR Video Support" << std::endl;
    std::cout << std::endl;

    // Parse command line arguments
    const char* videoFile = nullptr;
    bool sbsMode = false;
    bool debugMode = false;
    SimpleVRPlayer::VideoMode videoMode = SimpleVRPlayer::MODE_FLAT;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sbs") == 0 || strcmp(argv[i], "-s") == 0) {
            sbsMode = true;
            std::cout << "Side-by-side 3D mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
            debugMode = true;
            std::cout << "Debug/performance mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--180") == 0 || strcmp(argv[i], "-1") == 0) {
            videoMode = SimpleVRPlayer::MODE_SPHERE_180;
            std::cout << "180° sphere mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--360") == 0 || strcmp(argv[i], "-3") == 0) {
            videoMode = SimpleVRPlayer::MODE_SPHERE_360;
            std::cout << "360° sphere mode enabled" << std::endl;
        } else if (argv[i][0] != '-') {
            videoFile = argv[i];
        }
    }

    SimpleVRPlayer player;

    // Set modes BEFORE initializing (geometry creation depends on video mode)
    if (sbsMode) {
        player.setSBSMode(true);
    }
    if (debugMode) {
        player.setDebugMode(true);
    }
    player.setVideoMode(videoMode);

    // Initialize VR system (this creates geometry based on videoMode)
    if (!player.initialize()) {
        std::cerr << "Failed to initialize VR player" << std::endl;
        return 1;
    }

    // Load video file if provided
    if (videoFile) {
        if (!player.loadVideo(videoFile)) {
            std::cerr << "Warning: Failed to load video, will show test pattern" << std::endl;
        }
    } else {
        std::cout << "No video file specified, showing test pattern" << std::endl;
        std::cout << "Usage: ./simple-vr-player [options] <video_file.mp4>" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --sbs, -s     Enable side-by-side 3D mode" << std::endl;
        std::cout << "  --180, -1     Enable 180° sphere mode" << std::endl;
        std::cout << "  --360, -3     Enable 360° sphere mode" << std::endl;
        std::cout << "  --debug, -d   Enable performance metrics (FPS, timing)" << std::endl;
        std::cout << std::endl;
    }

    // Run VR rendering loop
    player.run();
    player.shutdown();

    return 0;
}
