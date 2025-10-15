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

// FFmpeg includes
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

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
    bool readFrame(uint8_t* rgbaBuffer, int* width, int* height);  // Now outputs RGBA
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
    if (avformat_open_input(&formatCtx, filename, nullptr, nullptr) < 0) {
        std::cerr << "ERROR: Could not open video file" << std::endl;
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

bool VideoDecoder::readFrame(uint8_t* rgbaBuffer, int* width, int* height) {
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
                // Transfer hardware frame to software frame
                swFrame->format = AV_PIX_FMT_NV12;  // CUDA typically outputs NV12
                ret = av_hwframe_transfer_data(swFrame, frame, 0);
                if (ret < 0) {
                    std::cerr << "ERROR: Failed to transfer frame from GPU to CPU" << std::endl;
                    av_packet_unref(packet);
                    return false;
                }
                sourceFrame = swFrame;
            }

            // Initialize swscale context if needed
            if (!swsCtx) {
                swsCtx = sws_getContext(
                    codecCtx->width, codecCtx->height,
                    (AVPixelFormat)sourceFrame->format,
                    codecCtx->width, codecCtx->height, AV_PIX_FMT_RGBA,
                    SWS_BILINEAR, nullptr, nullptr, nullptr);

                if (!swsCtx) {
                    std::cerr << "ERROR: Could not initialize swscale context" << std::endl;
                    av_packet_unref(packet);
                    return false;
                }
            }

            // Convert frame to RGBA
            uint8_t* dest[4] = { rgbaBuffer, nullptr, nullptr, nullptr };
            int destLinesize[4] = { codecCtx->width * 4, 0, 0, 0 };

            sws_scale(swsCtx, sourceFrame->data, sourceFrame->linesize, 0, codecCtx->height,
                     dest, destLinesize);

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
    bool initialize();
    bool loadVideo(const char* filename);
    void setSBSMode(bool enabled) { sbsMode = enabled; }
    void setDebugMode(bool enabled) { debugMode = enabled; }
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

    // Video playback resources
    VideoDecoder videoDecoder;
    DynamicTexture videoTexture;
    std::vector<uint8_t> frameBuffer;
    bool videoLoaded = false;
    bool sbsMode = false;  // Side-by-side 3D mode
    bool debugMode = false;  // Debug/performance mode

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
    void renderEye(uint32_t eyeIndex, VkImage image, uint32_t width, uint32_t height);
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
    }

    std::cout << "Swapchains created successfully" << std::endl;
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

    std::cout << "Rendering resources created successfully" << std::endl;
    return true;
}

void SimpleVRPlayer::renderEye(uint32_t eyeIndex, VkImage image, uint32_t width, uint32_t height) {
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

    // Transition destination image to TRANSFER_DST_OPTIMAL
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

    vkCmdPipelineBarrier(cmdBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    if (videoLoaded) {
        // Wait for video texture upload to complete before using it
        videoTexture.waitForUpload(vkDevice);

        // Transition source video texture to TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier srcBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        srcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        srcBarrier.image = videoTexture.getImage();
        srcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        srcBarrier.subresourceRange.baseMipLevel = 0;
        srcBarrier.subresourceRange.levelCount = 1;
        srcBarrier.subresourceRange.baseArrayLayer = 0;
        srcBarrier.subresourceRange.layerCount = 1;
        srcBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &srcBarrier);

        // Blit video texture to swapchain image (scales automatically)
        VkImageBlit blitRegion{};
        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.srcSubresource.mipLevel = 0;
        blitRegion.srcSubresource.baseArrayLayer = 0;
        blitRegion.srcSubresource.layerCount = 1;

        // Set source region based on SBS mode
        if (sbsMode) {
            // Side-by-side mode: use left half for left eye, right half for right eye
            int32_t videoWidth = videoDecoder.getWidth();
            int32_t videoHeight = videoDecoder.getHeight();
            int32_t halfWidth = videoWidth / 2;

            if (eyeIndex == 0) {
                // Left eye: left half of video
                blitRegion.srcOffsets[0] = {0, 0, 0};
                blitRegion.srcOffsets[1] = {halfWidth, videoHeight, 1};
            } else {
                // Right eye: right half of video
                blitRegion.srcOffsets[0] = {halfWidth, 0, 0};
                blitRegion.srcOffsets[1] = {videoWidth, videoHeight, 1};
            }
        } else {
            // Mono mode: use full video for both eyes
            blitRegion.srcOffsets[0] = {0, 0, 0};
            blitRegion.srcOffsets[1] = {(int32_t)videoDecoder.getWidth(), (int32_t)videoDecoder.getHeight(), 1};
        }

        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.dstSubresource.mipLevel = 0;
        blitRegion.dstSubresource.baseArrayLayer = 0;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.dstOffsets[0] = {0, 0, 0};
        blitRegion.dstOffsets[1] = {(int32_t)width, (int32_t)height, 1};

        vkCmdBlitImage(cmdBuffer,
            videoTexture.getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blitRegion, VK_FILTER_LINEAR);

        // Transition source texture back to SHADER_READ_ONLY
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        srcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &srcBarrier);
    } else {
        // Fallback: Clear with test pattern colors
        VkClearColorValue clearColor;
        if (eyeIndex == 0) {
            // Left eye: Cyan
            clearColor.float32[0] = 0.0f;  // R
            clearColor.float32[1] = 0.7f;  // G
            clearColor.float32[2] = 1.0f;  // B
            clearColor.float32[3] = 1.0f;  // A
        } else {
            // Right eye: Magenta
            clearColor.float32[0] = 1.0f;  // R
            clearColor.float32[1] = 0.0f;  // G
            clearColor.float32[2] = 0.7f;  // B
            clearColor.float32[3] = 1.0f;  // A
        }

        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel = 0;
        range.levelCount = 1;
        range.baseArrayLayer = 0;
        range.layerCount = 1;

        vkCmdClearColorImage(cmdBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);
    }

    // Transition destination image to COLOR_ATTACHMENT_OPTIMAL for presentation
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    vkCmdPipelineBarrier(cmdBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

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
            if (videoDecoder.readFrame(frameBuffer.data(), &width, &height)) {
                auto decodeEnd = std::chrono::steady_clock::now();
                decodeTime = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

                // Upload new frame to GPU
                auto uploadStart = std::chrono::steady_clock::now();
                videoTexture.update(vkDevice, vkQueue, frameBuffer.data(), width, height);
                auto uploadEnd = std::chrono::steady_clock::now();
                uploadTime = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
            } else {
                // End of video - loop back to start (or could stop playback)
                std::cout << "End of video reached, looping..." << std::endl;
                videoDecoder.close();
                videoLoaded = false;  // Will show test pattern
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
            renderEye(i, image, viewConfigs[i].recommendedImageRectWidth, viewConfigs[i].recommendedImageRectHeight);
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

            // Print comprehensive timing breakdown
            std::cout << "[PERF] FPS: " << std::fixed << std::setprecision(1) << avgFPS
                      << " | Frame: " << std::setprecision(2) << avgFrame << "ms" << std::endl;
            std::cout << "       Decode: " << avgDecode << "ms"
                      << " | Upload: " << avgUpload << "ms"
                      << " | RenderEye: " << avgRenderEye << "ms"
                      << " | OpenXR: " << avgOpenXR << "ms" << std::endl;

            // Calculate and print detailed OpenXR breakdown (these are single-frame samples, not averages)
            std::cout << "       [OpenXR] WaitFrame: " << waitFrameTime << "ms"
                      << " | BeginFrame: " << beginFrameTime << "ms"
                      << " | LocateViews: " << locateViewsTime << "ms" << std::endl;
            std::cout << "       [OpenXR] AcquireSwap: " << acquireSwapchainTime << "ms"
                      << " | EndFrame: " << endFrameTime << "ms" << std::endl;

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

    if (!videoTexture.create(vkDevice, vkPhysicalDevice, vkCommandPool, videoWidth, videoHeight)) {
        std::cerr << "ERROR: Failed to create video texture" << std::endl;
        return false;
    }

    // Allocate frame buffer for decoded frames
    frameBuffer.resize(videoWidth * videoHeight * 4);  // RGBA32

    // Decode and upload first frame
    int width, height;
    if (videoDecoder.readFrame(frameBuffer.data(), &width, &height)) {
        videoTexture.update(vkDevice, vkQueue, frameBuffer.data(), width, height);
        std::cout << "First frame loaded and uploaded to GPU" << std::endl;
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

    // Destroy rendering resources
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
    std::cout << "Phase 5: 3D Side-by-Side Video Support" << std::endl;
    std::cout << std::endl;

    // Parse command line arguments
    const char* videoFile = nullptr;
    bool sbsMode = false;
    bool debugMode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sbs") == 0 || strcmp(argv[i], "-s") == 0) {
            sbsMode = true;
            std::cout << "Side-by-side 3D mode enabled" << std::endl;
        } else if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
            debugMode = true;
            std::cout << "Debug/performance mode enabled" << std::endl;
        } else if (argv[i][0] != '-') {
            videoFile = argv[i];
        }
    }

    SimpleVRPlayer player;

    // Initialize VR system
    if (!player.initialize()) {
        std::cerr << "Failed to initialize VR player" << std::endl;
        return 1;
    }

    // Set SBS mode if enabled
    if (sbsMode) {
        player.setSBSMode(true);
    }

    // Set debug mode if enabled
    if (debugMode) {
        player.setDebugMode(true);
    }

    // Load video file if provided
    if (videoFile) {
        if (!player.loadVideo(videoFile)) {
            std::cerr << "Warning: Failed to load video, will show test pattern" << std::endl;
        }
    } else {
        std::cout << "No video file specified, showing test pattern" << std::endl;
        std::cout << "Usage: ./simple-vr-player [--sbs] [--debug] <video_file.mp4>" << std::endl;
        std::cout << "  --sbs, -s    Enable side-by-side 3D mode" << std::endl;
        std::cout << "  --debug, -d  Enable performance metrics (FPS, timing)" << std::endl;
        std::cout << std::endl;
    }

    // Run VR rendering loop
    player.run();
    player.shutdown();

    return 0;
}
