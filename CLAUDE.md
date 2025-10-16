# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simple VR Player is an OpenXR-based VR video player for Linux with PSVR2 support. It uses Vulkan for rendering and FFmpeg for video decoding, with hardware acceleration support (NVDEC). The application runs on the Monado OpenXR runtime and displays video content in virtual reality.

**Current Status:** Phase 6 complete - CUDA-Vulkan interop with 60 FPS performance achieved

## Build System

### Building the Project

The project requires CUDA support for Vulkan-CUDA interop:

```bash
# From repository root
cd build
cmake ..
make -j$(nproc)
```

**Required:** NVIDIA GPU with CUDA support (GTX 900 series or newer)

### Running the Application

The application requires the Monado OpenXR runtime to be running:

```bash
# Terminal 1: Start Monado service
cd /home/mike/src/monado/build
./src/xrt/targets/service/monado-service

# Terminal 2: Set environment and run player
export IPC_IGNORE_VERSION=1
export XR_RUNTIME_JSON=/home/mike/src/monado/build/openxr_monado-dev.json
cd /home/mike/src/simple-vr-player/build
./simple-vr-player [options] <video_file>
```

### Command-Line Options

- `--sbs` or `-s`: Enable side-by-side 3D mode (splits video horizontally for stereo)
- `--debug` or `-d`: Enable debug mode with performance metrics (FPS, timing breakdown)

## Architecture

### Core Components

The application consists of:
- **src/main.cpp**: Main application (~1465 lines) - OpenXR/Vulkan setup and rendering loop
- **src/cuda_interop.cu**: CUDA-Vulkan interop implementation with NV12→RGBA conversion kernel
- **src/cuda_interop.h**: Interop interface and data structures

This architecture separates concerns while maintaining simplicity.

### Key Classes

**VideoDecoder** (main.cpp, lines 35-291)
- Wraps FFmpeg's libavcodec/libavformat APIs
- Attempts hardware decode (h264_cuvid/NVDEC) first, falls back to software
- Outputs raw NV12 frames (no CPU-side conversion)
- Minimal CPU overhead - decode only

**VulkanCudaInterop** (cuda_interop.h/cu)
- Creates Vulkan images with external memory export
- Imports Vulkan memory into CUDA via file descriptor sharing
- GPU-accelerated NV12→RGBA conversion via CUDA kernel
- Zero-copy architecture - data never leaves GPU

**SimpleVRPlayer** (main.cpp, lines 572-1404)
- Main application class orchestrating OpenXR session and rendering
- Manages OpenXR instance, system, session, swapchains (one per eye)
- Handles Vulkan device/queue setup via OpenXR requirements
- Implements frame loop with video decode → CUDA convert → render cycle
- Supports both monoscopic and side-by-side stereoscopic video

### Rendering Pipeline

The renderEye() method implements a simple blit-based approach:
1. Allocates command buffer (per-frame, potential optimization)
2. Transitions swapchain image to TRANSFER_DST_OPTIMAL
3. Blits CUDA-converted RGBA texture to swapchain image (scales automatically)
   - In SBS mode: splits source texture horizontally per eye
   - In mono mode: uses full texture for both eyes
4. Transitions swapchain image to COLOR_ATTACHMENT_OPTIMAL
5. Submits and waits for completion

**Note:** Uses synchronous vkQueueWaitIdle() for simplicity. Render time is only 0.5ms so not a bottleneck.

### Video Decoding Flow (CUDA-Vulkan Interop)

1. **Hardware Decode**: h264_cuvid (NVDEC) → GPU NV12 frames
2. **GPU→CPU Transfer**: av_hwframe_transfer_data() copies NV12 to system memory (~3.5ms)
3. **CPU→GPU Upload**: cudaMemcpy to CUDA-accessible Vulkan image (~1.5ms)
4. **GPU Conversion**: CUDA kernel converts NV12→RGBA in-place (~0.9ms)
5. **Rendering**: Vulkan blits RGBA texture to swapchain (~0.5ms)

**Performance Note:** Zero-copy architecture via external memory keeps most data on GPU. Total video pipeline: 5.9ms.

## Performance Characteristics

Current performance on PSVR2 at 60Hz (3200x1600 video, 1600x1600 per eye):
- **60 FPS sustained** with CUDA-Vulkan interop (up from 40 FPS)
- **Frame time breakdown** (16.6ms total):
  - Decode: ~3.5ms (21%) - NVDEC decode + GPU→CPU NV12 transfer
  - CUDA Upload+Convert: ~2.4ms (14%) - CPU→GPU upload + NV12→RGBA kernel
  - RenderEye: ~0.5ms (3%) - Vulkan blit operations (both eyes)
  - OpenXR: ~10.2ms (61%) - vsync throttling to maintain 60 FPS
  - Total work: ~6.4ms (39%) - **System is capable of 90-120 FPS**

**Key Achievement:** 62% reduction in video pipeline time (10.4ms → 5.9ms) via CUDA interop.

Refer to `performance.md` for detailed analysis and remaining optimization opportunities.

## Code Patterns and Conventions

### Error Handling

Two macros are used throughout:
```cpp
XR_CHECK(result, msg)  // For OpenXR calls - prints error and returns false
VK_CHECK(result, msg)  // For Vulkan calls - prints error and returns false
```

These macros are designed for early-return error handling in bool-returning functions.

### Memory Management

- **Vulkan**: Manual resource management with explicit destroy calls in shutdown()
- **FFmpeg**: RAII-style - VideoDecoder::close() frees all resources
- **OpenXR**: Handles created in initialization, destroyed in shutdown()

No smart pointers are used - lifecycle is simple and deterministic.

### Synchronization

- **CUDA conversion**: Synchronous via cudaDeviceSynchronize() (potential optimization: CUDA streams)
- **Eye rendering**: Uses vkQueueWaitIdle() for simplicity (synchronous)
- **OpenXR frame pacing**: Handled by xrWaitFrame/xrBeginFrame/xrEndFrame

### Performance Tracking

When debug mode is enabled (--debug flag):
- Tracks frame times, decode times, upload times, rendering times per frame
- Averages over 1-second windows
- Prints detailed breakdown including OpenXR overhead

See lines 1264-1306 for implementation.

## Common Development Tasks

### Adding New Command-Line Options

Parse in main() starting at line 1416. Set flags on SimpleVRPlayer instance before run().

### Modifying Video Decoding

Edit VideoDecoder class (main.cpp, lines 35-291). Key methods:
- `open()`: Codec initialization and hardware decode setup
- `readFrame()`: Decode loop, outputs raw NV12 data
- `getDuration()`: Metadata extraction

### Modifying CUDA Conversion

Edit cuda_interop.cu for GPU-side processing:
- `cudaConvertNV12ToRGBA()`: Main conversion function
- NV12 to RGBA kernel with bilinear chroma upsampling

### Changing Rendering

Edit renderEye() (lines 937-1098) for per-eye rendering logic. For SBS mode, see the blit region calculation at lines 1003-1023.

### Performance Instrumentation

Add timing code following the pattern in renderFrame() (lines 1144-1308):
```cpp
auto start = std::chrono::steady_clock::now();
// ... code to time ...
auto end = std::chrono::steady_clock::now();
double ms = std::chrono::duration<double, std::milli>(end - start).count();
```

## Known Limitations and Future Work

### Current Limitations

1. **No shader-based rendering**: Uses Vulkan blit operations instead of textured quads
2. **No audio support**: Video-only playback
3. **No seeking/playback controls**: Videos play once then loop
4. **No 360°/180° sphere projection**: Only flat-screen rendering
5. **60Hz operation**: PSVR2 hardware supports 120Hz (Monado driver configured for 120Hz at psvr2.c:1087), but current session running at 60Hz

### Remaining Optimization Opportunities

See `performance.md` for detailed analysis. Remaining improvements:

**Medium Impact (0.5-1ms):**
1. **Pre-allocate eye command buffers**: Avoid per-frame allocation (~0.2-0.3ms savings)
2. **Async CUDA kernel execution**: Use CUDA streams to overlap with Vulkan work (~0.5-1ms savings)

**Low Impact (<0.5ms):**
3. **Optimize image layout transitions**: Keep image in GENERAL layout (~0.1-0.2ms)
4. **Batch eye rendering commands**: Single vkQueueSubmit for both eyes (~0.2-0.3ms)

**Completed Optimizations:**
- ✅ Vulkan-CUDA interop (62% video pipeline improvement)
- ✅ Hardware decode (NVDEC)
- ✅ GPU-accelerated NV12→RGBA conversion

### Future Features

Roadmap outlined in `roadmap.md`:
- Phase 6: 180°/360° sphere rendering for VR videos
- Phase 7: Playback controls (seek, pause, zoom)
- Audio synchronization
- Multiple video format support

## Development Environment

### Dependencies

Required packages (Ubuntu/Debian):
```bash
sudo apt install \
    build-essential cmake pkg-config \
    libvulkan-dev vulkan-headers \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libopenxr-dev \
    nvidia-cuda-toolkit
```

**Note:** CUDA toolkit is required for Vulkan-CUDA interop.

### Hardware Requirements

- **VR Headset**: PSVR2 with PC adapter (other OpenXR headsets may work)
- **GPU**: NVIDIA GPU with Vulkan support (for h264_cuvid: GTX 900 series or newer)
- **OS**: Linux (tested on Ubuntu)

### Testing

Basic test pattern (no video):
```bash
./simple-vr-player
# Shows cyan (left) / magenta (right) test pattern
```

With video file:
```bash
./simple-vr-player video.mp4
# For side-by-side 3D:
./simple-vr-player --sbs video_sbs.mp4
```

Enable performance monitoring:
```bash
./simple-vr-player --debug video.mp4
# Prints FPS and timing breakdown every second
```

## Related Documentation

- `roadmap.md`: Complete development roadmap with implementation phases and performance status
- `performance.md`: Detailed performance analysis, benchmark results, and optimization recommendations

## Related Codebases

- **Monado OpenXR Runtime**: `/home/mike/src/monado/`
  - Source code for the OpenXR runtime used by this application
  - PSVR2 driver: `src/xrt/drivers/psvr2/`
  - Compositor: `src/xrt/compositor/`
  - Useful for understanding OpenXR integration and display configuration
