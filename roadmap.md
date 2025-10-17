# Simple VR Player - Development Roadmap

**Current Status:** Phase 6 complete - 180°/360° sphere rendering (pending verification with headset)
**Performance:** 60 FPS sustained @ 60Hz (capable of 90-120 FPS)
**Next Phase:** Phase 7 - Controls and Polish

---

## ✅ Completed Phases

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | OpenXR boilerplate | ✅ Complete |
| 2 | Basic rendering | ✅ Complete |
| 3 | Video decoding (FFmpeg) | ✅ Complete |
| 4 | Dynamic texture upload | ✅ Complete |
| 5 | Side-by-side 3D support | ✅ Complete (pending verification) |
| 5.5 | CUDA-Vulkan interop optimization | ✅ Complete |
| 5.6 | 3D rendering pipeline with shaders | ✅ Complete |
| 6 | 180°/360° sphere rendering | ✅ Complete (pending verification) |

**See `CLAUDE.md` for implementation details and current architecture.**

---

## 🎯 Priority: Phase 5.6 - Enable 90Hz/120Hz Display Modes

**Goal:** Unlock higher refresh rates to fully utilize performance headroom

**Why This Matters:**
- Current: 60 FPS @ 60Hz (10.2ms vsync throttling)
- System work: Only 6.4ms per frame
- **120Hz potential**: 8.3ms frame budget - system can handle it!
- **Benefit**: 2x smoother motion, reduced motion sickness

### Investigation Steps

#### Step 1: Check Monado Driver Configuration

The PSVR2 driver is configured for 120Hz:
```c
// From /home/mike/src/monado/src/xrt/drivers/psvr2/psvr2.c:1087
hmd->base.hmd->screens[0].nominal_frame_interval_ns = time_s_to_ns(1.0f / 120.0f);
```

**Verify this is active:**
```bash
cd /home/mike/src/monado/src/xrt/drivers/psvr2
grep -n "nominal_frame_interval_ns" psvr2.c
# Should show line 1087 with 120.0f
```

#### Step 2: Check Monado Service Logs

**Start Monado with verbose logging:**
```bash
cd /home/mike/src/monado/build
XRT_LOG=trace ./src/xrt/targets/service/monado-service 2>&1 | tee monado.log
```

**Look for these in the logs:**
- Display mode detection
- Refresh rate selection
- PSVR2 initialization messages
- Compositor configuration

**Search for refresh rate info:**
```bash
grep -i "refresh\|frame.*interval\|hz\|120\|90\|60" monado.log
```

#### Step 3: Check OpenXR Runtime Info

**Query display refresh rate from your VR player:**

Add this debug code to `SimpleVRPlayer::initialize()` after session creation:

```cpp
// Query available refresh rates
XrSystemProperties systemProps = {XR_TYPE_SYSTEM_PROPERTIES};
xrGetSystemProperties(xrInstance, systemId, &systemProps);

std::cout << "System Name: " << systemProps.systemName << std::endl;

// Check view configuration
XrViewConfigurationProperties viewProps = {XR_TYPE_VIEW_CONFIGURATION_PROPERTIES};
xrGetViewConfigurationProperties(xrInstance, systemId,
    XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, &viewProps);

std::cout << "View config FOV mutable: " << viewProps.fovMutable << std::endl;

// Query swapchain formats and refresh rates
// Note: There's no standard OpenXR extension for querying refresh rates yet
// You may need to check XrFrameState.predictedDisplayPeriod
```

**During runtime, log frame intervals:**
```cpp
// In your frame loop, after xrWaitFrame:
double displayPeriod = frameState.predictedDisplayPeriod / 1e9; // Convert to seconds
double refreshRate = 1.0 / displayPeriod;
std::cout << "Display period: " << displayPeriod*1000 << "ms, Rate: " << refreshRate << "Hz" << std::endl;
```

#### Step 4: Check Compositor Settings

**Look for Monado configuration files:**
```bash
# Check for config files
ls -la ~/.config/monado/
ls -la ~/.local/share/monado/

# Check environment variables
env | grep -i monado
env | grep -i xr
```

**Search Monado source for refresh rate configuration:**
```bash
cd /home/mike/src/monado/src/xrt/compositor
grep -r "refresh\|frame.*rate\|display.*mode" .
```

#### Step 5: PSVR2 Hardware Detection

**Check if PSVR2 is reporting capabilities correctly:**

Look in Monado logs for:
- EDID information from PSVR2
- Supported display modes
- Active display mode selection

**USB device info:**
```bash
lsusb -v | grep -A 20 "Sony"
# Look for PSVR2 device and any mode information
```

#### Step 6: Try Forcing Display Mode (If Configurable)

**Check if Monado has environment variables to force refresh rate:**
```bash
cd /home/mike/src/monado
grep -r "XRT_\|MONADO_" . | grep -i "refresh\|rate\|mode"
```

**Potential approaches:**
- Environment variable to force 120Hz
- Monado runtime configuration file
- OpenXR layer settings

### Expected Outcomes

**Success indicators:**
- Monado logs show "120Hz" or "8.33ms frame interval"
- Your `--debug` output shows ~8.3ms OpenXR time instead of ~16.6ms
- Smooth 120 FPS playback

**If still 60Hz:**
- Document where the limitation is coming from
- Check if PSVR2 PC adapter has firmware limitations
- Investigate Monado compositor code for forced modes

### Fallback: 90Hz Mode

If 120Hz isn't achievable, try 90Hz:
- 11.1ms frame budget
- Your 6.4ms work leaves 4.7ms headroom
- Still a significant improvement over 60Hz

### Testing

Once higher refresh rate is enabled:

```bash
cd /home/mike/src/simple-vr-player/build
export IPC_IGNORE_VERSION=1
export XR_RUNTIME_JSON=/home/mike/src/monado/build/openxr_monado-dev.json
./simple-vr-player --debug --sbs video.mp4
```

**Look for in debug output:**
- Frame time should drop to ~8-11ms (from 16.6ms)
- OpenXR time should show vsync at new rate
- Smoother motion in headset

**Estimated Time:** 2-3 hours investigation + testing

**See `PHASE6_IMPLEMENTATION.md` for details on completed 180°/360° sphere rendering.**

---

## 📋 Phase 7: Controls and Polish

**Goal:** Add playback controls and quality-of-life features

**Priority:** Medium (after Phase 6)

### 7.1 Basic Controls

Keyboard controls via OpenXR action system or SDL:

- **Spacebar:** Play/Pause
- **Left/Right Arrow:** Seek -5s / +5s
- **Up/Down Arrow:** Volume control (when audio added)
- **Q/E:** Zoom in/out (adjust sphere radius or quad distance)
- **W:** Reset view position
- **ESC:** Quit

### 7.2 Audio Synchronization

**Current:** Video-only playback
**New:** Synchronized audio + video

```cpp
// Use FFmpeg's audio decoding + SDL2 for playback
// Sync video PTS to audio clock

double audioClock = getAudioTime();
double videoPTS = frame->pts * av_q2d(videoStream->time_base);

if (videoPTS > audioClock + 0.016) {
    // Video ahead - sleep
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
} else if (videoPTS < audioClock - 0.016) {
    // Video behind - drop frame
    continue;
}
```

**Dependencies:**
```bash
sudo apt install libsdl2-dev libasound2-dev
```

### 7.3 On-Screen Display (Optional)

Render UI overlay in VR:
- Playback time / duration
- Volume indicator
- File name
- Current mode (mono/sbs/180°/360°)

**Estimated Time:** 2-4 hours

---

## 🚀 Performance Optimization (Optional)

**Current Performance:** 60 FPS @ 60Hz (6.4ms work, 10.2ms vsync)

### Optional Micro-Optimizations

These are **only needed for 120Hz operation** - current performance is excellent.

#### Medium Impact (0.5-1ms)

**1. Pre-allocate Eye Rendering Command Buffers**
- Gain: ~0.3ms
- File: `src/main.cpp` (renderEye function)

**2. Async CUDA Kernel Execution**
- Gain: ~0.5-1ms
- File: `src/cuda_interop.cu`
- Use CUDA streams instead of synchronous execution

#### Low Impact (<0.5ms)

**3. Optimize Image Layout Transitions**
- Gain: ~0.1-0.2ms
- Keep images in GENERAL layout

**4. Batch Eye Rendering**
- Gain: ~0.2-0.3ms
- Single vkQueueSubmit for both eyes

### Recommended: Enable Higher Refresh Rate

**Better approach than micro-optimizations:**

Enable 90Hz or 120Hz mode in PSVR2/Monado settings to utilize existing performance headroom.

**See `performance.md` for detailed analysis and benchmarks.**

---

## 🎯 Future Enhancements (Beyond Phase 7)

- **Multi-format support:** HEVC, VP9, AV1 codecs
- **Playlist support:** Queue multiple videos
- **VR controller support:** Hand tracking and controller input
- **Subtitle support:** Render SRT/ASS subtitles in VR
- **Snapshot/recording:** Capture screenshots or record sessions
- **Network streaming:** RTSP, HLS, YouTube support

---

## 📚 Resources

### Documentation
- [OpenXR Specification](https://registry.khronos.org/OpenXR/specs/1.0/html/)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

### Reference Code
- **hello_xr:** `/home/mike/src/OpenXR-SDK-Source/src/tests/hello_xr/`
- **Monado compositor:** `/home/mike/src/monado/src/xrt/compositor/`

---

**Last Updated:** 2025-10-16
**Version:** 2.0 (Streamlined for future work)
