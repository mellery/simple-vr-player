cmake_minimum_required(VERSION 3.10)

# Use GCC 12 for CUDA (CUDA 12.1 doesn't support GCC 13)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-12)

project(SimpleVRPlayer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENXR REQUIRED openxr)
find_package(Vulkan REQUIRED)
find_package(CUDAToolkit REQUIRED)

# FFmpeg libraries
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

add_executable(simple-vr-player
    src/main.cpp
    src/cuda_interop.cu
)

target_include_directories(simple-vr-player PRIVATE
    ${OPENXR_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(simple-vr-player
    ${OPENXR_LIBRARIES}
    ${Vulkan_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    CUDA::cuda_driver
    CUDA::cudart
)

# Compiler warnings
target_compile_options(simple-vr-player PRIVATE -Wall -Wextra)
