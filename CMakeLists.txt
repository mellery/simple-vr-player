cmake_minimum_required(VERSION 3.10)
project(SimpleVRPlayer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENXR REQUIRED openxr)
find_package(Vulkan REQUIRED)

# FFmpeg libraries
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

add_executable(simple-vr-player
    src/main.cpp
)

target_include_directories(simple-vr-player PRIVATE
    ${OPENXR_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

target_link_libraries(simple-vr-player
    ${OPENXR_LIBRARIES}
    ${Vulkan_LIBRARIES}
    ${FFMPEG_LIBRARIES}
)

# Compiler warnings
target_compile_options(simple-vr-player PRIVATE -Wall -Wextra)
