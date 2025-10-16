#pragma once

#include <cmath>
#include <openxr/openxr.h>

// Simple matrix utilities for OpenXR/Vulkan rendering
namespace MathUtils {

// 4x4 matrix (column-major order, compatible with Vulkan/OpenGL)
struct Matrix4x4 {
    float m[16];

    Matrix4x4() {
        for (int i = 0; i < 16; i++) m[i] = 0.0f;
    }

    static Matrix4x4 Identity() {
        Matrix4x4 mat;
        mat.m[0] = mat.m[5] = mat.m[10] = mat.m[15] = 1.0f;
        return mat;
    }

    static Matrix4x4 Translation(float x, float y, float z) {
        Matrix4x4 mat = Identity();
        mat.m[12] = x;
        mat.m[13] = y;
        mat.m[14] = z;
        return mat;
    }

    static Matrix4x4 Scale(float x, float y, float z) {
        Matrix4x4 mat = Identity();
        mat.m[0] = x;
        mat.m[5] = y;
        mat.m[10] = z;
        return mat;
    }

    static Matrix4x4 Multiply(const Matrix4x4& a, const Matrix4x4& b) {
        Matrix4x4 result;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                float sum = 0.0f;
                for (int i = 0; i < 4; i++) {
                    sum += a.m[row + i * 4] * b.m[i + col * 4];
                }
                result.m[row + col * 4] = sum;
            }
        }
        return result;
    }

    // Create projection matrix from OpenXR FOV
    static Matrix4x4 CreateProjectionFov(const XrFovf& fov, float nearZ, float farZ) {
        const float tanLeft = tanf(fov.angleLeft);
        const float tanRight = tanf(fov.angleRight);
        const float tanDown = tanf(fov.angleDown);
        const float tanUp = tanf(fov.angleUp);

        const float tanWidth = tanRight - tanLeft;
        const float tanHeight = tanUp - tanDown;

        Matrix4x4 result;
        result.m[0] = 2.0f / tanWidth;
        result.m[5] = 2.0f / tanHeight;
        result.m[8] = (tanRight + tanLeft) / tanWidth;
        result.m[9] = (tanUp + tanDown) / tanHeight;
        result.m[10] = -(farZ + nearZ) / (farZ - nearZ);
        result.m[11] = -1.0f;
        result.m[14] = -(farZ * (nearZ + nearZ)) / (farZ - nearZ);

        return result;
    }

    // Create view matrix from OpenXR pose
    static Matrix4x4 CreateViewMatrix(const XrPosef& pose) {
        // Extract rotation (quaternion to matrix)
        const float x = pose.orientation.x;
        const float y = pose.orientation.y;
        const float z = pose.orientation.z;
        const float w = pose.orientation.w;

        Matrix4x4 rotation;
        rotation.m[0] = 1.0f - 2.0f * (y * y + z * z);
        rotation.m[1] = 2.0f * (x * y + w * z);
        rotation.m[2] = 2.0f * (x * z - w * y);

        rotation.m[4] = 2.0f * (x * y - w * z);
        rotation.m[5] = 1.0f - 2.0f * (x * x + z * z);
        rotation.m[6] = 2.0f * (y * z + w * x);

        rotation.m[8] = 2.0f * (x * z + w * y);
        rotation.m[9] = 2.0f * (y * z - w * x);
        rotation.m[10] = 1.0f - 2.0f * (x * x + y * y);

        rotation.m[15] = 1.0f;

        // Invert position (view matrix is inverse of camera transform)
        Matrix4x4 translation = Translation(-pose.position.x, -pose.position.y, -pose.position.z);

        // View matrix = rotation^T * translation
        Matrix4x4 rotationTranspose;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                rotationTranspose.m[row + col * 4] = rotation.m[col + row * 4];
            }
        }

        return Multiply(rotationTranspose, translation);
    }
};

} // namespace MathUtils
