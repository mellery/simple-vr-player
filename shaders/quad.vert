#version 450

// Vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

// Output to fragment shader
layout(location = 0) out vec2 fragTexCoord;

// Push constants for per-eye transformation
layout(push_constant) uniform PushConstants {
    mat4 mvp;       // Model-View-Projection matrix (64 bytes)
    vec2 uvOffset;  // UV offset for SBS mode (8 bytes)
    vec2 uvScale;   // UV scale for SBS mode (8 bytes)
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord * pc.uvScale + pc.uvOffset;
}
