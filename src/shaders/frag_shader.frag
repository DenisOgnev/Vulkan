#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coord;

layout(binding = 1) uniform sampler2D tex_sampler;
layout(binding = 2) uniform sampler2D tex_sampler2;

layout(location = 0) out vec4 out_color;

void main()
{
    out_color = mix(texture(tex_sampler, frag_tex_coord), texture(tex_sampler2, frag_tex_coord), 0.2);
    //out_color = vec4(frag_color, 1.0);
}