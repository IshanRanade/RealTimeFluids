#version 400

uniform mat4 u_MVP;

in vec3 vs_Pos;
in vec3 vs_Color;

out vec3 fs_Color;

void main()
{	
	fs_Color = vs_Color;

	gl_Position = u_MVP * vec4(vs_Pos.x, vs_Pos.y, vs_Pos.z, 1.0);
}
