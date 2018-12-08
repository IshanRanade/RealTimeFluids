#version 400

in vec3 fs_Color;

void main()
{
	gl_FragColor = vec4(fs_Color, 1.0);
}
