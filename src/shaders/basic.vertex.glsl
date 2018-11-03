#version 400

uniform mat4 u_MVP;

in vec3 vs_Pos;

void main()
{	
	//vec4 modelposition = u_Model * vs_Pos;
	//gl_Position = u_Proj * u_View * modelposition;

	gl_Position = u_MVP * vec4(vs_Pos.x, vs_Pos.y, vs_Pos.z, 1.0);
}
