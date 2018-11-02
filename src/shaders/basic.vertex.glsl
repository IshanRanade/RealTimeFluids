#version 400

uniform mat4 u_Model;
uniform mat3 u_ModelInvTr;
uniform mat4 u_View;
uniform mat4 u_Proj;

in vec3 vs_Pos;

void main()
{	
	//vec4 modelposition = u_Model * vs_Pos;
	//gl_Position = u_Proj * u_View * modelposition;

	gl_Position = vec4(vs_Pos.x, vs_Pos.y, vs_Pos.z, 1.0);
}
