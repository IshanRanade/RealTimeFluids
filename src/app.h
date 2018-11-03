#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <gl/gl.h> 

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>  
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


struct Camera {
	float scale = 1.0f;
	float x_trans = 0.0f, y_trans = 0.0f, z_trans = -10.0f;
	float x_angle = 0.0f, y_angle = 0.0f;
};

class App {
public:
	App();

	int width = 800;
	int height = 800;

	GLFWwindow *window;

	GLuint VAO;
	GLuint VBO;
	GLuint EBO;
	GLuint shaderProgram;

	Camera *camera;

	void start();
	void initGL();
	std::string readFileAsString(std::string filename);

	glm::mat4 M;
	glm::mat4 V;
	glm::mat4 P;
	glm::mat4 MVP;

	void draw();

	enum ControlState { NONE = 0, ROTATE, TRANSLATE };
	ControlState mouseState = NONE;

	double lastx = (double)width / 2;
	double lasty = (double)height / 2;
};
