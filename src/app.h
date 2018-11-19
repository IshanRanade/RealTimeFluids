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
#include <math.h>


const float PI = 3.14159265358979f;


struct Camera {
	float zoom;
	float phi;
	float theta;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 lookAt;
	glm::vec2 fov;
	glm::ivec3 resolution;

	Camera(int width, int height) {
		zoom = 1.0f;

		resolution.x = width;
		resolution.y = height;

		position = glm::vec3(50, 50, 50);
		lookAt = glm::vec3(0, 5, 0);
		up = glm::vec3(0, 1, 0);

		fov.y = 45.0;

		float yscaled = tan(fov.y * (PI / 180));
		float xscaled = (yscaled * resolution.x) / resolution.y;
		float fovx = (atan(xscaled) * 180) / PI;
		fov.y = 45.0;

		view = glm::normalize(lookAt - position);
		right = glm::normalize(glm::cross(view, up));
	}

	void update() {

	}
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
	GLuint PBO;
	GLuint displayTexture;
	GLuint shaderProgram;

	Camera *camera;

	void start();
	void initGL();
	void runSim();
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
