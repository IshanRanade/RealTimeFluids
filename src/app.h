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
    glm::vec2 pixelLength;

	Camera(int width, int height, glm::vec3 lookAtPosition) {
		resolution.x = width;
		resolution.y = height;

		position = glm::vec3(-lookAtPosition.x * 1.0f, -lookAtPosition.y * 0.7f, -lookAtPosition.z * 1.0f);
		lookAt = glm::vec3(lookAtPosition.x / 2.0f, lookAtPosition.y / 2.0f, lookAtPosition.z / 2.0f);
		up = glm::vec3(0, 1, 0);
        zoom = glm::length(position - lookAt);

        fov.y = 45.0;
        const float yscaled = glm::tan(fov.y * (PI / 180.0f));
        const float xscaled = (yscaled *  resolution.x) / resolution.y;
        pixelLength = glm::vec2(2 * xscaled / resolution.x, 2 * yscaled / resolution.y);

		view = glm::normalize(lookAt - position);
		right = glm::normalize(glm::cross(view, up));

        const glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
        const glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
        phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
        theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	}

	void update() {
		position.x = zoom * sin(phi) * sin(theta);
		position.y = zoom * cos(theta);
		position.z = zoom * cos(phi) * sin(theta);

		view = -glm::normalize(position);
		glm::vec3 v = view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		up = glm::cross(r, v);
		right = r;

		position += lookAt;
	}
};

class App {
public:
	App();

	int width = 1920;
	int height = 1080;

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

	double lastX = (double)width / 2;
	double lastY = (double)height / 2;
	bool leftMousePressed = false;
	bool rightMousePressed = false;
	bool middleMousePressed = false;
	bool camchanged = true;
};
