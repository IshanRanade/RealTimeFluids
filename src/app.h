#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class App {
public:
	int width = 800;
	int height = 800;

	GLFWwindow *window;

	void start();
	bool init();
	void mainLoop();
};