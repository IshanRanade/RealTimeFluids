#include "app.h"

void App::start() {
	init();
	mainLoop();
}

bool App::init() {
	glfwInit();
	window = glfwCreateWindow(width, height, "Real Time Fluid Sim", NULL, NULL);
	glfwMakeContextCurrent(window);
	//glfwSetKeyCallback()
	glfwSetKeyCallback(window, keyCallback);
	return true;
}

void App::mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		glfwSwapBuffers(window);
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}