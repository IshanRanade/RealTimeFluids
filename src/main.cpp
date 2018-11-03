#include "app.h"

App *app;

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(app->window, GL_TRUE);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (action == GLFW_PRESS)
	{
		if (button == GLFW_MOUSE_BUTTON_LEFT)
		{
			app->mouseState = app->ROTATE;
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			app->mouseState = app->TRANSLATE;
		}

	}
	else if (action == GLFW_RELEASE)
	{
		app->mouseState = app->NONE;
	}
}
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos) {
	const double s_r = 0.01;
	const double s_t = 0.01;

	double diffx = xpos - app->lastx;
	double diffy = ypos - app->lasty;
	app->lastx = xpos;
	app->lasty = ypos;

	if (app->mouseState == app->ROTATE)
	{
		//rotate
		app->camera->x_angle += (float)s_r * diffy;
		app->camera->y_angle += (float)s_r * diffx;
	}
	else if (app->mouseState == app->TRANSLATE)
	{
		//translate
		app->camera->x_trans += (float)(s_t * diffx);
		app->camera->y_trans += (float)(-s_t * diffy);
	}
}

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset) {
	const double s = 1.0;	// sensitivity
	app->camera->z_trans += (float)(s * yoffset);
}

int main() {
	app = new App();

	glfwSetKeyCallback(app->window, keyCallback);
	glfwSetMouseButtonCallback(app->window, mouseButtonCallback);
	glfwSetCursorPosCallback(app->window, mouseMotionCallback);
	glfwSetScrollCallback(app->window, mouseWheelCallback);

	app->start();
}
