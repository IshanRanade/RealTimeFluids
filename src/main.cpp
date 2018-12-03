#include "app.h"

App *app;

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(app->window, GL_TRUE);
    }
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		app->refresh = true;
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    app->leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    app->rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    app->middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (xpos == app->lastX || ypos == app->lastY) return; // otherwise, clicking back into window causes re-start
    if (app->leftMousePressed) {
        // compute new camera parameters
        app->camera->phi -= (xpos - app->lastX) / app->width;
        app->camera->theta -= (ypos - app->lastY) / app->height;
        app->camera->theta = std::fmax(0.001f, std::fmin(app->camera->theta, PI));
        app->camchanged = true;
    }
    else if (app->rightMousePressed) {
        app->camera->zoom += 10.0f * (ypos - app->lastY) / app->height;
        app->camera->zoom = std::fmax(0.1f, app->camera->zoom);
        app->camchanged = true;
    }
    else if (app->middleMousePressed) {
        glm::vec3 forward = app->camera->view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = app->camera->right;
        right.y = 0.0f;
        right = glm::normalize(right);

        app->camera->lookAt -= (float)(xpos - app->lastX) * right * 0.01f;
        app->camera->lookAt += (float)(ypos - app->lastY) * forward * 0.01f;
        app->camchanged = true;
    }
    app->lastX = xpos;
    app->lastY = ypos;
}

int main() {
    app = new App();

    glfwSetKeyCallback(app->window, keyCallback);
    glfwSetMouseButtonCallback(app->window, mouseButtonCallback);
    glfwSetCursorPosCallback(app->window, mousePositionCallback);

    app->start();
}
