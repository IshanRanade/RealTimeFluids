#include "app.h"

#include <windows.h>

App::App() {
	camera = new Camera();
}

void App::start() {
	init();
	mainLoop();
}

void App::init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	window = glfwCreateWindow(width, height, "Real Time Fluid Sim", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);

	initGL();
}

void App::initGL() {
	glewInit();

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	std::string vertexShaderFilename = std::string("../src/shaders/basic.vertex.glsl");
	std::string vertexShaderSourceString = readFileAsString(vertexShaderFilename);
	const char *vertexShaderSource = vertexShaderSourceString.c_str();

	std::string fragmentShaderFilename = std::string("../src/shaders/basic.frag.glsl");
	std::string fragmentShaderSourceString = readFileAsString(fragmentShaderFilename);
	const char *fragmentShaderSource = fragmentShaderSourceString.c_str();

	// Compile the vertex shader
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);
	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// Compile the fragment shader
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
	// check for shader compile errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// Link the shaders
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);





	float vertices[] = {
	   0.5f,  0.5f, 0.0f,  // top right
	   0.5f, -0.5f, 0.0f,  // bottom right
	  -0.5f, -0.5f, 0.0f,  // bottom left
	  -0.5f,  0.5f, 0.0f   // top left 
	};
	unsigned int indices[] = {
		0, 1, 3,  // first Triangle
		1, 2, 3   // second Triangle
	};
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glBindVertexArray(0);
}

void App::mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		draw();
	}

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	glfwDestroyWindow(window);
	glfwTerminate();
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void App::draw() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(shaderProgram);
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glfwSwapBuffers(window);
}

std::string App::readFileAsString(std::string filename) {
	std::string fileString;
	std::ifstream fileStream(filename);
	if (fileStream.is_open()) {
		std::stringstream sstr;
		sstr << fileStream.rdbuf();
		fileString = sstr.str();
		fileStream.close();
		return fileString;
	}
	else {
		std::cout << "file does not exist" << std::endl;
		return "";
	}
	
}