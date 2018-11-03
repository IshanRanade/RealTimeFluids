#include "app.h"

#include <windows.h>

#include "fluid.h"

App::App() {
	camera = new Camera();
	
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	window = glfwCreateWindow(width, height, "Real Time Fluid Sim", NULL, NULL);
	glfwMakeContextCurrent(window);

	initGL();

	initSim();
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


	float data[] = {
		-0.5f, -0.5f, 0.0f,
		1, 0, 0,
		0.5f,  0.5f, 0.0f,
		1,0,0,
		0.5f, -0.5f, 0.0f,
		1,0,0,
		-0.5f,  0.5f, 0.0f,
		1,0,0
	};

	unsigned int indices[] = {
		0, 1, 2, 3
	};

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

	GLuint vsPosLocation = glGetAttribLocation(shaderProgram, "vs_Pos");
	glEnableVertexAttribArray(vsPosLocation);
	glVertexAttribPointer(vsPosLocation, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	
	GLuint vsColorLocation = glGetAttribLocation(shaderProgram, "vs_Color");
	glEnableVertexAttribArray(vsColorLocation);
	glVertexAttribPointer(vsColorLocation, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)(3 * sizeof(float)));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void App::start() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		P = glm::frustum<float>(-camera->scale * ((float)width) / ((float)height),
			camera->scale * ((float)width / (float)height),
			-camera->scale, camera->scale, 1.0, 1000.0);

		M = glm::mat4();

		V = 
			glm::translate(glm::mat4(), glm::vec3(camera->x_trans, camera->y_trans, camera->z_trans))
			* glm::rotate(glm::mat4(), camera->x_angle, glm::vec3(1.0f, 0.0f, 0.0f))
			* glm::rotate(glm::mat4(), camera->y_angle, glm::vec3(0.0f, 1.0f, 0.0f));

		glm::mat3 MV_normal = glm::transpose(glm::inverse(glm::mat3(V) * glm::mat3(M)));
		glm::mat4 MV = V * M;
		MVP = P * MV;

		draw();
	}

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	glfwDestroyWindow(window);
	glfwTerminate();
}

void App::draw() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderProgram);

	GLuint uMVP = glGetUniformLocation(shaderProgram, "u_MVP");
	glUniformMatrix4fv(uMVP, 1, GL_FALSE, &MVP[0][0]);

	glBindVertexArray(VAO);
	glDrawElements(GL_POINTS, 4, GL_UNSIGNED_INT, 0);

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
