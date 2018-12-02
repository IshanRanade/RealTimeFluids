#include "app.h"

#include <windows.h>
#include <vector>

#include "fluid.h"

App::App() {
	camera = new Camera(width, height);
	
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	window = glfwCreateWindow(width, height, "Real Time Fluid Sim", NULL, NULL);
	glfwMakeContextCurrent(window);

	initGL();

	initSim();
}

void App::runSim() {
	void *vbo_dptr = NULL;

	cudaGraphicsResource_t resource = 0;
	cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &resource, NULL);
	size_t size;
	cudaGraphicsResourceGetMappedPointer(&vbo_dptr, &size, resource);

	iterateSim();
	fillVBOsWithMarkerParticles(vbo_dptr);

	cudaGraphicsUnmapResources(1, &resource, NULL);
	cudaGraphicsUnregisterResource(resource);
}

void App::initGL() {
	glewInit();

	glEnable(GL_PROGRAM_POINT_SIZE_EXT);
	glPointSize(5);

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

	std::vector<int> indices;
	for (int i = 0; i < NUM_MARKER_PARTICLES; ++i) {
		indices.push_back(i);
	}

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	int num_texels = width * height;
	int num_values = num_texels * 4;
	int sizeTextureData = sizeof(GLubyte) * num_values;

	glGenBuffers(1, &PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeTextureData, NULL, GL_DYNAMIC_COPY);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &displayTexture);
	glBindTexture(GL_TEXTURE_2D, displayTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int), &*indices.begin(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * NUM_MARKER_PARTICLES, NULL, GL_STATIC_DRAW);

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
		if (camchanged) {
			camera->update();
			camchanged = false;
		}

		runSim();

		glfwPollEvents();

		P = glm::perspective(glm::radians(camera->fov.y), (float)width / (float)height, 0.1f, 1000.0f);

		M = glm::mat4();

		V = glm::lookAt(camera->position, camera->lookAt, camera->up);

		glm::mat3 MV_normal = glm::transpose(glm::inverse(glm::mat3(V) * glm::mat3(M)));
		glm::mat4 MV = V * M;
		MVP = P * MV;

		draw();
	}

    freeSim();

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	
	glBindBuffer(GL_ARRAY_BUFFER, PBO);
	glDeleteBuffers(1, &PBO);

	glfwDestroyWindow(window);
	glfwTerminate();
}

void App::draw() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/*void *pbo_dptr = NULL;

	cudaGraphicsResource_t resource = 0;
	cudaGraphicsGLRegisterBuffer(&resource, PBO, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &resource, NULL);
	size_t size;
	cudaGraphicsResourceGetMappedPointer(&pbo_dptr, &size, resource);

	raycastPBO((uchar4*)pbo_dptr, camera->position, *camera);

	cudaGraphicsUnmapResources(1, &resource, NULL);
	cudaGraphicsUnregisterResource(resource);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
	glBindTexture(GL_TEXTURE_2D, displayTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex3f(1, 1, 0);
		glTexCoord2f(0, 1); glVertex3f(1, -1, 0);
		glTexCoord2f(1, 1); glVertex3f(-1, -1, 0);
		glTexCoord2f(1, 0); glVertex3f(-1, 1, 0);
	glEnd();*/

	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);

	glUseProgram(shaderProgram);

	GLuint uMVP = glGetUniformLocation(shaderProgram, "u_MVP");
	glUniformMatrix4fv(uMVP, 1, GL_FALSE, &MVP[0][0]);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBindVertexArray(VAO);
	glDrawElements(GL_POINTS, NUM_MARKER_PARTICLES, GL_UNSIGNED_INT, 0);

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
