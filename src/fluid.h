#pragma once

#include <glm/glm.hpp>

static struct GridCell {
	glm::vec3 worldPosition;
	float pressure;
	glm::vec3 velocityX;
	glm::vec3 velocityY;
	glm::vec3 velocityZ;
};

static int GRID_X = 32;
static int GRID_Y = 64;
static int GRID_Z = 32;

static int NUM_CELLS = GRID_X * GRID_Y * GRID_Z;

static int blockSize = 256;

static GridCell *dev_gridCells;

void initSim();
void fillVBOs(void *vbo);