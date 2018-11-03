#pragma once

#include <glm/glm.hpp>

static enum CellType {
	FLUID,
	AIR
};

static struct GridCell {
	CellType cellType;
	glm::vec3 worldPosition;
	float pressure;
	float oldPressure;
	glm::vec3 velocityX;
	glm::vec3 velocityY;
	glm::vec3 velocityZ;
	glm::vec3 oldVelocityX;
	glm::vec3 oldVelocityY;
	glm::vec3 oldVelocityZ;
};

static float TIME_STEP = 1.0 / 30.0;

static int GRID_X = 32;
static int GRID_Y = 64;
static int GRID_Z = 32;

static int NUM_CELLS = GRID_X * GRID_Y * GRID_Z;

static float CELL_WIDTH = 1.0;

static int blockSize = 256;

static GridCell *dev_gridCells;

void initSim();
void fillVBOs(void *vbo);