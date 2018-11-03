#pragma once

#include <glm/glm.hpp>

static enum CellType {
	FLUID,
	AIR,
	NONE
};

static struct GridCell {
	CellType cellType;
	glm::vec3 worldPosition;
	float pressure;
	float oldPressure;
	glm::vec3 velocity;
	glm::vec3 oldVelocity;
};

static struct MarkerParticle {
	glm::vec3 worldPosition;
	glm::vec3 color;
};

static float TIME_STEP = 1.0 / 30.0;

static int GRID_X = 16;
static int GRID_Y = 32;
static int GRID_Z = 16;

static int NUM_CELLS = GRID_X * GRID_Y * GRID_Z;

static float CELL_WIDTH = 1.0;
static int NUM_MARKER_PARTICLES = 100;

static int blockSize = 256;

static GridCell *dev_gridCells;
static MarkerParticle *dev_markerParticles;

void initSim();
void fillVBOsWithMarkerParticles(void *vbo);