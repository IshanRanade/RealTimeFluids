#pragma once

#include "app.h"

#include <glm/glm.hpp>

static enum CellType {
	FLUID,
	AIR
};

static struct GridCell {
	CellType cellType;
	glm::vec3 worldPosition;
	float pressure;
	float tempPressure;
	glm::vec3 velocity;
	glm::vec3 tempVelocity;
};

static struct MarkerParticle {
	glm::vec3 worldPosition;
	glm::vec3 color;
};

static struct Grid {
    GridCell* dev_cells;
    float* dev_nnzA;
    int* dev_colIndA;
    float* dev_X;
    float* dev_B;

    int level;
    int gridX, gridY, gridZ, numCells;
};

#define RAY_CAST 1
#define TIME_STEP (1.0f / 30.0f)

#define GRID_X 128
#define GRID_Y 128
#define GRID_Z 128

#define NUM_CELLS (GRID_X * GRID_Y * GRID_Z)
#define CELL_WIDTH 1.0f
#define WIDTH_DIV_TIME (CELL_WIDTH / TIME_STEP)

#define NUM_MARKER_PARTICLES 2000
#define PARTICLE_RADIUS 3.0f

#define DENSITY 1.0f
#define VISCOSITY 1.0f

#define BLOCK_SIZE 256
#define BLOCKS_PARTICLES ((NUM_MARKER_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define BLOCKS_CELLS ((NUM_CELLS + BLOCK_SIZE - 1) / BLOCK_SIZE)

static GridCell* dev_gridCells;
static MarkerParticle* dev_markerParticles;

static Grid* grids;
static int GRID_LEVELS;

/*static int nnz;
static float* csrValA;
static int* csrRowPtrA;
static int* csrColIndA;
static float* vecB;
static float* vecX;*/

void initSim();
void iterateSim();
void fillVBOsWithMarkerParticles(void *vbo);
void raycastPBO(uchar4* pbo, glm::vec3 camPos, Camera camera);