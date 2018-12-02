#pragma once

#include "app.h"

#include <glm/glm.hpp>

static enum CellType {
	FLUID,
	AIR
};

static struct GridCell {
	CellType cellType;
	float pressure;
	glm::vec3 velocity;
	glm::vec3 tempVelocity;
};

static struct MarkerParticle {
	glm::vec3 worldPosition;
	glm::vec3 color;
};

static struct Grid {
    float* dev_valA;
    int* dev_colIndA;
    float* dev_X;
    //float* dev_tempX;
    float* dev_B;
    float* dev_terrain;
    float* dev_tallHeight;
    //float *dev_r;

    int level;
    int sizeX, sizeY, sizeZ, numCells;
};

#define RAY_CAST 0
#define SPHERE_MARCH 0
#define TIME_STEP (1.0f / 30.0f)

#define GRID_X 16
#define GRID_Y 32
#define GRID_Z 16

#define NUM_CELLS (GRID_X * GRID_Y * GRID_Z)
#define CELL_WIDTH 1.0f
#define WIDTH_DIV_TIME (CELL_WIDTH / TIME_STEP)

#define NUM_MARKER_PARTICLES NUM_CELLS
#define PARTICLE_RADIUS 0.05f

#define MAX_VELOCITY 10.0f
#define GRAVITY 100.0f
#define VISCOSITY 0.5f
#define FLUID_DENSITY 1.0f
#define AIR_DENSITY 1.0f
#define GAUSS_ITERATIONS 32

#define BLOCK_SIZE 256
#define BLOCKS_PARTICLES ((NUM_MARKER_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define BLOCKS_CELLS ((NUM_CELLS + BLOCK_SIZE - 1) / BLOCK_SIZE)

static GridCell* dev_gridCells;
static MarkerParticle* dev_markerParticles;
static MarkerParticle* markerParticles;

static Grid* grids;
static int MAX_GRID_LEVEL;

static float* vecX;
static float* vecB;
static float* valA;
static int* colIndA;

void initSim();
void freeSim();
void iterateSim();
void fillVBOsWithMarkerParticles(void *vbo);
void raycastPBO(uchar4* pbo, Camera camera);