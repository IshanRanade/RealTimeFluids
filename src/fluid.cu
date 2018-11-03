#include "fluid.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/random.h>


#define ERRORCHECK 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__device__ int getCellCompressedIndex(int x, int y, int z, int GRID_X, int GRID_Y, int GRID_Z) {
	return z * GRID_X * GRID_Y + y * GRID_X + x;
}

__device__ glm::vec3 getCellUncompressedCoordinates(int index, int GRID_X, int GRID_Y, int GRID_Z) {
	int z = index / (GRID_X * GRID_Y);
	index -= (z * GRID_X * GRID_Y);
	int y = index / GRID_X;
	int x = index % GRID_X;

	return glm::vec3(x, y, z);
}

__global__ void fillVBOData(int n, void *vbo, MarkerParticle *particles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float *vboFloat = (float*)vbo;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		// Set the position
		vboFloat[6 * index + 0] = particle.worldPosition.x;
		vboFloat[6 * index + 1] = particle.worldPosition.y;
		vboFloat[6 * index + 2] = particle.worldPosition.z;

		// Set the color
		vboFloat[6 * index + 3] = particle.color.x;
		vboFloat[6 * index + 4] = particle.color.y;
		vboFloat[6 * index + 5] = particle.color.z;
	}
}

__global__ void initializeGridCells(int n, GridCell *cells, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		int *x;
		int *y;
		int *z;
		glm::vec3 coords = getCellUncompressedCoordinates(index, GRID_X, GRID_Y, GRID_Z);

		GridCell &cell = cells[index];
		cell.worldPosition = glm::vec3(coords.x * CELL_WIDTH, coords.y * CELL_WIDTH, coords.z * CELL_WIDTH);
	}
}

__global__ void setAllGridCellsToAir(int n, GridCell *cells, CellType airType) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.cellType = airType;
	}
}

__global__ void setGridCellsWithMarkerParticleToFluid(int n, GridCell *cells, MarkerParticle *particles, CellType fluidType, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		int compressedCellIndex = getCellCompressedIndex((int)particle.worldPosition.x / CELL_WIDTH, (int)particle.worldPosition.x / CELL_WIDTH, (int)particle.worldPosition.x / CELL_WIDTH, GRID_X, GRID_Y, GRID_Z);
		
		GridCell &cell = cells[compressedCellIndex];
		cell.cellType = fluidType;
	}
}

void fillVBOsWithMarkerParticles(void *vbo) {
	int blocks = (NUM_MARKER_PARTICLES + blockSize - 1) / blockSize;
	fillVBOData<<<blocks, blockSize>>>(NUM_MARKER_PARTICLES, vbo, dev_markerParticles);
	checkCUDAError("filling VBOs with marker particle data failed");
	cudaDeviceSynchronize();
}

__global__ void generateRandomWorldPositionsForParticles(int n, MarkerParticle *particles, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		// TODO: Fix these random generators
		thrust::default_random_engine rngX = thrust::default_random_engine(index | (index << 22));
		thrust::default_random_engine rngY = thrust::default_random_engine(index | (index << 15) ^ index);
		thrust::default_random_engine rngZ = thrust::default_random_engine(index ^ (index * 13));
		thrust::uniform_real_distribution<float> u01(0, 1);

		MarkerParticle &particle = particles[index];
		particle.worldPosition.x = 1.0 * u01(rngX) * GRID_X * CELL_WIDTH;
		particle.worldPosition.y = 1.0 * u01(rngX) * GRID_Y * CELL_WIDTH;
		particle.worldPosition.z = 1.0 * u01(rngX) * GRID_Z * CELL_WIDTH;

		particle.color = glm::vec3(0.2, 0.2, 1);
	}
}

//__device__ float getInterpolatedValue(float x, float y, float z, int componentIndex) {
//
//}
//
//__device__ glm::vec3 getVelocity(float x, float y, float z) {
//
//}
//
//__device__ glm::vec3 traceParticle(float x, float y, float z, float t) {
//
//}

__global__ void backwardsParticleTrace(int n, GridCell *cells, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH, float TIME_STEP) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		// For now just use simple Euler
		glm::vec3 cellPosition = cell.worldPosition + glm::vec3(CELL_WIDTH / 2.0, CELL_WIDTH / 2.0, CELL_WIDTH / 2.0);
		glm::vec3 oldPosition = cellPosition - TIME_STEP * cell.velocity;

		int prevCellIndex = getCellCompressedIndex((int)oldPosition.x, (int)oldPosition.y, (int)oldPosition.z, GRID_X, GRID_Y, GRID_Z);
		if (prevCellIndex < 0 || prevCellIndex >= GRID_X * GRID_Y * GRID_Z) {
			return;
		}

		GridCell &otherCell = cells[prevCellIndex];
		cell.tempVelocity = otherCell.velocity;
	}
}

__global__ void applyExternalForcesToGridCells(int n, GridCell *cells, float TIME_STEP) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		// Apply gravity
		cell.velocity += glm::vec3(0, -9.8 * TIME_STEP, 0);
	}
}
__global__ void moveMarkerParticlesThroughField(int n, GridCell *cells, MarkerParticle *particles, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH, float TIME_STEP) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		// Find the cell that this particle is in
		int cellIndex = getCellCompressedIndex(particle.worldPosition.x, particle.worldPosition.y, particle.worldPosition.z, GRID_X, GRID_Y, GRID_Z);
		GridCell &cell = cells[cellIndex];
		particle.worldPosition += TIME_STEP * cell.velocity;

		particle.worldPosition.x = glm::clamp(particle.worldPosition.x, 0.0f, GRID_X * CELL_WIDTH - 0.01f);
		particle.worldPosition.y = glm::clamp(particle.worldPosition.y, 0.0f, GRID_Y * CELL_WIDTH - 0.01f);
		particle.worldPosition.z = glm::clamp(particle.worldPosition.z, 0.0f, GRID_Z * CELL_WIDTH - 0.01f);
	}
}

__global__ void swapCellVelocities(int n, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.velocity = cell.tempVelocity;
	}
}

void initSim() {
	// Allocate space for all of the grid cells
	cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));
	cudaMemset(dev_gridCells, 0, NUM_CELLS * sizeof(GridCell));

	// Allocate space for all of the marker particles
	cudaMalloc(&dev_markerParticles, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));
	cudaMemset(dev_markerParticles, 0, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));

	// Create random world positions for all of the particles
	generateRandomWorldPositionsForParticles<<<BLOCKS_PARTICLES, blockSize>>>(NUM_MARKER_PARTICLES, dev_markerParticles, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH);
	checkCUDAError("generating initial world positions for marker particles failed");
	cudaDeviceSynchronize();

	// Initialize the grid cells
	initializeGridCells<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH);
	checkCUDAError("initializing the grid cells failed");
	cudaDeviceSynchronize();
}

void iterateSim() {
	// Make all the cells temporarily air cells
	setAllGridCellsToAir<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, AIR);
	checkCUDAError("marking all cells as air cells failed");
	cudaDeviceSynchronize();

	// Mark all cells with a marker particle as a fluid cell
	setGridCellsWithMarkerParticleToFluid<<<BLOCKS_PARTICLES, blockSize>>>(NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles, FLUID, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH);
	checkCUDAError("marking all cells with a marker particle as fluid cells failed");
	cudaDeviceSynchronize();

	// Apply convection using a backwards particle trace
	backwardsParticleTrace<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH, TIME_STEP);
	checkCUDAError("convecting velocities using a backwards particle trace failed");
	cudaDeviceSynchronize();

	// Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
	swapCellVelocities<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells);
	checkCUDAError("swapping velocities in cells failed");
	cudaDeviceSynchronize();

	// Apply external forces to grid cells
	applyExternalForcesToGridCells<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, TIME_STEP);
	checkCUDAError("applying external forces to cells failed");
	cudaDeviceSynchronize();

	// Apply viscosity

	// Calculate pressure

	// Apply pressure

	// Extrapolate fluid velocities into surrounding cells

	// Set the velocities of surrounding cells

	// Move the marker particles through the velocity field
	moveMarkerParticlesThroughField<<<BLOCKS_PARTICLES, blockSize>>>(NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH, TIME_STEP);
	checkCUDAError("moving marker particles through velocity field failed");
	cudaDeviceSynchronize();
}