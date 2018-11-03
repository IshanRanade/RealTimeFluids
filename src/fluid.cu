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

__device__ void getCellUncompressedIndex(int index, int *x, int *y, int *z, int GRID_X, int GRID_Y, int GRID_Z) {
	*z = index / (GRID_X * GRID_Y);
	index -= (*z * GRID_X * GRID_Y);
	*y = index / GRID_X;
	*x = index % GRID_X;
}

__global__ void fillGridCells(int n, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.worldPosition = glm::vec3(0, 0, 0);
	}
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

__global__ void setAllGridCellsToNoneCellType(int n, GridCell *cells) {
	
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

void initSim() {
	// Allocate space for all of the grid cells
	cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));
	cudaMemset(dev_gridCells, 0, NUM_CELLS * sizeof(GridCell));

	// Allocate space for all of the marker particles
	cudaMalloc(&dev_markerParticles, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));
	cudaMemset(dev_markerParticles, 0, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));

	// Create random world positions for all of the particles
	int particlesBlocks = (NUM_MARKER_PARTICLES + blockSize - 1) / blockSize;
	generateRandomWorldPositionsForParticles<<<particlesBlocks, blockSize>>>(NUM_MARKER_PARTICLES, dev_markerParticles, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH);
	checkCUDAError("generating initial world positions for marker particles failed");
	cudaDeviceSynchronize();

	//int cellBlocks = (NUM_CELLS + blockSize - 1) / blockSize;
	//fillGridCells<<<cellBlocks, blockSize>>>(NUM_CELLS, dev_gridCells);
	//checkCUDAError("initializing grid cells failed");
	//cudaDeviceSynchronize();
}