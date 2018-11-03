#include "fluid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>


__global__ void fillGridCells(int n, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.worldPosition = glm::vec3(0, 0, 0);
	}
}

__global__ void fillVBOData(int n, void *vbo, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float *vboFloat = (float*)vbo;

	if (index < n) {
		vboFloat[n * 2] = 0;
		vboFloat[n * 2 + 1] = 0;
		//vboFloat[n * 2] = 0;
		//vboFloat[n * 2 + 1] = 0;
		//vboFloat[n * 2 + 2] = 0;
	}
}

void fillVBOs(void *vbo) {
	int blocks = (NUM_CELLS + blockSize - 1) / blockSize;
	fillVBOData<<<blocks, blockSize>>>(NUM_CELLS, vbo, dev_gridCells);
}

void initSim() {
	cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));
	cudaMemset(dev_gridCells, 0, NUM_CELLS * sizeof(GridCell));

	int blocks = (NUM_CELLS + blockSize - 1) / blockSize;
	fillGridCells<<<blocks, blockSize>>>(NUM_CELLS, dev_gridCells);
}