#include "fluid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void fillGridCells(int n, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.worldPosition = glm::vec3(0, 0, 0);
	}
}

void initSim() {
	cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));
	cudaMemset(dev_gridCells, 0, NUM_CELLS * sizeof(GridCell));

	int blocks = (NUM_CELLS + blockSize - 1) / blockSize;
}