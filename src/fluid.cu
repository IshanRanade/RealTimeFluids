#include "fluid.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>


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

__global__ void fillGridCells(int n, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.worldPosition = glm::vec3(0, 0, 0);
	}
}

__global__ void fillVBOData(int n, void *vbo, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//float *vboFloat = (float*)vbo;

	if (index < 4) {
		((float*)vbo)[index] = 0;
		//vboFloat[n * 2 + 1] = 0;
		//vboFloat[n * 2] = 0;
		//vboFloat[n * 2 + 1] = 0;
		//vboFloat[n * 2 + 2] = 0;
	}
}

void fillVBOs(void *vbo) {
	int blocks = (NUM_CELLS + blockSize - 1) / blockSize;
	fillVBOData<<<blocks, blockSize>>>(NUM_CELLS, vbo, dev_gridCells);
	checkCUDAError("error");
	cudaDeviceSynchronize();
}

void initSim() {
	cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));
	cudaMemset(dev_gridCells, 0, NUM_CELLS * sizeof(GridCell));

	int blocks = (NUM_CELLS + blockSize - 1) / blockSize;
	fillGridCells<<<blocks, blockSize>>>(NUM_CELLS, dev_gridCells);
	checkCUDAError("initializing grid cells failed");
	cudaDeviceSynchronize();
}