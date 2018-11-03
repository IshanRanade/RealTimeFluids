// fluid.cpp
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <types.h>
#include <fluid.h>

static GridCell* dev_grid = NULL;

void initCuda() {
    cudaFree(dev_grid);
    cudaMalloc(&dev_grid, GRID_SIZE * sizeof(GridCell));
    cudaMemset(dev_grid, 0, GRID_SIZE * sizeof(GridCell));
    
    checkCUDAError("initCuda");
}

__global__
void kernUpdateGrid() {
}

__global__
void kernGridToVBO() {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = (blockIdx.z * blockDim.z) + threadIdx.z;
    int index = x + (y * GRID_X) + (z * GRID_X * GRID_Y);

    if (x < GRID_X && y < GRID_Y && z < GRID_Z) {
        // Each thread writes one vertex position representing the grid cell
        vbo[index].w = 0;
        vbo[index].x = x * CELL_SIZE;
        vbo[index].y = y * CELL_SIZE;
        vbo[index].z = z * CELL_SIZE;
    }
}
