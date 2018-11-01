// types.h
#pragma once

#include "glm/glm.hpp"

#define CELL_SIZE 1
#define GRID_X 128
#define GRID_Y 128
#define GRID_Z 32
#define GRID_SIZE GRID_X * GRID_Y * GRID_Z

struct GridCell {
    float pressure;
    glm::vec4 velocity_x;
    glm::vec4 velocity_y;
    glm::vec4 velocity_z;
}

/*
// fluid.h
#pragma once

#include <types.h>

void initCuda();
*/

/*
// fluid.cpp
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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
*/

/*
// main.cpp
#pragma once

#include <types.h>
#include <fluid.h>

void runCuda() {
    // Register buffers using OpenGL
    GLuint vertexArray;
    glGenBuffers( 1, &vertexArray );
    glBindBuffer( GL_ARRAY_BUFFER, vertexArray );
    glBufferData( GL_ARRAY_BUFFER, GRID_SIZE * 16, NULL, GL_DYNAMIC_COPY );
    cudaGLRegisterBufferObject( vertexArray );

    void * vertexPointer;
    cudaGLMapBufferObject(&vertexPointer, vertexBuffer);
    
    // Update and write grid information to VBO
    kernUpdateGrid<<<gridSz, blockSz>>>();
    kernGridToVBO<<<gridSz, blockSz>>>(vertexPointer);
    
    cudaGLUnmapbufferObject(vertexBuffer);
    glBindBuffer( GL_ARRAY_BUFFER, vertexBuffer );
    
    // Enable Vertex and Color arrays
    glEnableClientState( GL_VERTEX_ARRAY );
    glEnableClientState( GL_COLOR_ARRAY );
    glVertexPointer(3,GL_FLOAT,16,0);
    glColorPointer(4,GL_UNSIGNED_BYTE,16,12);

    glDrawArrays(GL_POINTS, 0, GRID_SIZE);
    SwapBuffer();

    frame++;
}
*/
