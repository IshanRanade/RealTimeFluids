// types.h
#pragma once

#include "glm/glm.hpp"

#define CELL_SIZE 1
#define BLOCK_SIZE 256

#define GRID_X 64
#define GRID_Y 64
#define GRID_Z 32
#define NUM_CELLS GRID_X * GRID_Y * GRID_Z

struct GridCell {
    float pressure;
    glm::vec3 velocityX;
    glm::vec3 velocityY;
    glm::vec3 velocityZ;
}
