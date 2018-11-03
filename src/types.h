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