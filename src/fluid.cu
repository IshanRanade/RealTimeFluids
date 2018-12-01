#include "fluid.h"
//#include "hierarchy.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/random.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>


#define ERRORCHECK 1
#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn2(msg, FILENAME, __LINE__)
void checkCUDAErrorFn2(const char *msg, const char *file, int line) {
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

__device__ int getCellCompressedIndex(int x, int y, int z, int gridX, int gridY) {
    return z * gridX * gridY + y * gridX + x;
}

__device__ glm::vec3 getCellUncompressedCoordinates(int index, int gridX, int gridY) {
    const int z = index / (gridX * gridY);
    index -= (z * gridX * gridY);
    const int y = index / gridX;
    const int x = index % gridX;

    return glm::vec3(x, y, z);
}

__global__ void fillVBOData(int n, void *vbo, MarkerParticle *particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

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

// Quadratic solver from scratchapixel.com
__device__ bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1) {
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) x0 = x1 = -0.5 * b / a;
    else {
        float q = (b > 0) ?
            -0.5 * (b + sqrt(discr)) :
            -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) {
        float temp = x0;
        x0 = x1;
        x1 = temp;
    }

    return true;
}

// Ray-Sphere Intersection from scratchapixel.com
__device__ float raySphereIntersect(glm::vec3 rayPos, glm::vec3 rayDir, glm::vec3 center, float radius2) {
    float t0, t1; // solutions for t if the ray intersects
    
    // analytic solution
    glm::vec3 L = rayPos - center;
    float a = glm::dot(rayDir, rayDir);
    float b = 2 * glm::dot(rayDir, L);
    float c = glm::dot(L, L) - radius2;
    if (!solveQuadratic(a, b, c, t0, t1)) return -1.0f;

    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }

    if (t0 < 0) {
        t0 = t1; // if t0 is negative, let's use t1 instead 
        if (t0 < 0) return -1.0f; // both t0 and t1 are negative 
    }

    return t0;
}

__device__ float smin(float a, float b, float k) {
    const float h = glm::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return glm::mix(b, a, h) - k * h * (1.0f - h);
}

__device__ glm::vec4 smin(glm::vec3 vecA, glm::vec3 vecB, float a, float b, float k) {
    const float h = glm::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return glm::vec4(glm::mix(vecA, vecB, h), glm::mix(b, a, h) - k * h * (1.0f - h));
}

__device__ bool inBounds(float value, float bounds) {
	return (value >= -bounds) && (value <= bounds);
}

__global__ void raycastPBO(int numParticles, uchar4* pbo, MarkerParticle* particles, glm::vec3 camPos, Camera camera, GridCell* cells) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < camera.resolution.x && idy < camera.resolution.y) {
        // Setup variables
        bool intersected = false;
        glm::vec3 rayPos = camPos;
        float distance = 1000.0f;
        const glm::vec3 view = camera.view;
        const glm::vec3 up = camera.up;
        const glm::vec3 right = camera.right;
        glm::vec3 normal = glm::vec3(0, 1, 0);

        const float yscaled = glm::tan(camera.fov.y * (3.1415927f / 180.0f));
        const float xscaled = (yscaled *  camera.resolution.x) / camera.resolution.y;
		glm::vec2 pixelLength = glm::vec2(2 * xscaled / camera.resolution.x, 2 * yscaled / camera.resolution.y);

		glm::vec3 rayDir = glm::normalize(view
			- right * pixelLength.x * ((float)idx - camera.resolution.x * 0.5f)
			- up * pixelLength.y * ((float)idy - camera.resolution.y * 0.5f)
		);

#if RAY_CAST
        // Ray-Sphere Intersection with all particles
        for (int i = 0; i < numParticles; ++i) {
            MarkerParticle& particle = particles[i];

            float t = raySphereIntersect(rayPos, rayDir, particle.worldPosition, PARTICLE_RADIUS);
            if (t > 0 && t < distance) {
                intersected = true;
                distance = t;
                normal = glm::normalize(rayPos + (rayDir * t) - particle.worldPosition);
                
            }
        }
        rayPos += rayDir * distance;

#elif SPHERE_MARCH
        int iterations = 0;
        const int maxIterations = 16;
        const float radius = 1.25f;

        // Sphere march for smoothed min marker particle
		while(distance > radius && iterations < maxIterations) {
			for(int i = 0; i < numParticles; ++i) {
				MarkerParticle& particle = particles[i];
                //distance = glm::min(distance, glm::distance(rayPos, particle.worldPosition));
				distance = smin(distance, glm::distance(rayPos, particle.worldPosition), PARTICLE_RADIUS);
				if(distance < radius) {
                    normal = glm::normalize(rayPos - particle.worldPosition);
					break;
				}
			}
			rayPos += rayDir * distance;
			++iterations;
		}
        intersected = distance < radius;
#endif

        const int index = idx + idy * camera.resolution.x;

		// Set the color
		if(intersected) {
            // Ray hit a marker particle
            //glm::vec3 color = glm::vec3(50.f, 50.f, 255.f);
            glm::vec3 color =  glm::abs(cells[getCellCompressedIndex(rayPos.x, rayPos.y, rayPos.z, GRID_X, GRID_Y)].velocity) * 40.0f;
            //color = cells[getCellCompressedIndex(rayPos.x, rayPos.y, rayPos.z, GRID_X, GRID_Y)].cellType == FLUID ? color : glm::vec3(0);
			const float depth = glm::clamp(glm::distance(rayPos, camPos) / 10.0f, 0.0f, 1.0f); // camera depth not fluid depth
            const glm::vec3 lightPos = glm::vec3(2, 1, 0);
            const float specularIntensity = 10.0f;

            const glm::vec3 refl = glm::normalize(glm::normalize(camPos - rayPos) + glm::normalize(lightPos));
            const float specularTerm = glm::pow(glm::max(glm::dot(refl, normal), 0.0f), specularIntensity);

            //color = color * (depth + specularTerm);
			pbo[index].x = glm::min(color.x, 255.0f);
			pbo[index].y = glm::min(color.y, 255.0f);
			pbo[index].z = glm::min(color.z, 255.0f);
			pbo[index].w = 0;
		}
        else {
            // Clear background
            pbo[index].x = 205.0f;
            pbo[index].y = 205.0f;
            pbo[index].z = 240.0f;
            pbo[index].w = 0;
        }
	}
}

void raycastPBO(uchar4* pbo, glm::vec3 camPos, Camera camera) {
	/*
	// Initialize 3D quad tree hierarchy
	TreeNode* root = buildTree(std::vector<MarkerParticle> particles, int currentDepth, glm::vec3 boundMin, glm::vec3 boundMax);
	int numNodes = tree::treeSize(root);
	std::vector<LinearNode> flatTree;
	for (int i = 0; i < n; i++) {
		flatTree.push_back(LinearNode());
	}
	int offset = 0;
	flattenTree(root, sortedGeoms, flatTree, &offset);
	deleteTree(root);
	*/

	const dim3 BLOCK_SIZE2d(8, 8);
	const dim3 blocksPerGrid2d(
		(camera.resolution.x + BLOCK_SIZE2d.x - 1) / BLOCK_SIZE2d.x,
		(camera.resolution.y + BLOCK_SIZE2d.y - 1) / BLOCK_SIZE2d.y);
	raycastPBO<<<blocksPerGrid2d, BLOCK_SIZE2d >>>(NUM_MARKER_PARTICLES, pbo, dev_markerParticles, camPos, camera, dev_gridCells);
	checkCUDAError("raymarch to form PBO failed");
	cudaDeviceSynchronize();	
}

__global__ void setAllGridCellsToAir(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.cellType = AIR;
	}
}

__global__ void setGridCellsWithMarkerParticleToFluid(int n, GridCell* cells, MarkerParticle* particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		const int compressedCellIndex = getCellCompressedIndex(particle.worldPosition.x / CELL_WIDTH, particle.worldPosition.y / CELL_WIDTH, particle.worldPosition.z / CELL_WIDTH, GRID_X, GRID_Y);
		
        cells[compressedCellIndex].cellType = FLUID;

        const glm::vec3 cellPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);
        for (int i = 0; i < 5; ++i) {
            const int x = i == 0 ? -1 : i == 1 ? 1 : 0;
            const int y = i == 2 ? -1 : 0;
            const int z = i == 3 ? -1 : i == 4 ? 1 : 0;
            const int adjacent = getCellCompressedIndex(cellPos.x + x, cellPos.y + y, cellPos.z + z, GRID_X, GRID_Y);

            // Special case for bounds adjacent
            if (cellPos.x + x >= 0 && cellPos.x + x < GRID_X && cellPos.y + y >= 0 && cellPos.y + y < GRID_Y && cellPos.z + z >= 0 && cellPos.z + z < GRID_Z) {
                cells[adjacent].cellType = FLUID;
            }
        }
	}
}

void fillVBOsWithMarkerParticles(void *vbo) {
    const int blocks = (NUM_MARKER_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
	fillVBOData<<<blocks, BLOCK_SIZE>>>(NUM_MARKER_PARTICLES, vbo, dev_markerParticles);
	checkCUDAError("filling VBOs with marker particle data failed");
	cudaDeviceSynchronize();
}

__global__ void generateRandomWorldPositionsForParticles(int n, MarkerParticle *particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		// TODO: Fix these random generators
		thrust::default_random_engine rngX = thrust::default_random_engine(index | (index << 22));
		thrust::default_random_engine rngY;
		thrust::default_random_engine rngZ = thrust::default_random_engine(index ^ (index * 13));
		thrust::uniform_real_distribution<float> u01(0, 1);

		MarkerParticle &particle = particles[index];
		particle.worldPosition.x = u01(rngX) * GRID_X * CELL_WIDTH;
		particle.worldPosition.y = 0.5f * u01(rngX) * GRID_Y * CELL_WIDTH + glm::sin(particle.worldPosition.x / 12.0f);
		particle.worldPosition.z = u01(rngX) * GRID_Z * CELL_WIDTH;

		particle.color = glm::vec3(0.2, 0.2, 1);
	}
}

__global__ void backwardsParticleTrace(int n, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		// For now just use simple Euler
		const glm::vec3 cellPosition = (getCellUncompressedCoordinates(index, GRID_X, GRID_Y) * CELL_WIDTH) + glm::vec3(CELL_WIDTH / 2.0, CELL_WIDTH / 2.0, CELL_WIDTH / 2.0);
        const glm::vec3 oldPosition = cellPosition + TIME_STEP * cell.velocity;

		const int prevCellIndex = getCellCompressedIndex(oldPosition.x, oldPosition.y, oldPosition.z, GRID_X, GRID_Y);
        if (prevCellIndex < 0 || prevCellIndex >= NUM_CELLS)
            return;

		GridCell &otherCell = cells[prevCellIndex];
		cell.tempVelocity = cell.velocity + TIME_STEP * otherCell.velocity;

	}
}

__global__ void applyExternalForcesToGridCells(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		// Apply gravity
		cell.tempVelocity = cell.velocity + glm::vec3(0, -9.8 * TIME_STEP, 0);
	}
}
__global__ void moveMarkerParticlesThroughField(int n, GridCell *cells, MarkerParticle *particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		// Find the cell that this particle is in
		int cellIndex = getCellCompressedIndex(particle.worldPosition.x, particle.worldPosition.y, particle.worldPosition.z, GRID_X, GRID_Y);
		
	    GridCell &cell = cells[cellIndex];

        // clamp velocity max
	    //if (glm::length(cell.velocity) > 5.0f)
            //cell.velocity *= 5.0f / glm::length(cell.velocity);

		particle.worldPosition += TIME_STEP * cell.velocity;
        float tempPos = particle.worldPosition.x;
		particle.worldPosition.x = glm::clamp(particle.worldPosition.x, 0.0f, GRID_X * CELL_WIDTH - 0.01f);
        if (glm::abs(tempPos - particle.worldPosition.x) > 0.01f)
            cell.velocity.x = 0.0f;

        tempPos = particle.worldPosition.y;
		particle.worldPosition.y = glm::clamp(particle.worldPosition.y, 0.0f, GRID_Y * CELL_WIDTH - 0.01f);
        if (glm::abs(tempPos - particle.worldPosition.y) > 0.01f)
            cell.velocity.y = 0.0f;

        tempPos = particle.worldPosition.z;
		particle.worldPosition.z = glm::clamp(particle.worldPosition.z, 0.0f, GRID_Z * CELL_WIDTH - 0.01f);
        if (glm::abs(tempPos - particle.worldPosition.z) > 0.01f)
            cell.velocity.z = 0.0f;
	}
}

__global__ void applyViscosity(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		glm::vec3 cellCoords = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);

		int cellTopIndex    = getCellCompressedIndex(cellCoords.x, cellCoords.y + 1, cellCoords.z, GRID_X, GRID_Y);
		int cellBottomIndex = getCellCompressedIndex(cellCoords.x, cellCoords.y - 1, cellCoords.z, GRID_X, GRID_Y);
		int cellLeftIndex   = getCellCompressedIndex(cellCoords.x - 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y);
		int cellRightIndex  = getCellCompressedIndex(cellCoords.x + 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y);
		int cellFrontIndex  = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z + 1, GRID_X, GRID_Y);
		int cellBackIndex   = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z - 1, GRID_X, GRID_Y);

		float indices[6] = { cellTopIndex, cellBottomIndex, cellLeftIndex, cellRightIndex, cellFrontIndex, cellBackIndex };

		float laplacianX = 0.0;
		float laplacianY = 0.0;
		float laplacianZ = 0.0;
		for (int i = 0; i < 6; ++i) {
			int currCellIndex = indices[i];
			if (currCellIndex >= 0 && currCellIndex < GRID_X * GRID_Y * GRID_Z) {
				laplacianX += cells[currCellIndex].velocity.x;
				laplacianY += cells[currCellIndex].velocity.y;
				laplacianZ += cells[currCellIndex].velocity.z;
			}
		}

		laplacianX -= 6 * cell.velocity.x;
		laplacianY -= 6 * cell.velocity.y;
		laplacianZ -= 6 * cell.velocity.z;

		cell.tempVelocity = cell.velocity + TIME_STEP * VISCOSITY * glm::vec3(laplacianX, laplacianY, laplacianZ);
	}
}

__global__ void swapCellVelocities(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.velocity = cell.tempVelocity;
	}
}

__global__ void setupPressureCalc(Grid grid, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= grid.numCells)
        return;

	const glm::vec3 cellPos = getCellUncompressedCoordinates(index, grid.sizeX, grid.sizeY);

	int nonSolid = -6;
	float airCells = 0.0f;
	for (int i = 0; i < 6; ++i) {
        const int x = i == 0 ? -1 : i == 1 ? 1 : 0;
        const int y = i == 2 ? -1 : i == 3 ? 1 : 0;
        const int z = i == 4 ? -1 : i == 5 ? 1 : 0;
        const int adjacent = getCellCompressedIndex(cellPos.x + x, cellPos.y + y, cellPos.z + z, grid.sizeX, grid.sizeY);

        // Special case for bounds adjacent
		if (cellPos.x + x < 0 || cellPos.x + x >= grid.sizeX || cellPos.y + y < 0 || cellPos.y + y >= grid.sizeY || cellPos.z + z < 0 || cellPos.z + z >= grid.sizeZ) {
            grid.dev_colIndA[index * 6 + i] = -1;
            ++nonSolid;
			continue;
		}

        GridCell& cell = cells[adjacent];

        if (cell.cellType == AIR)
            airCells += 1.0f;

        // Set index of adjacent cell
        grid.dev_colIndA[index * 6 + i] = adjacent;

		// Set value of matrix element
        grid.dev_valA[index * 7 + i + 1] = 1.0f;
	}
	// Set matrix value for current grid cell
    grid.dev_valA[index * 7] = nonSolid;

	// Set value of b vector for pressure linear solver
	float divU = 0.0f;
    GridCell& cell = cells[index];
	if (cellPos.x + 1 < grid.sizeX) {
        GridCell& adjacent = cells[index + 1];
		divU += adjacent.velocity.x - cell.velocity.x;
	}
	if (cellPos.y + 1 < grid.sizeY) {
        GridCell& adjacent = cells[index + grid.sizeX];
		divU += adjacent.velocity.y - cell.velocity.y;
	}
	if (cellPos.z + 1 < grid.sizeZ) {
        GridCell& adjacent = cells[index + grid.sizeX * grid.sizeY];
        divU += adjacent.velocity.z - cell.velocity.z;
	}
	
    grid.dev_B[index] = (WIDTH_DIV_TIME) * divU * (cell.cellType == FLUID ? FLUID_DENSITY : AIR_DENSITY) - airCells;
}

__global__ void gaussSeidelPressure(Grid grid) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= grid.numCells)
        return;

    float numerator = grid.dev_B[index];
    for (int j = 0; j < 6; ++j) {
        if(grid.dev_colIndA[index * 6 + j] != -1)
            numerator -= grid.dev_valA[index * 7 + j + 1] * grid.dev_X[grid.dev_colIndA[index * 6 + j]];
    }
    grid.dev_X[index] = numerator / grid.dev_valA[index * 7];
}

void gaussSeidelPressureCPU(int numCells, float* valA, int* colIndA, float* vecX, float* vecB) {
    for (int i = 0; i < numCells; ++i) {
        float numerator = vecB[i];
        for (int j = 0; j < 6; ++j) {
            if (colIndA[i * 6 + j] != -1)
                numerator -= valA[i * 7 + j + 1] * vecX[colIndA[i * 6 + j]];
        }
        vecX[i] = numerator / valA[i * 7];
    }
}

/*__global__ void restrictKernel(int bottomLevelSize, Grid *topLevel, Grid *bottomLevel)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

}

__global__ void restrictVector(Grid smallGrid, Grid largeGrid) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= smallGrid.numCells)
        return;

    float average = 0.0f;
    int count = 0;

    for (int i = 0; i < 6; ++i) {
        //const int adjacent = getCellCompressedIndex(cellPos.x + x, cellPos.y + y, cellPos.z + z, grid.sizeX, grid.sizeY);

        // Special case for bounds adjacent
        //if (cellPos.x + x < 0 || cellPos.x + x >= grid.numCells || cellPos.y + y < 0 || cellPos.y + y >= grid.numCells) {
            // out of bounds
            //--count;
            //continue;
        //}

        //average += 
    }
}

void V_Cycle(int level) {
    if(level == MAX_GRID_LEVEL)
    {
        // Solve the linear system directly
        // For now just do some gauss seidel iterations
        for(int i = 0; i < 5; i++)
        {
            int numBlocks = ((grids[level].numCells + BLOCK_SIZE - 1) / BLOCK_SIZE);
            gaussSeidelPressure<<<numBlocks, BLOCK_SIZE>>>(grids[level]);
        }
    } else
    {
        //for(int i = 0; i < )
    }
}*/

// change vecX to pressure array, take out pressure from cells, delete this kernel
__global__ void copyPressureToCells(int numCells, float* pressure, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numCells) {
        //if (cells[index].cellType == FLUID)
            //printf("%d: %f\n", index, pressure[index]);
        cells[index].pressure = cells[index].cellType == AIR ? 1.0f : pressure[index];
	}
}

__global__ void applyPressure(int numCells, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numCells) {
        GridCell& cell = cells[index];
        if (cell.cellType != FLUID)
            return;

        const glm::vec3 gridPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);
        glm::vec3 deltaPressure = glm::vec3(0.0f, 0.0f, 0.0f);
        if (gridPos.x - 1 >= 0) {
            const int nextPos = getCellCompressedIndex(gridPos.x - 1, gridPos.y, gridPos.z, GRID_X, GRID_Y);
            deltaPressure.x = cell.pressure - cells[nextPos].pressure;
        }

        if (gridPos.y - 1 >= 0) {
            const int nextPos = getCellCompressedIndex(gridPos.x, gridPos.y - 1, gridPos.z, GRID_X, GRID_Y);
            deltaPressure.y = cell.pressure - cells[nextPos].pressure;
        }

        if (gridPos.z - 1 >= 0) {
            const int nextPos = getCellCompressedIndex(gridPos.x, gridPos.y, gridPos.z - 1, GRID_X, GRID_Y);
            deltaPressure.z = cell.pressure - cells[nextPos].pressure;
        }

        cell.tempVelocity = cell.velocity - deltaPressure * TIME_STEP / ((cell.cellType == FLUID ? FLUID_DENSITY : AIR_DENSITY) * CELL_WIDTH);
    }
}

__global__ void extrapolateFluidVelocities(int numCells, GridCell* cells)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < numCells)
    {
        GridCell &cell = cells[index];

        if (cell.cellType == AIR) {
            glm::vec3 pos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);

            glm::vec3 averageVelocity = glm::vec3(0.0);
            int count = 0;
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        if (pos.x + x >= 0 && pos.x < GRID_X &&
                            pos.y + y >= 0 && pos.y < GRID_Y &&
                            pos.z + z >= 0 && pos.z < GRID_Z)
                        {
                            int surroundingVelocityIndex = getCellCompressedIndex(pos.x + x, pos.y + y, pos.z + z, GRID_X, GRID_Y);
                            averageVelocity += cells[surroundingVelocityIndex].velocity;
                            ++count;
                        }
                    }
                }
            }

            cell.tempVelocity = averageVelocity / float(count);
        }
    }
}

void initHierarchicalPressureGrids() {
    // Calculate number of grid levels
    MAX_GRID_LEVEL = 3;//std::floor(log2(std::min(std::min(GRID_X, GRID_Y), GRID_Z)));

    // Create grid array and primary grid
    grids = new Grid[MAX_GRID_LEVEL];
    grids[0].level = 0;
    grids[0].sizeX = GRID_X;
    grids[0].sizeY = GRID_Y;
    grids[0].sizeZ = GRID_Z;
    grids[0].numCells = NUM_CELLS;

    // 7 nonzero values per row
    cudaMalloc(&grids[0].dev_valA, NUM_CELLS * 7 * sizeof(float));

    // 6 column index
    cudaMalloc(&grids[0].dev_colIndA, NUM_CELLS * 6 * sizeof(int));

    cudaMalloc(&grids[0].dev_X, NUM_CELLS * sizeof(float));
    cudaMalloc(&grids[0].dev_B, NUM_CELLS * sizeof(float));
    //cudaMalloc(&grids[0].dev_r, NUM_CELLS * sizeof(float));

    for(int d = 1; d < MAX_GRID_LEVEL; ++d) {
        // Create and allocate space for sub grid cells
        grids[d].level = d;
        grids[d].sizeX = grids[d - 1].sizeX / 2;
        grids[d].sizeY = grids[d - 1].sizeY / 2;
        grids[d].sizeZ = grids[d - 1].sizeZ / 2;
        grids[d].numCells = grids[d].sizeX * grids[d].sizeY * grids[d].sizeZ;
        cudaMalloc(&grids[d].dev_valA, grids[d].numCells * 7 * sizeof(float));
        cudaMalloc(&grids[d].dev_colIndA, grids[d].numCells * 6 * sizeof(int));
        cudaMalloc(&grids[d].dev_X, grids[d].numCells * sizeof(float));
        cudaMalloc(&grids[d].dev_B, grids[d].numCells * sizeof(float));
        //cudaMalloc(&grids[d].dev_r, grids[d].numCells * sizeof(float));
    }
}

void initSim() {
    // Allocate space for grid cells
    cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));
    checkCUDAError("malloc grid cells failed");

    // Init hierarchical pressure grids
    initHierarchicalPressureGrids();
    checkCUDAError("init hierarchical pressure grids failed");

	// Allocate space for all of the marker particles
	cudaMalloc(&dev_markerParticles, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));
    checkCUDAError("malloc marker particles failed");

	// Create random world positions for all of the particles
	generateRandomWorldPositionsForParticles<<<BLOCKS_PARTICLES, BLOCK_SIZE>>>(NUM_MARKER_PARTICLES, dev_markerParticles);
	checkCUDAError("generating initial world positions for marker particles failed");
	cudaDeviceSynchronize();

    valA = (float*)malloc(NUM_CELLS * 7 * sizeof(float));
    colIndA = (int*)malloc(NUM_CELLS * 6 * sizeof(int));
    vecB = (float*)malloc(NUM_CELLS * sizeof(float));
    vecX = (float*)malloc(NUM_CELLS * sizeof(float));
}

void freeSim() {
    cudaFree(dev_markerParticles);
    cudaFree(dev_gridCells);
    for (int d = 0; d < MAX_GRID_LEVEL; ++d) {
        cudaFree(grids[d].dev_B);
        cudaFree(grids[d].dev_X);
        cudaFree(grids[d].dev_colIndA);
        cudaFree(grids[d].dev_valA);
        cudaFree(grids[d].dev_B);
        free(vecB);
        free(vecX);
        free(valA);
        free(colIndA);
    }
}

void iterateSim() {
	// Make all the cells temporarily air cells
	setAllGridCellsToAir<<<BLOCKS_CELLS, BLOCK_SIZE>>>(NUM_CELLS, dev_gridCells);
	checkCUDAError("marking all cells as air cells failed");
	cudaDeviceSynchronize();

	// Mark all cells with a marker particle as a fluid cell
	setGridCellsWithMarkerParticleToFluid<<<BLOCKS_PARTICLES, BLOCK_SIZE>>>(NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles);
	checkCUDAError("marking all cells with a marker particle as fluid cells failed");
	cudaDeviceSynchronize();

	// Apply convection to velocities using a backwards particle trace
	backwardsParticleTrace<<<BLOCKS_CELLS, BLOCK_SIZE>>>(NUM_CELLS, dev_gridCells);
	checkCUDAError("convecting velocities using a backwards particle trace failed");
	cudaDeviceSynchronize();

	// Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
	swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("swapping velocities in cells failed");
	cudaDeviceSynchronize();

	// Apply external forces to grid cell velocities
	applyExternalForcesToGridCells<<<BLOCKS_CELLS, BLOCK_SIZE>>>(NUM_CELLS, dev_gridCells);
	checkCUDAError("applying external forces to cells failed");
	cudaDeviceSynchronize();

	// Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
	swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("swapping velocities in cells failed");
	cudaDeviceSynchronize();

	// Apply viscosity to velocities
	applyViscosity<<<BLOCKS_CELLS, BLOCK_SIZE>>>(NUM_CELLS, dev_gridCells);
	checkCUDAError("applying viscosity failed");
	cudaDeviceSynchronize();

	// Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
	swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("swapping velocities in cells failed");
	cudaDeviceSynchronize();

    // Setup pressure calculation
    setupPressureCalc << <BLOCKS_CELLS, BLOCK_SIZE >> > (grids[0], dev_gridCells);
    checkCUDAError("setup pressure calc failed");
    cudaDeviceSynchronize();

    // Gauss Seidel Pressure Solver
    for (int i = 0; i < GAUSS_ITERATIONS; ++i) {
        memset(vecX, 0.0f, NUM_CELLS * sizeof(float));
        cudaMemcpy(valA, grids[0].dev_valA, NUM_CELLS * 7 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(colIndA, grids[0].dev_colIndA, NUM_CELLS * 6 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(vecB, grids[0].dev_B, NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
        gaussSeidelPressureCPU(NUM_CELLS, valA, colIndA, vecX, vecB);
        cudaMemcpy(grids[0].dev_X, vecX, NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice);

        /*gaussSeidelPressure << <BLOCKS_CELLS, BLOCK_SIZE >> > (grids[0]);
        checkCUDAError("gauss seidel iteration failed");
        cudaDeviceSynchronize();*/
    }

    // Copy pressure to cells
    copyPressureToCells << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, grids[0].dev_X, dev_gridCells);
    checkCUDAError("copy pressure to cells failed");
    cudaDeviceSynchronize();

	// Apply pressure
    applyPressure << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("applying pressure to cell velocities failed");
    cudaDeviceSynchronize();

    // Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
    swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("swapping velocities in cells failed");
    cudaDeviceSynchronize();

	// Extrapolate fluid velocities into surrounding cells
    extrapolateFluidVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("extrapolating velocities failed");
    cudaDeviceSynchronize();

    // Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
    swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("swapping velocities in cells failed");
    cudaDeviceSynchronize();

	// Set the velocities of surrounding cells

	// Move the marker particles through the velocity field
	moveMarkerParticlesThroughField<<<BLOCKS_PARTICLES, BLOCK_SIZE>>>(NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles);
	checkCUDAError("moving marker particles through velocity field failed");
	cudaDeviceSynchronize();
}
