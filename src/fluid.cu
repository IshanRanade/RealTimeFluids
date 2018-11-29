#include "fluid.h"
//#include "hierarchy.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cusolverSp.h>
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

__device__ int getCellCompressedIndex(int x, int y, int z) {
	return z * GRID_X * GRID_Y + y * GRID_X + x;
}

__device__ glm::vec3 getCellUncompressedCoordinates(int index) {
    const int z = index / (GRID_X * GRID_Y);
	index -= (z * GRID_X * GRID_Y);
    const int y = index / GRID_X;
    const int x = index % GRID_X;

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

__global__ void raycastPBO(int numParticles, uchar4 *pbo, MarkerParticle *particles, glm::vec3 camPos, Camera camera) {
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
            glm::vec3 color = glm::vec3(50.f, 50.f, 255.f);
			float depth = glm::clamp(glm::distance(rayPos, camPos) / 10.0f, 0.0f, 1.0f);
            glm::vec3 lightPos = glm::vec3(2, 1, 0);
            float specularIntensity = 10.0f;

            glm::vec3 refl = glm::normalize(glm::normalize(camPos - rayPos) + glm::normalize(lightPos));
            float specularTerm = glm::pow(glm::max(glm::dot(refl, normal), 0.0f), specularIntensity);

            color = color * (depth + specularTerm);
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
	raycastPBO<<<blocksPerGrid2d, BLOCK_SIZE2d >>>(NUM_MARKER_PARTICLES, pbo, dev_markerParticles, camPos, camera);
	checkCUDAError("raymarch to form PBO failed");
	cudaDeviceSynchronize();	
}

__global__ void initializeGridCells(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		int *x;
		int *y;
		int *z;
		glm::vec3 coords = getCellUncompressedCoordinates(index);

		GridCell &cell = cells[index];
		cell.worldPosition = glm::vec3(coords.x * CELL_WIDTH, coords.y * CELL_WIDTH, coords.z * CELL_WIDTH);
	}
}

__global__ void setAllGridCellsToAir(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.cellType = AIR;
	}
}

__global__ void setGridCellsWithMarkerParticleToFluid(int n, GridCell *cells, MarkerParticle *particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		int compressedCellIndex = getCellCompressedIndex((int)particle.worldPosition.x / CELL_WIDTH, (int)particle.worldPosition.x / CELL_WIDTH, (int)particle.worldPosition.x / CELL_WIDTH);
		
		GridCell &cell = cells[compressedCellIndex];
		cell.cellType = FLUID;
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

__global__ void backwardsParticleTrace(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		// For now just use simple Euler
		const glm::vec3 cellPosition = (getCellUncompressedCoordinates(index) * CELL_WIDTH) + glm::vec3(CELL_WIDTH / 2.0, CELL_WIDTH / 2.0, CELL_WIDTH / 2.0);
        const glm::vec3 oldPosition = cellPosition - TIME_STEP * cell.velocity;

		int prevCellIndex = getCellCompressedIndex((int)oldPosition.x, (int)oldPosition.y, (int)oldPosition.z);
		if (prevCellIndex < 0 || prevCellIndex >= GRID_X * GRID_Y * GRID_Z) {
			return;
		}

		GridCell &otherCell = cells[prevCellIndex];
		cell.tempVelocity = otherCell.velocity;
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
		int cellIndex = getCellCompressedIndex(particle.worldPosition.x, particle.worldPosition.y, particle.worldPosition.z);
		GridCell &cell = cells[cellIndex];
		particle.worldPosition += TIME_STEP * cell.velocity;

		particle.worldPosition.x = glm::clamp(particle.worldPosition.x, 0.0f, GRID_X * CELL_WIDTH - 0.01f);
		particle.worldPosition.y = glm::clamp(particle.worldPosition.y, 0.0f, GRID_Y * CELL_WIDTH - 0.01f);
		particle.worldPosition.z = glm::clamp(particle.worldPosition.z, 0.0f, GRID_Z * CELL_WIDTH - 0.01f);
	}
}

__global__ void applyViscosity(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		glm::vec3 cellCoords = getCellUncompressedCoordinates(index);

		int cellTopIndex    = getCellCompressedIndex(cellCoords.x, cellCoords.y + 1, cellCoords.z);
		int cellBottomIndex = getCellCompressedIndex(cellCoords.x, cellCoords.y - 1, cellCoords.z);
		int cellLeftIndex   = getCellCompressedIndex(cellCoords.x - 1, cellCoords.y, cellCoords.z);
		int cellRightIndex  = getCellCompressedIndex(cellCoords.x + 1, cellCoords.y, cellCoords.z);
		int cellFrontIndex  = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z + 1);
		int cellBackIndex   = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z - 1);

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

__global__ void setupPressureCalc(int numCells, float* csrValA, int* csrRowPtrA, int* csrColIndA, float* vecB, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index > 10 && index < 12) {
		glm::vec3 gridPos = getCellUncompressedCoordinates(index);

		// Starting index of current row
		csrRowPtrA[index] = index * 27;

		int nonSolid = -26;
		float airCells = 0.0f;
		for (int i = 0; i < 26; ++i) {
			int x = i % 3 - 1;
			int y = (i / 3) % 3 - 1;
			int z = i / 9 - 1;
			int adjacent = index + x + (y * GRID_X) + (z * GRID_X * GRID_Y);
			if (gridPos.x + x < 0 || gridPos.x + x >= GRID_X || gridPos.y + y < 0 || gridPos.y + y >= GRID_Y || gridPos.x + x < 0 || gridPos.z + z >= GRID_Z) {
				csrColIndA[index * 27 + i] = 0;
				csrRowPtrA[index * 27 + i] = 0;
				++nonSolid;
				continue;
			}
			GridCell cell = cells[adjacent];

			// Set index of adjacent cell
			csrColIndA[index * 27 + i] = adjacent;

			// Set value of matrix element
			csrRowPtrA[index * 27 + i] = cell.cellType == FLUID ? 1.0f : 0.0f;
			airCells += cell.cellType == AIR ? 1.0f : 0.0f;
		}
		// Set matrix value for current grid cell
		csrColIndA[index * 27 + 26] = nonSolid;
		csrRowPtrA[index * 27 + 26] = index;

		// Set value of b vector for pressure linear solver
		float divU = 0.0f;
		if (gridPos.x - 1 > 0) {
			divU += cells[index - 1].velocity.x - cells[index].velocity.x;
		}
		if (gridPos.y - 1 > 0) {
			divU += cells[index - GRID_X].velocity.x - cells[index].velocity.x;
		}
		if (gridPos.z - 1 > 0) {
			divU += cells[index - GRID_X * GRID_Y].velocity.x - cells[index].velocity.x;
		}
		vecB[index] = (WIDTH_DIV_TIME) * divU - airCells;
	}
}

__global__ void copyPressureToCells(int numCells, float* vecX, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numCells) {
		//printf("%d: %f\n", index, vecX[index]);
		cells[index].pressure = vecX[index];
	}
}

void initHierarchicalPressureGrids() {
    // Calculate number of grid levels`
    GRID_LEVELS = std::floor(log2(std::min(std::min(GRID_X, GRID_Y), GRID_Z)));

    // Allocate space for primary grid cells
    cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));

    // Create grid array and primary grid
    grids = new Grid[GRID_LEVELS];
    grids[0].setGrid(0, GRID_X, GRID_Y, GRID_Z);
    grids[0].dev_cells = dev_gridCells;

    for(int d = 1; d < GRID_LEVELS; ++d) {
        // Create and allocate space for sub grid cells
        grids[d].setGrid(d, grids[d - 1].gridX / 2, grids[d - 1].gridY / 2, grids[d - 1].gridZ / 2);
        cudaMalloc(&grids[d].dev_cells, grids[d].numCells * sizeof(GridCell));
    }
}

void initSim() {
    // Init hierarchical pressure grids
    initHierarchicalPressureGrids();

	// Allocate space for all of the marker particles
	cudaMalloc(&dev_markerParticles, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));

	// Allocate space for sparse linear solver of pressures
	/*nnz = NUM_CELLS * 27;
	cudaMalloc(&csrValA, nnz * sizeof(float));
	cudaMalloc(&csrRowPtrA, NUM_CELLS * sizeof(int));
	cudaMalloc(&csrColIndA, nnz * sizeof(int));
	cudaMalloc(&vecX, NUM_CELLS * sizeof(float));
	cudaMalloc(&vecB, NUM_CELLS * sizeof(float));*/

	// Create random world positions for all of the particles
	generateRandomWorldPositionsForParticles<<<BLOCKS_PARTICLES, BLOCK_SIZE>>>(NUM_MARKER_PARTICLES, dev_markerParticles);
	checkCUDAError("generating initial world positions for marker particles failed");
	cudaDeviceSynchronize();

	// Initialize the grid cells
	initializeGridCells<<<BLOCKS_CELLS, BLOCK_SIZE>>>(NUM_CELLS, dev_gridCells);
	checkCUDAError("initializing the grid cells failed");
	cudaDeviceSynchronize();
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

	// Calculate pressure
	/*setupPressureCalc << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, csrValA, csrRowPtrA, csrColIndA, vecB, dev_gridCells);
	checkCUDAError("setup pressure calc failed");
	cudaDeviceSynchronize();

	cusolverSpHandle_t cusolver_handle;
	cusolverStatus_t cusolver_status;
	cusolver_status = cusolverSpCreate(&cusolver_handle);
	//std::cout << "status create cusolver handle: " << cusolver_status << std::endl;
	int singularity = 0;
	cusparseMatDescr_t descrA;
	cusparseCreateMatDescr(&descrA);
	cusolver_status = cusolverSpScsrlsvqr(cusolver_handle, NUM_CELLS, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, vecB, 1e-5, 0, vecX, &singularity);

	copyPressureToCells << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, vecX, dev_gridCells);
	checkCUDAError("copy pressure to cells failed");
	cudaDeviceSynchronize();*/

	// Apply pressure

	// Extrapolate fluid velocities into surrounding cells

	// Set the velocities of surrounding cells

	// Move the marker particles through the velocity field
	moveMarkerParticlesThroughField<<<BLOCKS_PARTICLES, BLOCK_SIZE>>>(NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles);
	checkCUDAError("moving marker particles through velocity field failed");
	cudaDeviceSynchronize();
}
