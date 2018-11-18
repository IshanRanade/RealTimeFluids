#include "fluid.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/random.h>


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

__device__ int getCellCompressedIndex(int x, int y, int z, int GRID_X, int GRID_Y, int GRID_Z) {
	return z * GRID_X * GRID_Y + y * GRID_X + x;
}

__device__ glm::vec3 getCellUncompressedCoordinates(int index, int GRID_X, int GRID_Y, int GRID_Z) {
	int z = index / (GRID_X * GRID_Y);
	index -= (z * GRID_X * GRID_Y);
	int y = index / GRID_X;
	int x = index % GRID_X;

	return glm::vec3(x, y, z);
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

__device__ float smin(float a, float b, float k) {
  float h = glm::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
  return glm::mix(b, a, h) - k * h * (1.0f - h);
}

__device__ bool inBounds(float value, float bounds) {
	return (value >= -bounds) && (value <= bounds);
}

__global__ void raymarchPBO(int numParticles, uchar4 *pbo, MarkerParticle *particles, glm::vec3 camPos, Camera camera) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < camera.resolution.x && idy < camera.resolution.y) {

		int iterations = 0;
		const int maxIterations = 10;
		glm::vec3 rayPos = camPos;
		float distance = 9999.0f;
		float epsilon = 0.35f;
		glm::vec3 view = camera.view;
		glm::vec3 up = camera.up;
		glm::vec3 right = camera.right;

		float yscaled = glm::tan(camera.fov.y * (3.1415927f / 180.0f));
		float xscaled = (yscaled *  camera.resolution.x) / camera.resolution.y;
		glm::vec2 pixelLength = glm::vec2(2 * xscaled / camera.resolution.x, 2 * yscaled / camera.resolution.y);

		glm::vec3 rayDir = glm::normalize(view
			- right * pixelLength.x * ((float)idx - camera.resolution.x * 0.5f)
			- up * pixelLength.y * ((float)idy - camera.resolution.y * 0.5f)
		);

		while(distance > epsilon && iterations < maxIterations) {
			for(int i = 0; i < numParticles; ++i) {
				MarkerParticle& particle = particles[i];
				//if(particle.cellType == FLUID) {
				distance = smin(distance, glm::distance(rayPos, particle.worldPosition), 1.0f);
				if(distance < epsilon) {
					break;
				}
				//}
			}
			rayPos += rayDir * distance;
			++iterations;
		}
		int index = idx + idy * camera.resolution.x;

		// Set the color
		if(distance < epsilon) {
			float depth = glm::clamp(glm::distance(rayPos, camPos) / 10.0f, 0.0f, 1.0f);
			pbo[index].x = 50.f * depth;
			pbo[index].y = 50.f * depth;
			pbo[index].z = 255.f * depth;
			pbo[index].w = 0;
		}
		else {
			pbo[index].x = 200.f;
			pbo[index].y = 200.f;
			pbo[index].z = 255.f;
			pbo[index].w = 0;
		}
	}
}

void raymarchPBO(uchar4* pbo, glm::vec3 camPos, Camera camera) {
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(camera.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(camera.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	raymarchPBO<<<blocksPerGrid2d, blockSize2d >>>(NUM_MARKER_PARTICLES, pbo, dev_markerParticles, camPos, camera);
	checkCUDAError("raymarch to form PBO failed");
	cudaDeviceSynchronize();
}

__global__ void initializeGridCells(int n, GridCell *cells, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		int *x;
		int *y;
		int *z;
		glm::vec3 coords = getCellUncompressedCoordinates(index, GRID_X, GRID_Y, GRID_Z);

		GridCell &cell = cells[index];
		cell.worldPosition = glm::vec3(coords.x * CELL_WIDTH, coords.y * CELL_WIDTH, coords.z * CELL_WIDTH);
	}
}

__global__ void setAllGridCellsToAir(int n, GridCell *cells, CellType airType) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.cellType = airType;
	}
}

__global__ void setGridCellsWithMarkerParticleToFluid(int n, GridCell *cells, MarkerParticle *particles, CellType fluidType, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		int compressedCellIndex = getCellCompressedIndex((int)particle.worldPosition.x / CELL_WIDTH, (int)particle.worldPosition.x / CELL_WIDTH, (int)particle.worldPosition.x / CELL_WIDTH, GRID_X, GRID_Y, GRID_Z);
		
		GridCell &cell = cells[compressedCellIndex];
		cell.cellType = fluidType;
	}
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

__global__ void backwardsParticleTrace(int n, GridCell *cells, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH, float TIME_STEP) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		// For now just use simple Euler
		glm::vec3 cellPosition = cell.worldPosition + glm::vec3(CELL_WIDTH / 2.0, CELL_WIDTH / 2.0, CELL_WIDTH / 2.0);
		glm::vec3 oldPosition = cellPosition - TIME_STEP * cell.velocity;

		int prevCellIndex = getCellCompressedIndex((int)oldPosition.x, (int)oldPosition.y, (int)oldPosition.z, GRID_X, GRID_Y, GRID_Z);
		if (prevCellIndex < 0 || prevCellIndex >= GRID_X * GRID_Y * GRID_Z) {
			return;
		}

		GridCell &otherCell = cells[prevCellIndex];
		cell.tempVelocity = otherCell.velocity;
	}
}

__global__ void applyExternalForcesToGridCells(int n, GridCell *cells, float TIME_STEP) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		// Apply gravity
		cell.tempVelocity = cell.velocity + glm::vec3(0, -9.8 * TIME_STEP, 0);
	}
}
__global__ void moveMarkerParticlesThroughField(int n, GridCell *cells, MarkerParticle *particles, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH, float TIME_STEP) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		MarkerParticle &particle = particles[index];

		// Find the cell that this particle is in
		int cellIndex = getCellCompressedIndex(particle.worldPosition.x, particle.worldPosition.y, particle.worldPosition.z, GRID_X, GRID_Y, GRID_Z);
		GridCell &cell = cells[cellIndex];
		particle.worldPosition += TIME_STEP * cell.velocity;

		particle.worldPosition.x = glm::clamp(particle.worldPosition.x, 0.0f, GRID_X * CELL_WIDTH - 0.01f);
		particle.worldPosition.y = glm::clamp(particle.worldPosition.y, 0.0f, GRID_Y * CELL_WIDTH - 0.01f);
		particle.worldPosition.z = glm::clamp(particle.worldPosition.z, 0.0f, GRID_Z * CELL_WIDTH - 0.01f);
	}
}

__global__ void applyViscosity(int n, GridCell *cells, int GRID_X, int GRID_Y, int GRID_Z, float CELL_WIDTH, float TIME_STEP, float KINEMATIC_VISCOSITY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];

		glm::vec3 cellCoords = getCellUncompressedCoordinates(index, GRID_X, GRID_Y, GRID_Z);

		int cellTopIndex    = getCellCompressedIndex(cellCoords.x, cellCoords.y + 1, cellCoords.z, GRID_X, GRID_Y, GRID_Z);
		int cellBottomIndex = getCellCompressedIndex(cellCoords.x, cellCoords.y - 1, cellCoords.z, GRID_X, GRID_Y, GRID_Z);
		int cellLeftIndex   = getCellCompressedIndex(cellCoords.x - 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y, GRID_Z);
		int cellRightIndex  = getCellCompressedIndex(cellCoords.x + 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y, GRID_Z);
		int cellFrontIndex  = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z + 1, GRID_X, GRID_Y, GRID_Z);
		int cellBackIndex   = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z - 1, GRID_X, GRID_Y, GRID_Z);

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

		cell.tempVelocity = cell.velocity + TIME_STEP * KINEMATIC_VISCOSITY * glm::vec3(laplacianX, laplacianY, laplacianZ);
	}
}

__global__ void swapCellVelocities(int n, GridCell *cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		GridCell &cell = cells[index];
		cell.velocity = cell.tempVelocity;
	}
}

__global__ void setupPressureCalc(int numCells, float* csrValA, int* csrRowPtrA, int* csrColIndA, float* vecB, GridCell* cells, int GRID_X, int GRID_Y, int GRID_Z, float WidthDivTime) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index > 10 && index < 12) {
		glm::vec3 gridPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y, GRID_Z);

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
		vecB[index] = (WidthDivTime) * divU - airCells;
	}
}

__global__ void copyPressureToCells(int numCells, float* vecX, GridCell* cells) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numCells) {
		printf("%d: %f\n", index, vecX[index]);
		cells[index].pressure = vecX[index];
	}
}

void initSim() {
	// Allocate space for all of the grid cells
	cudaMalloc(&dev_gridCells, NUM_CELLS * sizeof(GridCell));
	cudaMemset(dev_gridCells, 0, NUM_CELLS * sizeof(GridCell));

	// Allocate space for all of the marker particles
	cudaMalloc(&dev_markerParticles, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));
	cudaMemset(dev_markerParticles, 0, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));

	// Allocate space for sparse linear solver of pressures
	/*nnz = NUM_CELLS * 27;
	cudaMalloc(&csrValA, nnz * sizeof(float));
	cudaMalloc(&csrRowPtrA, NUM_CELLS * sizeof(int));
	cudaMalloc(&csrColIndA, nnz * sizeof(int));
	cudaMalloc(&vecX, NUM_CELLS * sizeof(float));
	cudaMalloc(&vecB, NUM_CELLS * sizeof(float));*/

	// Create random world positions for all of the particles
	generateRandomWorldPositionsForParticles<<<BLOCKS_PARTICLES, blockSize>>>(NUM_MARKER_PARTICLES, dev_markerParticles, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH);
	checkCUDAError("generating initial world positions for marker particles failed");
	cudaDeviceSynchronize();

	// Initialize the grid cells
	initializeGridCells<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH);
	checkCUDAError("initializing the grid cells failed");
	cudaDeviceSynchronize();
}

void iterateSim() {
	// Make all the cells temporarily air cells
	setAllGridCellsToAir<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, AIR);
	checkCUDAError("marking all cells as air cells failed");
	cudaDeviceSynchronize();

	// Mark all cells with a marker particle as a fluid cell
	setGridCellsWithMarkerParticleToFluid<<<BLOCKS_PARTICLES, blockSize>>>(NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles, FLUID, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH);
	checkCUDAError("marking all cells with a marker particle as fluid cells failed");
	cudaDeviceSynchronize();

	// Apply convection to velocities using a backwards particle trace
	backwardsParticleTrace<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH, TIME_STEP);
	checkCUDAError("convecting velocities using a backwards particle trace failed");
	cudaDeviceSynchronize();

	// Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
	swapCellVelocities << <BLOCKS_CELLS, blockSize >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("swapping velocities in cells failed");
	cudaDeviceSynchronize();

	// Apply external forces to grid cell velocities
	applyExternalForcesToGridCells<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, TIME_STEP);
	checkCUDAError("applying external forces to cells failed");
	cudaDeviceSynchronize();

	// Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
	swapCellVelocities << <BLOCKS_CELLS, blockSize >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("swapping velocities in cells failed");
	cudaDeviceSynchronize();

	// Apply viscosity to velocities
	applyViscosity<<<BLOCKS_CELLS, blockSize>>>(NUM_CELLS, dev_gridCells, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH, TIME_STEP, KINEMATIC_VISCOSITY);
	checkCUDAError("applying viscosity failed");
	cudaDeviceSynchronize();

	// Set each cell velocity to be the temp velocity, needed since previous step had to save old velocities during calculations
	swapCellVelocities << <BLOCKS_CELLS, blockSize >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("swapping velocities in cells failed");
	cudaDeviceSynchronize();

	// Calculate pressure
	/*setupPressureCalc << <BLOCKS_CELLS, blockSize >> > (NUM_CELLS, csrValA, csrRowPtrA, csrColIndA, vecB, dev_gridCells, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH / TIME_STEP);
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

	copyPressureToCells << <BLOCKS_CELLS, blockSize >> > (NUM_CELLS, vecX, dev_gridCells);
	checkCUDAError("copy pressure to cells failed");
	cudaDeviceSynchronize();*/

	// Apply pressure

	// Extrapolate fluid velocities into surrounding cells

	// Set the velocities of surrounding cells

	// Move the marker particles through the velocity field
	moveMarkerParticlesThroughField<<<BLOCKS_PARTICLES, blockSize>>>(NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles, GRID_X, GRID_Y, GRID_Z, CELL_WIDTH, TIME_STEP);
	checkCUDAError("moving marker particles through velocity field failed");
	cudaDeviceSynchronize();
}
