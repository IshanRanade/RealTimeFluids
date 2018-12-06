#include "fluid.h"
#include "hierarchy.h"
#include "lodepng.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/random.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <locale>


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

static LinearNode* dev_flatTree;
static float* vecX;
static float* vecB;
static float* valA;
static int* colIndA;

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

__global__ void fillVBOData(int n, void *vbo, MarkerParticle *particles, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    float *vboFloat = (float*)vbo;

    if (index < n) {
        MarkerParticle &particle = particles[index];

		int cellIndex = getCellCompressedIndex(particle.worldPosition.x, particle.worldPosition.y, particle.worldPosition.z, GRID_X, GRID_Y);
		GridCell &cell = cells[cellIndex];

        // Set the position
        vboFloat[6 * index + 0] = particle.worldPosition.x;
        vboFloat[6 * index + 1] = particle.worldPosition.y;
        vboFloat[6 * index + 2] = particle.worldPosition.z;

        // Set the color
		glm::vec3 color = glm::vec3(0.2f, 0.2f, 1.0f);

		if (cellIndex < NUM_CELLS) {
			//color = glm::abs(cell.velocity) * 40.0f;
		}

        vboFloat[6 * index + 3] = color.x;
        vboFloat[6 * index + 4] = color.y;
        vboFloat[6 * index + 5] = color.z;
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

// Intersection for node bounds
__device__ float boundsIntersectionTest(Bounds b, glm::vec3 rayPos, glm::vec3 rayDir, bool near = true) {
    float tmin = -999999.f;
    float tmax = 999999.f;

    for (int axis = 0; axis < 3; ++axis) {
        float axisDir = rayDir[axis];
        if (axisDir != 0) {
            float t1 = (b.min[axis] - rayPos[axis]) / axisDir;
            float t2 = (b.max[axis] - rayPos[axis]) / axisDir;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > 0 && ta > tmin)
                tmin = ta;
            if (tb < tmax)
                tmax = tb;
        }
    }

    if (tmax >= tmin && tmax > 0) {
        return near ? tmin : tmax;
    }
    return -1.f;
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

#if 0
__device__ float smin(float a, float b, float k) {
    const float h = glm::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return glm::mix(b, a, h) - k * h * (1.0f - h);
}

__device__ glm::vec4 smin(glm::vec3 vecA, glm::vec3 vecB, float a, float b, float k) {
    const float h = glm::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return glm::vec4(glm::mix(vecA, vecB, h), glm::mix(b, a, h) - k * h * (1.0f - h));
}
#endif

__device__ bool inBounds(const float value, const float bounds) {
    return (value >= -bounds) && (value <= bounds);
}

__device__ bool inBounds(const glm::vec3 min, const glm::vec3 max, const glm::vec3 pos) {
    return (pos.x >= min.x && pos.x <= max.x &&
        pos.y >= min.y && pos.y <= max.y &&
        pos.z >= min.z && pos.z <= max.z);
}

__global__ void raycastPBO(int numParticles, uchar4* pbo, MarkerParticle* particles, Camera camera, LinearNode* tree, int* particleIds, unsigned char* waterTex) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < camera.resolution.x && idy < camera.resolution.y) {
        // Setup variables
        bool intersected = false;
        glm::vec3 rayPos = camera.position;
        float tMin = 999999.0f;
        const glm::vec3 view = camera.view;
        const glm::vec3 up = camera.up;
        const glm::vec3 right = camera.right;
        glm::vec3 normal = glm::vec3(0, 1, 0);
        
        const glm::vec3 rayDir = glm::normalize(view
            - right * camera.pixelLength.x * ((float)idx - camera.resolution.x * 0.5f)
            - up * camera.pixelLength.y * ((float)idy - camera.resolution.y * 0.5f)
        );

#if QUAD_TREE
        // Quad tree traversal
        int nextOffset = 0;
        int currentNode = 0;
        int nodesToVisit[32];

        while(true) {
            LinearNode& node = tree[currentNode];
            const float boundsT = boundsIntersectionTest(node.bounds, rayPos, rayDir);
            if(boundsT > 0 && boundsT < tMin) {
                if(node.particleCount > 0) {
                    for (int i = 0; i < node.particleCount; ++i) {
                        const int particleId = particleIds[node.particlesOffset + i];
                        const float t = raySphereIntersect(rayPos, rayDir, particles[particleId].worldPosition, PARTICLE_RADIUS_SQUARE);
						if (t > 0 && t < tMin) {
							intersected = true;
                            normal = glm::normalize(rayPos + (rayDir * t) - particles[particleId].worldPosition);
							tMin = t;
						}
                    }
                    if (nextOffset == 0) break;
                    currentNode = nodesToVisit[--nextOffset];
                } else if (node.particleCount == -1) {
                    nodesToVisit[nextOffset++] = node.childOffset[0];
                    nodesToVisit[nextOffset++] = node.childOffset[1];
                    nodesToVisit[nextOffset++] = node.childOffset[2];
                    currentNode = currentNode + 1;
                }
            } else {
                if (nextOffset == 0) break;
                currentNode = nodesToVisit[--nextOffset];
            }
        }
#else
        // Ray-Sphere Intersection with all particles
        for (int i = 0; i < numParticles; ++i) {
            MarkerParticle& particle = particles[i];

            float t = raySphereIntersect(rayPos, rayDir, particle.worldPosition, PARTICLE_RADIUS);
            if (t > 0 && t < tMin) {
                intersected = true;
                tMin = t;
                normal = glm::normalize(rayPos + (rayDir * t) - particle.worldPosition);
            }
        }
#endif
        rayPos += rayDir * tMin;
        
        const int index = idx + idy * camera.resolution.x;
		const glm::vec3 clearColor = glm::vec3(245.0f, 245.0f, 220.0f);

        // Set the color
        if (intersected) {
            // Ray hit a marker particle
            glm::vec3 color = glm::vec3(0, 133.f, 164.f);
			float height = (rayPos.y / GRID_Y);
			color =  height * color + (1.f - height) * glm::vec3(0, 60.f, 81.f);

            // Debug velocity color
            //color = glm::abs(cells[getCellCompressedIndex(rayPos.x, rayPos.y, rayPos.z, GRID_X, GRID_Y)].velocity) * 40.0f;

            //float fresnel = glm::clamp(1.0f - glm::dot(normal, -rayPos), 0.0f, 1.0f);
            //fresnel = glm::pow(fresnel, 3.0f) * 0.65f;
            //color = fresnel * color + (1.0f - fresnel) * clearColor;

#if BLINN_PHONG
			const int normalX = (rayPos.x * (512.0f / GRID_X));
			const int normalZ = (rayPos.z * rayPos.y * (512.0f / GRID_Z / GRID_Y));
			const int normalId = 3 * (normalX + 512 * normalZ);
			normal = glm::normalize(glm::vec3(waterTex[normalId], waterTex[normalId + 2], waterTex[normalId + 1]));
            const glm::vec3 lightPos = glm::vec3(2, 1, 0);
            const float roughness = 10.0f;

            const glm::vec3 refl = glm::normalize(glm::normalize(camera.position - rayPos) + glm::normalize(lightPos));
            const float specularTerm = glm::pow(glm::max(glm::dot(refl, normal), 0.0f), roughness);

            color = color * (1.0f + specularTerm);
#endif

#if QUAD_TREE
			const float tMax = boundsIntersectionTest(tree[0].bounds, camera.position, rayDir, false);
            const float depth = glm::min((tMax - tMin) * 0.2f, 1.0f);
            color = depth * color + (1.0f - depth) * clearColor;
#endif

            pbo[index].x = glm::min(color.x, 255.0f);
            pbo[index].y = glm::min(color.y, 255.0f);
            pbo[index].z = glm::min(color.z, 255.0f);
            pbo[index].w = 0;
        }
        else {
            // Clear background
            pbo[index].x = clearColor.x;
            pbo[index].y = clearColor.y;
            pbo[index].z = clearColor.z;
            pbo[index].w = 0;
        }
    }
}

__global__ void checkParticlesToRender(int* particleIds, MarkerParticle* particles, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= NUM_MARKER_PARTICLES)
        return;

    const glm::vec3 cellPos = particles[index].worldPosition;
    for (int d = 0; d < 6; ++d) {
        const int x = d == 0 ? -1 : d == 1 ? 1 : 0;
        const int y = d == 2 ? -1 : d == 3 ? 1 : 0;
        const int z = d == 4 ? -1 : d == 5 ? 1 : 0;
	/*for (int x = -1; x < 1; ++x) {
		for (int y = -1; y < 1; ++y) {
			for (int z = -1; z < 1; ++z) {*/
		const int cellId = getCellCompressedIndex(cellPos.x + x, cellPos.y + y, cellPos.z + z, GRID_X, GRID_Y);

		if (inBounds(glm::vec3(0), glm::vec3(GRID_X, GRID_Y, GRID_Z), cellPos + glm::vec3(x, y, z))) {
			if (cells[cellId].cellType == AIR) {
				particleIds[index] = index;
				return;
			}
		}
		else {
			particleIds[index] = index;
			return;
		}
    }
    particleIds[index] = -1;
}

void raycastPBO(uchar4* pbo, Camera camera) {
#if QUAD_TREE
    // Initialize flat 3D quad tree hierarchy
    cudaMemcpy(markerParticles, dev_markerParticles, NUM_MARKER_PARTICLES * sizeof(MarkerParticle), cudaMemcpyDeviceToHost);
    //checkParticlesToRender << <BLOCKS_PARTICLES, BLOCK_SIZE >> > (dev_particleIds, dev_markerParticles, dev_gridCells);
    //cudaMemcpy(particleIds, dev_particleIds, NUM_MARKER_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAError("copy marker particles to cpu failed");

    std::vector<int> particles;
    for(int i = 0; i < NUM_MARKER_PARTICLES; ++i) {
        //if(particleIds[i] != -1)
			particles.push_back(i);
    }

    TreeNode* root = buildTree(particles, markerParticles, 1, glm::vec3(0), glm::vec3(GRID_X, GRID_Y, GRID_Z));
    const int numNodes = treeSize(root);
    std::vector<LinearNode> flatTree;
    for (int i = 0; i < numNodes; i++) {
        flatTree.push_back(LinearNode());
    }
    int offset = 0;
    particles.clear();
    flattenTree(root, particles, flatTree, &offset);
    deleteTree(root);
    cudaMemcpy(dev_flatTree, flatTree.data(), flatTree.size() * sizeof(LinearNode), cudaMemcpyHostToDevice);
    checkCUDAError("copy tree to device failed");
    if(particles.size() <= NUM_MARKER_PARTICLES)
        cudaMemcpy(dev_particleIds, particles.data(), particles.size() * sizeof(int), cudaMemcpyHostToDevice);
    else
        cudaMemcpy(dev_particleIds, particles.data(), NUM_MARKER_PARTICLES * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("copy particle ids to device failed");
#endif

    // Launch ray cast kernel
    const dim3 BLOCK_SIZE2d(16, 16); // TODO: tune block size
    const dim3 blocksPerGrid2d(
        (camera.resolution.x + BLOCK_SIZE2d.x - 1) / BLOCK_SIZE2d.x,
        (camera.resolution.y + BLOCK_SIZE2d.y - 1) / BLOCK_SIZE2d.y);
    raycastPBO << <blocksPerGrid2d, BLOCK_SIZE2d >> > (NUM_MARKER_PARTICLES, pbo, dev_markerParticles, camera, dev_flatTree, dev_particleIds, dev_waterTexture);
    checkCUDAError("raymarch to form PBO failed");
    cudaDeviceSynchronize();
}

__global__ void setAllGridCellsToAir(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        GridCell &cell = cells[index];
        cell.cellType = AIR;
		cell.layer = -1;
    }
}

__global__ void setGridCellsWithMarkerParticleToFluid(int n, GridCell* cells, MarkerParticle* particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        MarkerParticle &particle = particles[index];

        const int compressedCellIndex = getCellCompressedIndex(particle.worldPosition.x / CELL_WIDTH, particle.worldPosition.y / CELL_WIDTH, particle.worldPosition.z / CELL_WIDTH, GRID_X, GRID_Y);

        cells[compressedCellIndex].cellType = FLUID;
		cells[compressedCellIndex].layer = 0;

        //cells[compressedCellIndex].velocity = glm::vec3(0, -0.1, 0);

        /*const glm::vec3 cellPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);
        for (int i = 0; i < 5; ++i) {
            const int x = i == 0 ? -1 : i == 1 ? 1 : 0;
            const int y = i == 2 ? -1 : 0;
            const int z = i == 3 ? -1 : i == 4 ? 1 : 0;
            const int adjacent = getCellCompressedIndex(cellPos.x + x, cellPos.y + y, cellPos.z + z, GRID_X, GRID_Y);

            // Special case for bounds adjacent
            if (cellPos.x + x >= 0 && cellPos.x + x < GRID_X && cellPos.y + y >= 0 && cellPos.y + y < GRID_Y && cellPos.z + z >= 0 && cellPos.z + z < GRID_Z) {
                cells[adjacent].cellType = FLUID;
            }
        }*/
    }
}

void fillVBOsWithMarkerParticles(void *vbo) {
    fillVBOData << <BLOCKS_PARTICLES, BLOCK_SIZE >> > (NUM_MARKER_PARTICLES, vbo, dev_markerParticles, dev_gridCells);
    checkCUDAError("filling VBOs with marker particle data failed");
    cudaDeviceSynchronize();
}

__global__ void generateRandomWorldPositionsForParticles(MarkerParticle *particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < NUM_MARKER_PARTICLES) {
    /*    MarkerParticle &particle = particles[index];
        particle.worldPosition = getCellUncompressedCoordinates(index / 2, GRID_X, GRID_Y);
        if (index < n / 2)
            particle.worldPosition += glm::vec3(0.5);
        particle.color = glm::vec3(0.2, 0.2, 1);*/
		thrust::default_random_engine rngX = thrust::default_random_engine(index | (index << 22));
		thrust::default_random_engine rngY = thrust::default_random_engine(index | (index << 15) ^ index);
		thrust::default_random_engine rngZ = thrust::default_random_engine(index ^ (index * 13));
		thrust::uniform_real_distribution<float> u01(0, 1);

		MarkerParticle &particle = particles[index];
		//particle.worldPosition.x = 0.2 * u01(rngX) * GRID_X * CELL_WIDTH + 0.4 * GRID_X * CELL_WIDTH;
		//particle.worldPosition.y = 0.3 * u01(rngX) * GRID_Y * CELL_WIDTH + 0.4 * GRID_Y * CELL_WIDTH;
		//particle.worldPosition.z = 0.2 * u01(rngX) * GRID_Z * CELL_WIDTH + 0.4 * GRID_Z * CELL_WIDTH;

		particle.worldPosition.x = 0.1 * u01(rngX) * GRID_X * CELL_WIDTH + 0.501f;
		particle.worldPosition.y = 0.5 * u01(rngX) * GRID_Y * CELL_WIDTH + 0.501f;// +0.4 * GRID_Y * CELL_WIDTH;
		particle.worldPosition.z = 0.9 * u01(rngX) * GRID_Z * CELL_WIDTH + 0.501f;
    }
}

__device__ glm::vec3 getInterpolatedVelocity(int cellIndex, glm::vec3 particlePosition, GridCell *cells) {
	glm::vec3 cellCoords = getCellUncompressedCoordinates(cellIndex, GRID_X, GRID_Y);
	glm::vec3 cellCenter = cellCoords + glm::vec3(0.5, 0.5, 0.5);

	int cellXPlusIndex;
	int cellXMinusIndex;
	float xLerp;
	if (particlePosition.x >= cellCenter.x) {
		cellXMinusIndex = cellIndex;

		if (cellCoords.x + 1 < GRID_X) {
			cellXPlusIndex = getCellCompressedIndex(cellCoords.x + 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y);
			xLerp = particlePosition.x - cellCenter.x;
		}
		else {
			cellXPlusIndex = cellIndex;
			xLerp = 1.0f;
		}
	}
	else {
		cellXPlusIndex = cellIndex;

		if (cellCoords.x - 1 >= 0) {
			cellXMinusIndex = getCellCompressedIndex(cellCoords.x - 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y);
			xLerp = particlePosition.x - (cellCenter.x - 1);
		}
		else {
			cellXMinusIndex = cellIndex;
			xLerp = 1.0f;
		}
	}

	int cellYPlusIndex;
	int cellYMinusIndex;
	float yLerp;
	if (particlePosition.y >= cellCenter.y) {
		cellYMinusIndex = cellIndex;

		if (cellCoords.y + 1 < GRID_Y) {
			cellYPlusIndex = getCellCompressedIndex(cellCoords.x, cellCoords.y + 1, cellCoords.z, GRID_X, GRID_Y);
			yLerp = particlePosition.y - cellCenter.y;
		}
		else {
			cellYPlusIndex = cellIndex;
			yLerp = 1.0f;
		}
	}
	else {
		cellYPlusIndex = cellIndex;

		if (cellCoords.y - 1 >= 0) {
			cellYMinusIndex = getCellCompressedIndex(cellCoords.x, cellCoords.y - 1, cellCoords.z, GRID_X, GRID_Y);
			yLerp = particlePosition.y - (cellCenter.y - 1);
		}
		else {
			cellYMinusIndex = cellIndex;
			yLerp = 1.0f;
		}
	}


	int cellZPlusIndex;
	int cellZMinusIndex;
	float zLerp;
	if (particlePosition.z >= cellCenter.z) {
		cellZMinusIndex = cellIndex;

		if (cellCoords.z + 1 < GRID_Z) {
			cellZPlusIndex = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z + 1, GRID_X, GRID_Y);
			zLerp = particlePosition.z - cellCenter.z;
		}
		else {
			cellZPlusIndex = cellIndex;
			zLerp = 1.0f;
		}
	}
	else {
		cellZPlusIndex = cellIndex;

		if (cellCoords.z - 1 >= 0) {
			cellZMinusIndex = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z - 1, GRID_X, GRID_Y);
			zLerp = particlePosition.z - (cellCenter.z - 1);
		}
		else {
			cellZMinusIndex = cellIndex;
			zLerp = 1.0f;
		}
	}

	glm::vec3 interpolatedVelocity;
	interpolatedVelocity.x = cells[cellXMinusIndex].velocity.x * (1.0f - xLerp) + cells[cellXPlusIndex].velocity.x * xLerp;
	interpolatedVelocity.y = cells[cellYMinusIndex].velocity.y * (1.0f - yLerp) + cells[cellYPlusIndex].velocity.y * yLerp;
	interpolatedVelocity.z = cells[cellZMinusIndex].velocity.z * (1.0f - zLerp) + cells[cellZPlusIndex].velocity.z * zLerp;

	return interpolatedVelocity;
}

__global__ void backwardsParticleTrace(int n, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        GridCell &cell = cells[index];

        // For now just use simple Euler
        const glm::vec3 cellPosition = (getCellUncompressedCoordinates(index, GRID_X, GRID_Y) * CELL_WIDTH) + glm::vec3(CELL_WIDTH / 2.0, CELL_WIDTH / 2.0, CELL_WIDTH / 2.0);
        
		const glm::vec3 rungeKuttaPosition = cellPosition - (TIME_STEP / 2.0f) * cell.velocity;

		glm::vec3 interpolatedVelocity;
		if (rungeKuttaPosition.x < 0 || rungeKuttaPosition.x >= GRID_X || rungeKuttaPosition.y < 0 || rungeKuttaPosition.y >= GRID_Y || rungeKuttaPosition.z < 0 || rungeKuttaPosition.z >= GRID_Z) {
			interpolatedVelocity = cell.velocity;
		}
		else {
			interpolatedVelocity = getInterpolatedVelocity(getCellCompressedIndex(rungeKuttaPosition.x, rungeKuttaPosition.y, rungeKuttaPosition.z, GRID_X, GRID_Y), rungeKuttaPosition, cells);
		}

		const glm::vec3 oldPosition = cellPosition - TIME_STEP * interpolatedVelocity;

		if (oldPosition.x < 0 || oldPosition.x >= GRID_X || oldPosition.y < 0 || oldPosition.y >= GRID_Y || oldPosition.z < 0 || oldPosition.z >= GRID_Z) {
			return;
		}

        const int prevCellIndex = getCellCompressedIndex(oldPosition.x, oldPosition.y, oldPosition.z, GRID_X, GRID_Y);
        cell.tempVelocity = cells[prevCellIndex].velocity;
    }
}

__global__ void applyExternalForcesToGridCells(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        GridCell &cell = cells[index];

		if (cell.cellType == FLUID) {

			// Apply gravity
			cell.tempVelocity = cell.velocity + glm::vec3(0, -GRAVITY * TIME_STEP, 0);
		}
		else {
			cell.tempVelocity = cell.velocity;
		}
    }
}

__global__ void moveMarkerParticlesThroughField(int n, GridCell *cells, MarkerParticle *particles) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        MarkerParticle &particle = particles[index];

        // Find the cell that this particle is in
        int cellIndex = getCellCompressedIndex(particle.worldPosition.x, particle.worldPosition.y, particle.worldPosition.z, GRID_X, GRID_Y);
        
		glm::vec3 interpolatedVelocity = getInterpolatedVelocity(cellIndex, particle.worldPosition, cells);

		glm::vec3 rungeKuttaPosition = particle.worldPosition + (TIME_STEP / 2.0f) * interpolatedVelocity;

		glm::vec3 newVelocity;
		if (rungeKuttaPosition.x < 0 || rungeKuttaPosition.x >= GRID_X || rungeKuttaPosition.y < 0 || rungeKuttaPosition.y >= GRID_Y || rungeKuttaPosition.z < 0 || rungeKuttaPosition.z >= GRID_Z) {
			newVelocity = interpolatedVelocity;
		}
		else {
			newVelocity = getInterpolatedVelocity(getCellCompressedIndex(rungeKuttaPosition.x, rungeKuttaPosition.y, rungeKuttaPosition.z, GRID_X, GRID_Y), rungeKuttaPosition, cells);
		}

        particle.worldPosition += TIME_STEP * newVelocity;
        particle.worldPosition.x = glm::clamp(particle.worldPosition.x, 0.501f, GRID_X * CELL_WIDTH - 0.501f);
        particle.worldPosition.y = glm::clamp(particle.worldPosition.y, 0.501f, GRID_Y * CELL_WIDTH - 0.501f);
        particle.worldPosition.z = glm::clamp(particle.worldPosition.z, 0.501f, GRID_Z * CELL_WIDTH - 0.501f);
    }
}

__global__ void applyViscosity(int n, GridCell *cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        GridCell &cell = cells[index];

		if (cell.cellType != FLUID) {
			cell.tempVelocity = cell.velocity;
			return;
		}

        glm::vec3 cellCoords = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);

		float laplacianX = 0.0;
		float laplacianY = 0.0;
		float laplacianZ = 0.0;

		int count = 0;

		if (cellCoords.x + 1 < GRID_X) {
			int adjacent = getCellCompressedIndex(cellCoords.x + 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y);

			if (cells[adjacent].cellType == FLUID) {
				count++;

				laplacianX += cells[adjacent].velocity.x;
				laplacianY += cells[adjacent].velocity.y;
				laplacianZ += cells[adjacent].velocity.z;
			}
		}

		if (cellCoords.x - 1 >= 0) {
			int adjacent = getCellCompressedIndex(cellCoords.x - 1, cellCoords.y, cellCoords.z, GRID_X, GRID_Y);

			if (cells[adjacent].cellType == FLUID) {
				count++;

				laplacianX += cells[adjacent].velocity.x;
				laplacianY += cells[adjacent].velocity.y;
				laplacianZ += cells[adjacent].velocity.z;
			}
		}


		if (cellCoords.y + 1 < GRID_Y) {
			int adjacent = getCellCompressedIndex(cellCoords.x, cellCoords.y + 1, cellCoords.z, GRID_X, GRID_Y);

			if (cells[adjacent].cellType == FLUID) {
				count++;

				laplacianX += cells[adjacent].velocity.x;
				laplacianY += cells[adjacent].velocity.y;
				laplacianZ += cells[adjacent].velocity.z;
			}
		}

		if (cellCoords.y - 1 >= 0) {
			int adjacent = getCellCompressedIndex(cellCoords.x, cellCoords.y - 1, cellCoords.z, GRID_X, GRID_Y);

			if (cells[adjacent].cellType == FLUID) {
				count++;

				laplacianX += cells[adjacent].velocity.x;
				laplacianY += cells[adjacent].velocity.y;
				laplacianZ += cells[adjacent].velocity.z;
			}
		}


		if (cellCoords.z + 1 < GRID_Z) {
			int adjacent = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z + 1, GRID_X, GRID_Y);

			if (cells[adjacent].cellType == FLUID) {
				count++;

				laplacianX += cells[adjacent].velocity.x;
				laplacianY += cells[adjacent].velocity.y;
				laplacianZ += cells[adjacent].velocity.z;
			}
		}

		if (cellCoords.z - 1 >= 0) {
			int adjacent = getCellCompressedIndex(cellCoords.x, cellCoords.y, cellCoords.z - 1, GRID_X, GRID_Y);

			if (cells[adjacent].cellType == FLUID) {
				count++;

				laplacianX += cells[adjacent].velocity.x;
				laplacianY += cells[adjacent].velocity.y;
				laplacianZ += cells[adjacent].velocity.z;
			}
		}

        laplacianX -= count * cell.velocity.x;
        laplacianY -= count * cell.velocity.y;
        laplacianZ -= count * cell.velocity.z;

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

	//if (cells[index].cellType == AIR)
	//	return;

    const glm::vec3 cellPos = getCellUncompressedCoordinates(index, grid.sizeX, grid.sizeY);

    float nonSolid = 1.0f;
    float airCells = 0.0f;
    for (int i = 0; i < 6; ++i) {
        const int x = i == 0 ? -1 : i == 1 ? 1 : 0;
        const int y = i == 2 ? -1 : i == 3 ? 1 : 0;
        const int z = i == 4 ? -1 : i == 5 ? 1 : 0;
   
        // Special case for bounds adjacent
        if (cellPos.x + x < 0 || cellPos.x + x >= grid.sizeX || cellPos.y + y < 0 || cellPos.y + y >= grid.sizeY || cellPos.z + z < 0 || cellPos.z + z >= grid.sizeZ) {
            grid.dev_colIndA[index * 6 + i] = -1;
            continue;
        }

	


		const int adjacent = getCellCompressedIndex(cellPos.x + x, cellPos.y + y, cellPos.z + z, grid.sizeX, grid.sizeY);
        GridCell& adjacentCell = cells[adjacent];

		++nonSolid;

		if (adjacentCell.cellType == AIR) {
			grid.dev_colIndA[index * 6 + i] = -1;
			++airCells;
			continue;
			
		}

        // Set index of adjacent cell
        grid.dev_colIndA[index * 6 + i] = adjacent;

        // Set value of matrix element
        grid.dev_valA[index * 7 + i + 1] = 1.0f;
    }
    // Set matrix value for current grid cell
	if (nonSolid == 0) {
		printf("here");
	}
    grid.dev_valA[index * 7] = -nonSolid;

    // Set value of b vector for pressure solver
    float divU = 0.0f;
    GridCell& cell = cells[index];

    float divPlus = 0.0f;
    float divMinus = 0.0f;
    if (cellPos.x + 1 < grid.sizeX) {
        GridCell& adjacent = cells[index + 1];
        divPlus = 0.5 * (cell.velocity.x + adjacent.velocity.x);
    }
    if (cellPos.x - 1 >= 0) {
        GridCell& adjacent = cells[index - 1];
        divMinus = 0.5 * (cell.velocity.x + adjacent.velocity.x);
    }
    divU += (divPlus - divMinus);

    divPlus = 0.0f;
    divMinus = 0.0f;
    if (cellPos.y + 1 < grid.sizeY) {
        GridCell& adjacent = cells[index + grid.sizeX];
        divPlus = 0.5 * (cell.velocity.y + adjacent.velocity.y);
    }
    if (cellPos.y - 1 >= 0) {
        GridCell& adjacent = cells[index - grid.sizeX];
        divMinus = 0.5 * (cell.velocity.y + adjacent.velocity.y);
    }
    divU += (divPlus - divMinus);

    divPlus = 0.0f;
    divMinus = 0.0f;
    if (cellPos.z + 1 < grid.sizeZ) {
        GridCell& adjacent = cells[index + grid.sizeX * grid.sizeY];
        divPlus = 0.5 * (cell.velocity.y + adjacent.velocity.y);
    }
    if (cellPos.z - 1 >= 0) {
        GridCell& adjacent = cells[index - grid.sizeX * grid.sizeY];
        divMinus = 0.5 * (cell.velocity.z + adjacent.velocity.z);
    }
    divU += (divPlus - divMinus);
    divU /= CELL_WIDTH;

	grid.dev_B[index] = divU * (cell.cellType == FLUID ? FLUID_DENSITY : AIR_DENSITY) / (TIME_STEP) - ATMOSPHERIC_PRESSURE * airCells;
	
	//if (cell.cellType == AIR) {
	//	grid.dev_B[index] = 0.0f;
	//}
}

__global__ void gaussSeidelPressure(Grid grid, int redBlack) {
	//const int index = blockIdx.x * blockDim.x + threadIdx.x;
	//
 //   if (index >= grid.numCells)
 //       return;

	//if (( (index / NUM_CELLS) + (index % NUM_CELLS)) % 2 == redBlack) {
	//	
	//	float numerator = grid.dev_B[index];
	//	for (int j = 0; j < 6; ++j) {
	//		if (grid.dev_colIndA[index * 6 + j] != -1)
	//			numerator -= grid.dev_valA[index * 7 + j + 1] * grid.dev_X[grid.dev_colIndA[index * 6 + j]];
	//	}
	//	grid.dev_X[index] = numerator / grid.dev_valA[index * 7];
	//}
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

// change vecX to pressure array, take out pressure from cells, delete this kernel
__global__ void copyPressureToCells(int numCells, float* pressure, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numCells) {
		/*if (index < 100) {
			if (cells[index].cellType == FLUID) {
				printf("%d: %f\n", index, cells[index].pressure);
			}
		}*/
		cells[index].pressure = cells[index].cellType == AIR ? ATMOSPHERIC_PRESSURE : pressure[index];
    }
}

__global__ void clampCellVelocities(int numCells, GridCell *cells) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numCells) {
		GridCell &cell = cells[index];

		if (glm::length(cell.velocity) > MAX_VELOCITY) {
			cell.velocity = glm::normalize(cell.velocity) * MAX_VELOCITY;
		}
	}
}

__global__ void setAirCellPressureAndVelocity(int numCells, GridCell* cells) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numCells) {
		GridCell& cell = cells[index];

		if (cell.cellType == AIR) {

			glm::vec3 cellPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);
			if (cellPos.y - 1 >= 0) {
				int belowCellIndex = getCellCompressedIndex(cellPos.x, cellPos.y - 1, cellPos.z, GRID_X, GRID_Y);
				
				if (cells[belowCellIndex].cellType == FLUID) {
					cell.pressure = ATMOSPHERIC_PRESSURE;
					//cell.tempVelocity = cell.velocity;
					return;
				}
			}
            cell.pressure = ATMOSPHERIC_PRESSURE;
			//cell.velocity = glm::vec3(0.0);
        }
	}
}

__global__ void applyPressure(int numCells, GridCell* cells) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < numCells) {
        GridCell& cell = cells[index];
   //     if (cell.cellType != FLUID) {
   //         cell.pressure = 1.0f;
			//cell.tempVelocity = glm::vec3(0);
   //         return;
   //     }

		if (cell.cellType == FLUID) {
			const glm::vec3 cellPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);
			glm::vec3 deltaPressure = glm::vec3(0);

			if (cellPos.x + 1 < GRID_X) {
				GridCell& adjacent = cells[index + 1];
				deltaPressure.x += adjacent.pressure;
			}
			if (cellPos.x - 1 >= 0) {
				GridCell& adjacent = cells[index - 1];
				deltaPressure.x -= adjacent.pressure;
			}
			if (cellPos.y + 1 < GRID_Y) {
				GridCell& adjacent = cells[index + GRID_X];
				deltaPressure.y += adjacent.pressure;
			}
			if (cellPos.y - 1 >= 0) {
				GridCell& adjacent = cells[index - GRID_X];
				deltaPressure.y -= adjacent.pressure;
			}
			if (cellPos.z + 1 < GRID_Z) {
				GridCell& adjacent = cells[index + GRID_X * GRID_Y];
				deltaPressure.z += adjacent.pressure;
			}
			if (cellPos.z - 1 >= 0) {
				GridCell& adjacent = cells[index - GRID_X * GRID_Y];
				deltaPressure.z -= adjacent.pressure;
			}

			//if(index < 100)
			//printf("%d\n", deltaPressure.y);

			/*if (glm::length(deltaPressure) > 10.0) {
				deltaPressure = glm::normalize(deltaPressure) * 10.0f;
			}*/
			cell.tempVelocity = cell.velocity - deltaPressure * TIME_STEP / ((cell.cellType == FLUID ? FLUID_DENSITY : AIR_DENSITY) * CELL_WIDTH);
		}
	}
}

__global__ void setVelocitiesIntoSolidsAsZero(int numCells, GridCell* cells) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numCells)
	{
		GridCell &cell = cells[index];
		glm::vec3 cellPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);
		

		if (cellPos.x == GRID_X - 1 && cell.velocity.x > 0) {
			//cell.velocity.x *= -1.0;
			cell.velocity.x = 0.0f;
		}
		if (cellPos.x == 0 && cell.velocity.x < 0) {
			//cell.velocity.x *= 1.0;
			cell.velocity.x = 0.0f;
		}

		if (cellPos.y == GRID_Y - 1 && cell.velocity.y > 0) {
			//cell.velocity.y *= -1.0;
			cell.velocity.y = 0.0f;
		}
		if (cellPos.y == 0 && cell.velocity.y < 0) {
			//cell.velocity.y *= 1.0;
			cell.velocity.y = 0.0f;
		}

		if (cellPos.z == GRID_Z - 1 && cell.velocity.z > 0) {
			//cell.velocity.z *= -1.0;
			cell.velocity.z = 0.0f;
		}
		if (cellPos.z == 0 && cell.velocity.z < 0) {
			//cell.velocity.z *= 1.0;
			cell.velocity.z = 0.0f;
		}
	}
}

__global__ void extrapolateFluidVelocities(int numCells, GridCell* cells, int layer)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numCells)
	{
		GridCell &cell = cells[index];
		glm::vec3 cellPos = getCellUncompressedCoordinates(index, GRID_X, GRID_Y);

		glm::vec3 averageVelocity = glm::vec3(0.0f);
		int count = 0;
		if (cell.layer == -1) {
			for (int d = 0; d < 6; ++d) {
				const int x = d == 0 ? -1 : d == 1 ? 1 : 0;
				const int y = d == 2 ? -1 : d == 3 ? 1 : 0;
				const int z = d == 4 ? -1 : d == 5 ? 1 : 0;

				if (cellPos.x + x >= 0 && cellPos.x + x < GRID_X && cellPos.y + y >= 0 && cellPos.y + y < GRID_Y &&cellPos.z + z >= 0 && cellPos.z + z < GRID_Z) {
					int adjacentCell = getCellCompressedIndex(cellPos.x + x, cellPos.y + y, cellPos.z + z, GRID_X, GRID_Y);

					if (cells[adjacentCell].layer == layer - 1) {
						//if (cells[adjacentCell].cellType != FLUID) {
							averageVelocity += cells[adjacentCell].velocity;
							count++;
						//}
					}
				}
			}

			cell.layer = layer;

			if (count == 0) {
				cell.tempVelocity = cell.velocity;
			}
			else {
				cell.tempVelocity = (averageVelocity) / (float(count));
			}
		}
		else {
			cell.tempVelocity = cell.velocity;
		}
	}
}

void initHierarchicalPressureGrids() {
    // Calculate number of grid levels
    MAX_GRID_LEVEL = 1;//std::floor(log2(std::min(std::min(GRID_X, GRID_Y), GRID_Z)));

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

    for (int d = 1; d < MAX_GRID_LEVEL; ++d) {
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
	markerParticles = (MarkerParticle*)malloc(NUM_MARKER_PARTICLES * sizeof(MarkerParticle));
    cudaMalloc(&dev_markerParticles, NUM_MARKER_PARTICLES * sizeof(MarkerParticle));
    checkCUDAError("malloc marker particles failed");

    // Create random world positions for all of the particles
    generateRandomWorldPositionsForParticles << <BLOCKS_PARTICLES, BLOCK_SIZE >> > (dev_markerParticles);
    checkCUDAError("generating initial world positions for marker particles failed");
    cudaDeviceSynchronize();

    // Allocate space for quad tree hierarchy
    cudaMalloc(&dev_flatTree, sizeof(LinearNode) * 512);
    cudaMalloc(&dev_particleIds, sizeof(int) * NUM_MARKER_PARTICLES);
    checkCUDAError("allocating space for flat tree failed");
    cudaDeviceSynchronize();

	// Read water normal texture
	std::vector<unsigned char> image;
	unsigned width, height;
	unsigned error = lodepng::decode(image, width, height, "../img/water_normal.png", LCT_RGB);
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	else {
		cudaMalloc(&dev_waterTexture, sizeof(unsigned char) * width * height * 3);
		cudaMemcpy(dev_waterTexture, image.data(), sizeof(unsigned char) * image.size(), cudaMemcpyHostToDevice);
	}

    particleIds = (int*)malloc(NUM_MARKER_PARTICLES * sizeof(int));
    valA = (float*)malloc(NUM_CELLS * 7 * sizeof(float));
    colIndA = (int*)malloc(NUM_CELLS * 6 * sizeof(int));
    vecB = (float*)malloc(NUM_CELLS * sizeof(float));
    vecX = (float*)malloc(NUM_CELLS * sizeof(float));
}

__global__ void resetGridCells(int numCells, GridCell *cells) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < numCells) {
		GridCell &cell = cells[index];

		cell.velocity = glm::vec3(0);
		cell.tempVelocity = glm::vec3(0);
	}
}

void restartSim() {
	// Create random world positions for all of the particles
	generateRandomWorldPositionsForParticles << <BLOCKS_PARTICLES, BLOCK_SIZE >> > (dev_markerParticles);
	checkCUDAError("generating initial world positions for marker particles failed");
	cudaDeviceSynchronize();

	// Reset the velocities of the grid cells to 0
	resetGridCells << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("resetting the grid cells failed");
	cudaDeviceSynchronize();
}

void freeSim() {
    cudaFree(dev_markerParticles);
    cudaFree(dev_gridCells);
    cudaFree(dev_flatTree);
    cudaFree(dev_particleIds);
	free(markerParticles);
    free(particleIds);
    free(vecB);
    free(vecX);
    free(valA);
    free(colIndA);
    for (int d = 0; d < MAX_GRID_LEVEL; ++d) {
        cudaFree(grids[d].dev_B);
        cudaFree(grids[d].dev_X);
        cudaFree(grids[d].dev_colIndA);
        cudaFree(grids[d].dev_valA);
        cudaFree(grids[d].dev_B);
    }
}

void iterateSim() {
    // Make all the cells temporarily air cells
    setAllGridCellsToAir << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("marking all cells as air cells failed");
    cudaDeviceSynchronize();

    // Mark all cells with a marker particle as a fluid cell
    setGridCellsWithMarkerParticleToFluid << <BLOCKS_PARTICLES, BLOCK_SIZE >> > (NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles);
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
    applyExternalForcesToGridCells << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("applying external forces to cells failed");
    cudaDeviceSynchronize();

    swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("swapping velocities in cells failed");
    cudaDeviceSynchronize();

    // Apply viscosity to velocities
    applyViscosity<<<BLOCKS_CELLS, BLOCK_SIZE>>>(NUM_CELLS, dev_gridCells);
    checkCUDAError("applying viscosity failed");
    cudaDeviceSynchronize();

    swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("swapping velocities in cells failed");
    cudaDeviceSynchronize();

    // Setup pressure calculation
    setupPressureCalc << <BLOCKS_CELLS, BLOCK_SIZE >> > (grids[0], dev_gridCells);
    checkCUDAError("setup pressure calc failed");
    cudaDeviceSynchronize();

	//int GAUSS_SEIDEL_BLOCKS = 
    // Gauss Seidel Pressure Solver
	memset(vecX, 0.0f, NUM_CELLS * sizeof(float));
	cudaMemcpy(valA, grids[0].dev_valA, NUM_CELLS * 7 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(colIndA, grids[0].dev_colIndA, NUM_CELLS * 6 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(vecB, grids[0].dev_B, NUM_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < GAUSS_ITERATIONS; ++i) {
		//printf("%d: %f\n", i, vecX[1]);
        gaussSeidelPressureCPU(NUM_CELLS, valA, colIndA, vecX, vecB);
    }
	//printf("\n\n\n");
	cudaMemcpy(grids[0].dev_X, vecX, NUM_CELLS * sizeof(float), cudaMemcpyHostToDevice);

    // Copy pressure to cells
    copyPressureToCells << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, grids[0].dev_X, dev_gridCells);
    checkCUDAError("copy pressure to cells failed");
    cudaDeviceSynchronize();

    // Apply pressure
    applyPressure << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("applying pressure to cell velocities failed");
    cudaDeviceSynchronize();

    swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
    checkCUDAError("swapping velocities in cells failed");
    cudaDeviceSynchronize();


	// Extrapolate fluid velocities into surrounding cells
	for (int i = 1; i < 5; i++) {
		extrapolateFluidVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells, i);
		checkCUDAError("extrapolating velocities failed");
		cudaDeviceSynchronize();

		swapCellVelocities << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
		checkCUDAError("swapping velocities in cells failed");
		cudaDeviceSynchronize();
	}
	
	// Set velocity of cells pointing to solids
	setVelocitiesIntoSolidsAsZero << <BLOCKS_CELLS, BLOCK_SIZE >> > (NUM_CELLS, dev_gridCells);
	checkCUDAError("setting velocities into solids as zero failed");
	cudaDeviceSynchronize();

    // Move the marker particles through the velocity field
    moveMarkerParticlesThroughField << <BLOCKS_PARTICLES, BLOCK_SIZE >> > (NUM_MARKER_PARTICLES, dev_gridCells, dev_markerParticles);
    checkCUDAError("moving marker particles through velocity field failed");
    cudaDeviceSynchronize();
}
