Real Time Eulerian Fluids
=========================

* Salaar Kohari and Ishan Ranade
* Tested on: Windows 10, Intel Xeon @ 3.1GHz 32GB, GTX 980 4GB

### Introduction

An open-source repository featuring **real-time Eulerian fluid simulation and rendering** based on SIGGRAPH research, primarily [Real-Time Eulerian Water Simulation Using a Restricted Tall Cell Grid](http://matthias-mueller-fischer.ch/publications/tallCells.pdf). Fluid simulation is performed using discrete Navier Stokes on a collocated MAC grid with marker particles moving through fluid cells. Rendering involves raycasting through a quad tree of particles and shading the result. All parts of the simulation and rendering are implemented on the GPU except the Gauss-Seidel solver and quad tree generation.

### Build and Run

To run application:

1) Create a "build" folder in the root directory
2) cd into the "build" folder
3) Run the following, replace DCUDA_TOOLKIT_ROOT_DIR with the path to your toolkit:

`cmake -G "Visual Studio 15 2017 Win64" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2" -T v140,cuda=9.2 ..`

4) In the "build" folder you will see a .sln project, open that in Visual Studio

### Simulation Procedure

##### Initialization

1. Tune simulation parameters (time step, grid, particles)
2. Allocate memory for grid cells, pressure solver, and marker particles
3. Initialize particle positions with initial scene conditions

##### Advection

1. Set cells with marker particles to fluid
2. Apply convection using backwards particle trace
3. Add external forces (gravity/wind)
4. Apply viscosity
5. Setup pressure calculations for solver
6. Gauss-Seidel to solve for pressure (CPU)
7. Velocity fluid/solid extrapolation
8. Move marker particles using cell velocity interpolation

##### Rendering

1. Cull particles below the surface and boundaries
2. Create quad tree for marker particles (CPU)
3. Ray trace kernel for each pixel
4. Check hierarchy for possible water particle collisions
5. Compute ray-sphere intersections for those particles
6. Use depth and normals to color the result
7. Fill a pbo to display with OpenGL

### References

- [Real-Time Eulerian Water Simulation Using a Restricted Tall Cell Grid](http://matthias-mueller-fischer.ch/publications/tallCells.pdf)
- [Fluid Flow for the Rest of Us: Tutorial of the Marker and Cell Method in
Computer Graphics](https://pdfs.semanticscholar.org/9d47/1060d6c48308abcc98dbed850a39dbfea683.pdf)
- Contact us for questions or more references
