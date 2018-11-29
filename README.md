Real Time Eulerian Fluids
=========================

* Salaar Kohari and Ishan Ranade
* Tested on: Windows 10, Intel Xeon @ 3.1GHz 32GB, GTX 980 4GB

### Introduction

An open-source repository featuring **real-time Eulerian fluid simulation and rendering** based on SIGGRAPH research, primarily [Real-Time Eulerian Water Simulation Using a Restricted Tall Cell Grid](http://matthias-mueller-fischer.ch/publications/tallCells.pdf).

### Build and Run

To run application:

1) Create a "build" folder in the root directory
2) cd into the "build" folder
3) Run the following, replace DCUDA_TOOLKIT_ROOT_DIR with the path to your toolkit:

`cmake -G "Visual Studio 15 2017 Win64" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2" -T v140,cuda=9.2 ..`

4) In the "build" folder you will see a .sln project, open that in Visual Studio

### Simulation Procedure

##### Initialization

1. Tune simulation parameters, i.e. time step and grid
2. Allocate arrays for velocity, pressure, marker particles
3. Initialize tall cell grid with some scene file

##### Advection

1. Advect velocity of tall cells
2. Advect pressure of tall cells
3. Advect attributes of marker particles
4. Add/remove marker particles based on new tall cells

##### Rendering

1. Ray trace kernel for each pixel
2. Check hierarchy for possible water particle collisions
3. Compute ray-sphere intersections for those particles
4. Use near/far distance to color collisions
5. Fill a pbo to render with OpenGL

### References

- [Real-Time Eulerian Water Simulation Using a Restricted Tall Cell Grid, Chentanez and Muller](http://matthias-mueller-fischer.ch/publications/tallCells.pdf)
- [Paper Name](https://www.paperlink.com)
- [Paper Name](https://www.paperlink.com)