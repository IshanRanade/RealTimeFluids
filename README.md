To run application:

1) Create a "build" folder in the root directory
2) cd into the "build" folder
3) Run the following, replace DCUDA_TOOLKIT_ROOT_DIR with the path to your toolkit:

`cmake -G "Visual Studio 15 2017 Win64" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2" -T v140,cuda=9.2 ..`

4) In the "build" folder you will see a .sln project, open that in Visual Studio


Initialization Steps

1. Setup OpenGL, get pbo working so we can fill it in CUDA
2. Copy rasterization project as base code

Grid Initialization Steps

1. Allocate arrays for velocity, pressure, marker particles
2. Create the height 3d grid, MAC grid, helper accessor methods
3. Initialize every grid cell with some scene file

Advection Steps

1. Figure out time step 
2. Advect pressure of grid cells
3. Advect velocity of grid cells
4. Advect attributes of marker particles
5. Advect level set
6. Add/remove marker particles based on new level set

Render Steps

1. Write a render kernel
2. Use raymarching through each pixel
3. Fill a pbo
