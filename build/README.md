### Build and Run

To run application:

1) cd into the "build" folder in a terminal window
2) Run the following, replace DCUDA_TOOLKIT_ROOT_DIR with the path to your toolkit, replace cuda version with any version at least 8.0:

`cmake -G "Visual Studio 15 2017 Win64" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2" -T v140,cuda=9.2 ..`

3) In the "build" folder you will see a .sln project, open that in Visual Studio