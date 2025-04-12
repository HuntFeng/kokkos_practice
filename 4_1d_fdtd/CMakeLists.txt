cmake_minimum_required(VERSION 3.28)
project(1D_FDTD)
set(CMAKE_CXX_STANDARD 20)

find_package(gridformat REQUIRED)
find_package(Kokkos REQUIRED)
add_executable(1d_fdtd 1d_fdtd.cpp)
target_link_libraries(1d_fdtd INIReader gridformat::gridformat Kokkos::kokkos)
