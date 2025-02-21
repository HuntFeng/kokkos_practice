find_package(Kokkos CONFIG)
add_executable(hello_kokkos hello_kokkos.cpp)
target_link_libraries(hello_kokkos Kokkos::kokkos)
