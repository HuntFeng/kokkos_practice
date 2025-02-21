find_package(Kokkos CONFIG)
add_executable(vector_add vector_add.cpp)
target_link_libraries(vector_add Kokkos::kokkos)
