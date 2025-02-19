find_package(Kokkos CONFIG)
add_executable(HelloKokkos HelloKokkos.cpp)
target_link_libraries(HelloKokkos Kokkos::kokkos)