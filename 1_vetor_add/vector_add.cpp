#include <Kokkos_Core.hpp>

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    // a scope guard is needed to ensure things get deallocated before finalize
    {
        size_t size = 1000;
        Kokkos::View<int *> vec_a("vec_a", size);
        Kokkos::View<int *> vec_b("vec_b", size);
        Kokkos::View<int *> vec_c("vec_c", size);
        Kokkos::parallel_for("fill vec_a", size, KOKKOS_LAMBDA(int i) { vec_a(i) = 1; });
        Kokkos::parallel_for("fill vec_b", size, KOKKOS_LAMBDA(int i) { vec_b(i) = 2; });
        Kokkos::parallel_for("sum", size, KOKKOS_LAMBDA(int i) { vec_c(i) = vec_a(i) + vec_b(i); });
        int sum;
        Kokkos::parallel_reduce(
            "accumulate", size,
            KOKKOS_LAMBDA(int i, int &partial_sum) { partial_sum += vec_c(i); }, sum);
        KOKKOS_ASSERT(sum == vec_c(0) * size);
        Kokkos::printf("sum = %d\n", sum);
    }
    Kokkos::finalize();

    return 0;
}
