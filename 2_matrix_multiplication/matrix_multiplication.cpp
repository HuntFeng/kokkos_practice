#include <Kokkos_Core.hpp>

using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamPolicy;
using Kokkos::ThreadVectorRange;
using Kokkos::View;

void print(const Kokkos::View<int **> &x, int row, int col) {
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      Kokkos::printf("%d ", x(i, j));
    }
    Kokkos::printf("\n");
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  // a scope guard is needed to ensure things get deallocated before finalize
  {
    const size_t M = 3;
    const size_t K = 2;
    const size_t N = 4;
    View<int **> mat_a("mat_a", M, K);
    View<int **> mat_b("mat_b", K, N);
    View<int **> mat_c("mat_c", M, N);

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        mat_a(i, j) = (i == j) ? 1 : 0;
      }
    }

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        mat_b(i, j) = i + j;
      }
    }

    Kokkos::printf("mat_a = \n");
    print(mat_a, M, K);
    Kokkos::printf("mat_b = \n");
    print(mat_b, K, N);

    parallel_for(
        TeamPolicy<>(M, N, K), KOKKOS_LAMBDA(TeamPolicy<>::member_type team) {
          int i = team.league_rank();
          int j = team.team_rank();
          mat_c(i, j) = 0;
          parallel_reduce(
              ThreadVectorRange(team, K),
              [=](int k, int &partial_sum) {
                partial_sum += mat_a(i, k) * mat_b(k, j);
              },
              mat_c(i, j));
        });

    Kokkos::printf("mat_c = \n");
    print(mat_c, M, N);
  }
  Kokkos::finalize();

  return 0;
}
