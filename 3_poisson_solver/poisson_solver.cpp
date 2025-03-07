#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <tuple>

using Kokkos::MDRangePolicy;
using Kokkos::parallel_for;
using Kokkos::printf;
using Kokkos::View;

int read_input(std::array<double, 2>& x1_range, std::array<double, 2>& x2_range, std::array<int, 2>& n) {
    INIReader reader("input.ini");

    if (reader.ParseError() < 0) {
        printf("Can't load 'input.ini'\n");
        return -1;
    }

    x1_range[0] = reader.GetReal("dimension", "x1_min", 0);
    x1_range[1] = reader.GetReal("dimension", "x1_max", 1);
    x2_range[0] = reader.GetReal("dimension", "x2_min", 0);
    x2_range[1] = reader.GetReal("dimension", "x2_max", 1);
    n[0] = reader.GetInteger("dimension", "nx", -1);
    n[1] = reader.GetInteger("dimension", "ny", -1);

    if (x1_range[1] - x1_range[0] <= 0 || x2_range[1] - x2_range[0] <= 0) {
        printf("Error: One or more domain size is non-positive\n");
        return -1;
    }
    if (n[0] < 0 || n[1] < 0) {
        printf("Error: One or more of nx, ny is missing\n");
        return -1;
    }
    return 0;
}

std::tuple<View<double**>, View<double**>>
prepare_mesh(std::array<double, 2>& x1_range, std::array<double, 2>& x2_range, std::array<int, 2> n) {
    View<double**> mesh_x("mesh_x", n[0], n[1]);
    View<double**> mesh_y("mesh_y", n[0], n[1]);
    parallel_for(MDRangePolicy({ 0, 0 }, { n[0], n[1] }), [=](int i, int j) {
        mesh_x(i, j) = x1_range[0] + i * (x1_range[1] - x1_range[0]) / (n[0] - 1);
        mesh_y(i, j) = x2_range[0] + j * (x2_range[1] - x2_range[0]) / (n[1] - 1);
    });
    return { mesh_x, mesh_y };
}

void init_field(const View<double**>& u) {
    parallel_for(
        MDRangePolicy({ 0, 0 }, { u.extent(0), u.extent(1) }), [=](int i, int j) {
            if (i == 0 || j == 0 || i == u.extent(0) - 1 || j == u.extent(1) - 1)
                u(i, j) = 1.0;
            else
                u(i, j) = 0.0;
        }
    );
}

void advance(const View<double**>& u, const View<double**>& u_old) {
    // iterate on old value and then update to new value
    parallel_for(
        MDRangePolicy({ 0, 0 }, { u.extent(0), u.extent(1) }), [=](int i, int j) {
            if (i > 0 && i < u.extent(0) - 1 && j > 0 && j < u.extent(1) - 1) {
                u(i, j) = (u_old(i - 1, j) + u_old(i + 1, j) + u_old(i, j - 1) + u_old(i, j + 1)) / 4;
            }
        }
    );
}

void save(const View<double**>& u, const View<double**>& mesh_x, const View<double**>& mesh_y, int step) {
    std::fstream fs;
    fs.open(std::format("data/step_{:03d}.csv", step), std::fstream::out);
    fs << "x, y, u\n";
    for (int i = 0; i < u.extent(0); ++i) {
        for (int j = 0; j < u.extent(1); ++j) {
            fs << std::format("{}, {}, {}\n", mesh_x(i, j), mesh_y(i, j), u(i, j));
        }
    }
}

int main(int argc, char** argv) {
    printf("Reading inputs...\n");
    std::array<double, 2> x1_range, x2_range;
    std::array<int, 2> n;
    int success = read_input(x1_range, x2_range, n);
    if (success < 0) {
        return -1;
    }
    printf("x1_range=[%f, %f], x2_range=[%f, %f], nx, ny=%d, %d\n", x1_range[0], x1_range[1], x2_range[0], x2_range[1], n[0], n[1]);

    Kokkos::initialize(argc, argv);
    // a scope guard is needed to ensure things get deallocated before finalize
    {
        printf("Initializing fields...\n");
        const auto [mesh_x, mesh_y] = prepare_mesh(x1_range, x2_range, n);
        View<double**> u("u", n[0], n[1]);
        View<double**> u_old("u_old", n[0], n[1]);
        init_field(u);

        int total_steps = 100;
        int diag_step = 10;
        std::filesystem::remove_all("data");
        std::filesystem::create_directory("data");
        printf("Step 0 / %03d\n", total_steps);
        save(u, mesh_x, mesh_y, 0);

        for (int step = 1; step < total_steps; ++step) {
            Kokkos::deep_copy(u_old, u);
            advance(u, u_old);
            if (step % diag_step == 0) {
                printf("Step %03d / %03d\n", step, total_steps);
                save(u, mesh_x, mesh_y, step);
            }
        }
    }
    Kokkos::finalize();

    return 0;
}
