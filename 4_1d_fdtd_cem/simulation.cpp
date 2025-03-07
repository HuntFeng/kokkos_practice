#include <INIReader.h>

#include <Kokkos_Core.hpp>
#include <filesystem>
#include <gridformat/gridformat.hpp>

using Kokkos::View, Kokkos::deep_copy;

int read_input(
    std::array<double, 2>& x1_range,
    std::array<double, 2>& x2_range,
    std::array<int, 2>& n
) {
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

// webkit 4 space tab, &* stick to type, {} of function is in newline

void write_data(View<double*> field, int step) {
    std::vector<double> host_data(field.extent(0));
    Kokkos::deep_copy(
        Kokkos::View<double*, Kokkos::HostSpace>(
            host_data.data(), field.extent(0)
        ),
        field
    );
}

int main(int argc, char const* argv[]) {
    std::filesystem::remove_all("data");
    std::filesystem::create_directory("data");

    GridFormat::ImageGrid<2, double> grid {
        { 1.0, 1.0 }, // domain size
        { 10, 12 } // number of cells (pixels) in each direction
    };

    GridFormat::VTKHDFImageGridTimeSeriesWriter writer { grid };
    // GridFormat::Test::write_test_time_series<2>(writer);
    // GridFormat::Writer writer { GridFormat::time_series, grid };
    // GridFormat::Writer writer { GridFormat::vtk_hdf, grid };
    // GridFormat::TimeSeriesGridWriter writer { GridFormat::time_series, grid };
    //
    return 0;
}
