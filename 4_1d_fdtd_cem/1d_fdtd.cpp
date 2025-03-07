#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <gridformat/gridformat.hpp>

int main(int argc, char const* argv[]) {
    Kokkos::initialize();
    {
        size_t total_step = 200;
        if (argc > 1) {
            INIReader reader(argv[1]);
            total_step = reader.GetInteger("time", "total_step", 200);
            if (reader.ParseError() < 0) {
                std::printf("Can't load {%s} file\n", argv[1]);
                return -1;
            }
        }

        GridFormat::ImageGrid<1, double> grid {
            { 1.0 }, // domain size
            { 200 } // number of cells (pixels) in each direction
        };
        GridFormat::VTKHDFTimeSeriesWriter writer {
            grid,
            "data"
        };

        size_t num_cell = grid.number_of_cells(0);
        Kokkos::printf("Num cells = %d", num_cell);

        Kokkos::View<double*> ez("Ez", num_cell);
        Kokkos::View<double*> hy("Hy", num_cell);
        Kokkos::deep_copy(ez, 0);
        Kokkos::deep_copy(hy, 0);
        double imp0 = 377.0;

        for (int n = 0; n < total_step; n++) {
            /* update magnetic field */
            hy[num_cell - 1] = hy[num_cell - 2];
            for (int i = 0; i < num_cell - 1; i++)
                hy[i] = hy[i] + (ez[i + 1] - ez[i]) / imp0;

            /* update electric field */
            ez[0] = ez[1];
            for (int i = 1; i < num_cell; i++)
                ez[i] = ez[i] + (hy[i] - hy[i - 1]) * imp0;

            /* hardwire a source node */
            // ez[0] = Kokkos::exp(-(n - 30.) * (n - 30.) / 100.);
            ez[50] += Kokkos::exp(-(n - 30.) * (n - 30.) / 100.);

            writer.set_cell_field("ez", [&](const auto cell) {
                return ez(cell.location[0]);
            });
            writer.set_cell_field("hy", [&](const auto cell) {
                return hy(cell.location[0]);
            });
            writer.write(n);
            Kokkos::printf("Step %d / %d\n", n + 1, total_step);
        }
    }
    Kokkos::finalize();
    return 0;
}