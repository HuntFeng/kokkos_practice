/**
 * 1D FDTD simulation of a lossless dielectric region
 * followed by a lossy layer which matches the impedance
 * of the dielectric.
 */
#include <INIReader.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <gridformat/gridformat.hpp>

int main(int argc, char const* argv[]) {
    Kokkos::initialize();
    {
        size_t total_step = 200;
        double loss = 0.0;
        if (argc > 1) {
            INIReader reader(argv[1]);
            if (reader.ParseError() < 0) {
                std::printf("Can't load {%s} file\n", argv[1]);
                return -1;
            }

            total_step = reader.GetInteger("time", "total_step", total_step);
            loss = reader.GetReal("constant", "loss", loss);
        }

        GridFormat::ImageGrid<1, double> grid{
            {1.0}, // domain size
            {200}  // number of cells (pixels) in each direction
        };
        GridFormat::VTKHDFTimeSeriesWriter writer{grid, "data"};

        size_t ncell = grid.number_of_cells(0);
        Kokkos::printf("Num cells = %d\n", ncell);
        Kokkos::printf("Loss = %f\n", loss);

        Kokkos::View<double*> ez("Ez", ncell);
        Kokkos::View<double*> hy("Hy", ncell - 1);
        Kokkos::View<double*> ceze("ceze", ncell);
        Kokkos::View<double*> cezh("cezh", ncell);
        Kokkos::View<double*> chye("chye", ncell - 1);
        Kokkos::View<double*> chyh("chyh", ncell - 1);
        Kokkos::deep_copy(ez, 0);
        Kokkos::deep_copy(hy, 0);
        double imp0 = 377.0;

        // electric field update coefficients
        for (int i = 0; i < ncell; ++i) {
            if (i < 100) {
                ceze[i] = 1.0;
                cezh[i] = imp0;
            } else if (i < 180) {
                ceze[i] = 1.0;
                cezh[i] = imp0 / 9.0;
            } else {
                ceze[i] = (1.0 - loss) / (1.0 + loss);
                cezh[i] = imp0 / 9.0 / (1.0 + loss);
            }
        }

        // magnetic field update coefficients
        for (int i = 0; i < ncell - 1; ++i) {
            if (i < 180) {
                chyh[i] = 1.0;
                chye[i] = 1.0 / imp0;
            } else {
                chyh[i] = (1.0 - loss) / (1.0 + loss);
                chye[i] = 1.0 / imp0 / (1.0 + loss);
            }
        }

        // abc coefficients
        double temp = Kokkos::sqrt(cezh[0] * chye[0]);
        double abc_coef_left = (temp - 1.0) / (temp + 1.0);

        temp = Kokkos::sqrt(cezh[ncell - 1] * chye[ncell - 2]);
        double abc_coef_right = (temp - 1.0) / (temp + 1.0);

        double ez_old_left = 0.0, ez_old_right = 0.0;

        for (int n = 0; n < total_step; n++) {
            // update magnetic field
            for (int i = 0; i < ncell - 1; ++i)
                hy[i] = chyh[i] * hy[i] + chye[i] * (ez[i + 1] - ez[i]);

            // correction for Hy adjacent to TFSF boundary
            hy[49] -= exp(-(n - 30.) * (n - 30.) / 100.) / imp0;

            for (int i = 1; i < ncell - 1; ++i)
                ez[i] = ceze[i] * ez[i] + cezh[i] * (hy[i] - hy[i - 1]);

            // correction for Ez adjacent to TFSF boundary
            ez[50] += exp(-(n + 0.5 - (-0.5) - 30.) * (n + 0.5 - (-0.5) - 30.) /
                          100.);

            // abc on the left and right
            ez[0] = ez_old_left + abc_coef_left * (ez[1] - ez[0]);
            ez_old_left = ez[1];
            ez[ncell - 1] =
                ez_old_right + abc_coef_right * (ez[ncell - 2] - ez[ncell - 1]);
            ez_old_right = ez[ncell - 2];

            writer.set_cell_field(
                "ez", [&](const auto cell) { return ez(cell.location[0]); });
            writer.set_cell_field(
                "hy", [&](const auto cell) { return hy(cell.location[0]); });
            writer.write(n);
            Kokkos::printf("Step %d / %d\n", n + 1, total_step);
        }
    }
    Kokkos::finalize();
    return 0;
}
