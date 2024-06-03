#ifndef PLOT_H
#define PLOT_H

#include <madness/mra/mra.h>

using namespace madness;

const double L = 10.0;  // Length of the 1D cubic cell

// Convenience routine for plotting
template <typename T, std::size_t NDIM>
void plot1D(const char* filename, const Function<T, NDIM>& f) {
    Vector<T, NDIM> lo{}, hi{};
    for (std::size_t i = 0; i < NDIM; ++i) {
        lo[i] = -L;
        hi[i] = L;
    }
    plot_line(filename, 401, lo, hi, f);
}

/// The ordinate is distance from lo
template <typename T, std::size_t NDIM>
void plot_area(const char* filename, int npt, const Vector<double,NDIM>& topleft, const Vector<double,NDIM>& topright, const Vector<double, NDIM> bottomleft, 
                const Function<T,NDIM>& f) {
    Vector<double,NDIM> h = 1/double(npt-1) * (topright - topleft);
    Vector<double, NDIM> v = 1/double(npt-1) * (bottomleft - topleft);

    double step_h = 0.0;
    for (std::size_t i=0; i<NDIM; ++i) step_h += h[i]*h[i];
    step_h = sqrt(step_h);

    double step_v = 0.0;
    for (std::size_t j=0; j<NDIM; ++j) step_v += v[j]*v[j];
    step_v = sqrt(step_v);

    World& world = f.world();
    f.reconstruct();
    if (world.rank() == 0) {
        FILE* file = fopen(filename,"w");
    if(!file)
        MADNESS_EXCEPTION("plot_line: failed to open the plot file", 0);
        for (int i = 0; i < npt; ++i) {
            for (int j = 0; j < npt; ++j) {
                Vector<double,NDIM> sample_point = topleft + h*double(i) + v*double(j);
                fprintf(file, "%.14e ", i*step_h);
                fprintf(file, "%.14e ", j*step_v);
                plot_line_print_value(file, f.eval(sample_point));
                fprintf(file,"\n");
            }
        }
        fclose(file);
    }
    world.gop.fence();
}

template <typename T, std::size_t NDIM>
void plot2D(const char* filename, const Function<T, NDIM>& f) {
    Vector<T, NDIM> topleft{}, topright{}, bottomleft{};
    topleft[0] = -L;
    topleft[1] = -L;
    topright[0] = L;
    topright[1] = -L;
    bottomleft[0] = -L;
    bottomleft[1] = L;
    plot_area(filename, 401, topleft, topright, bottomleft, f);
}

//--------------------------------------------------------------------------------------------------------------------//

#endif