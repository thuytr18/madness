#ifndef PLOT_H
#define PLOT_H

#include <madness/mra/mra.h>

using namespace madness;

// for harmonic oscillator and gaussian potential L = 5.0
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

template <typename T, std::size_t NDIM>
void plot_volume(const char* filename, int npt,
                 const Vector<double, NDIM>& topleftfront, const Vector<double, NDIM>& toprightfront, const Vector<double, NDIM>& bottomleftfront, 
                 const Vector<double, NDIM>& topleftback, const Vector<double, NDIM>& toprightback, const Vector<double, NDIM>& bottomleftback, 
                 const Function<T, NDIM>& f) {
    Vector<double, NDIM> h = 1/double(npt-1) * (toprightfront - topleftfront);
    Vector<double, NDIM> v = 1/double(npt-1) * (bottomleftfront - topleftfront);
    Vector<double, NDIM> w = 1/double(npt-1) * (topleftback - topleftfront);

    double step_h = 0.0;
    for (std::size_t i = 0; i < NDIM; ++i) step_h += h[i]*h[i];
    step_h = sqrt(step_h);

    double step_v = 0.0;
    for (std::size_t j = 0; j < NDIM; ++j) step_v += v[j]*v[j];
    step_v = sqrt(step_v);

    double step_w = 0.0;
    for (std::size_t k = 0; k < NDIM; ++k) step_w += w[k]*w[k];
    step_w = sqrt(step_w);

    World& world = f.world();
    f.reconstruct();
    if (world.rank() == 0) {
        FILE* file = fopen(filename, "w");
        if (!file)
            MADNESS_EXCEPTION("plot_volume: failed to open the plot file", 0);
        for (int i = 0; i < npt; ++i) {
            for (int j = 0; j < npt; ++j) {
                for (int k = 0; k < npt; ++k) {
                    Vector<double, NDIM> sample_point = topleftfront + h*double(i) + v*double(j) + w*double(k);
                    fprintf(file, "%.14e ", i*step_h);
                    fprintf(file, "%.14e ", j*step_v);
                    fprintf(file, "%.14e ", k*step_w);
                    plot_line_print_value(file, f.eval(sample_point));
                    fprintf(file, "\n");
                }
            }
        }
        fclose(file);
    }
    world.gop.fence();
}

template <typename T, std::size_t NDIM>
void plot3D(const char* filename, const Function<T, NDIM>& f) {
    Vector<T, NDIM> topleftfront{}, toprightfront{}, bottomleftfront{}, topleftback{}, toprightback{}, bottomleftback{};
    topleftfront[0] = -L;
    topleftfront[1] = -L;
    topleftfront[2] = -L;

    toprightfront[0] = L;
    toprightfront[1] = -L;
    toprightfront[2] = -L;

    bottomleftfront[0] = -L;
    bottomleftfront[1] = L;
    bottomleftfront[2] = -L;

    topleftback[0] = -L;
    topleftback[1] = -L;
    topleftback[2] = L;

    toprightback[0] = L;
    toprightback[1] = -L;
    toprightback[2] = L;

    bottomleftback[0] = -L;
    bottomleftback[1] = L;
    bottomleftback[2] = L;

    plot_volume(filename, 41, topleftfront, toprightfront, bottomleftfront, topleftback, toprightback, bottomleftback, f);
}



//--------------------------------------------------------------------------------------------------------------------//

#endif