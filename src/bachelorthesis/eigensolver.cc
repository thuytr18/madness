#include <cmath>
#include <cstddef>
#include <iostream>
#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/legendre.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/operator.h>
#include <madness/mra/vmra.h>
#include <madness/mra/derivative.h>
#include <madness/tensor/gentensor.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <madness/world/worldmpi.h>
#include <madness/tensor/tensor.h>
#include <utility>
#include <vector>
#include "potential.h"
#include "guesses.h"
#include "taylorseries.h"
#include "plot.h"
#include "eigensolver.h"


using namespace madness;

int main(int argc, char** argv) {
    // Initializing
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world,argc,argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    //-------------------------------------------------------------------------------//
    const double thresh = 1e-6; // Threshold
    // Number of levels // for harmonic oscillator: 10, for gaussian potential: 5, for double well potential: 4, for exponential potential: 5, for morse potential: 4
    constexpr int num_levels = 4;
    constexpr int max_iter = 100; // Maximum number of iterations
    constexpr int NDIM = 1; // Dimension

    //-------------------------------------------------------------------------------//

    // Set the defaults

    FunctionDefaults<NDIM>::set_k(6);        
    FunctionDefaults<NDIM>::set_thresh(thresh);
    FunctionDefaults<NDIM>::set_cubic_cell(-L, L);  // 1D cubic cell

    //-------------------------------------------------------------------------------//

    // Create the potential generator

    //HarmonicPotentialGenerator<double, NDIM> potential_generator(world);                // Generator for harmonic potential
    //GaussianPotentialGenerator<double, NDIM> gaussian_potential_generator(world);       // Generator for gaussian potential
    //DoubleWellPotentialGenerator<double, NDIM> doublewell_potential_generator(world);   // Generator for double well potential
    ExponentialPotentialGenerator<double, NDIM> exponential_potential_generator(world); // Generator for exponential potential
    //MorsePotentialGenerator<double, NDIM> morse_potential_generator(world);             // Generator for exponential potential

    //-------------------------------------------------------------------------------//

    //Create the potential with potential generator

    //Function<double, NDIM> V = potential_generator.create_harmonicpotential(0.0);

    // Parameters mu and sigma for first gaussian potential
    Vector<double, NDIM> mu{};
    mu.fill(0.0);
    //mu.fill(-1.5);
    Tensor<double> sigma(NDIM, NDIM); // Create a covariance matrix
    for (int i = 0; i < NDIM; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            sigma(i, j) = (i == j) ? 1.0 : 0.0; // Set the diagonal elements to 1
        }
    }

    //Function<double, NDIM> V = gaussian_potential_generator.create_gaussianpotential(-10, mu, sigma);    // Create the gaussian potential

    // Parameters mu1 and sigma1 for second gaussian potential
    Vector<double, NDIM> mu1{};
    mu1.fill(1.5);
    Tensor<double> sigma1(NDIM, NDIM); // Create a covariance matrix
    for (int i = 0; i < NDIM; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            sigma1(i, j) = (i == j) ? 1.0 : 0.0; // Set the diagonal elements to 1
        }
    }
    
    //Function<double, NDIM> V = doublewell_potential_generator.create_doublewellpotential(-5, mu, sigma, -5, mu1, sigma1); // Create the double well potential

    Function<double, NDIM> V = exponential_potential_generator.create_exponentialpotential(-10.0, 0.5); // Create the exponential potential

    //Function<double, NDIM> V = morse_potential_generator.create_morsepotential(5.0, 1.0, 0.75); // Create the morse potential
 

    //-------------------------------------------------------------------------------//

    // Create the harmonic guess generator

    //HarmonicGuessGenerator<double, NDIM> harmonic_guess_generator(world);      // Harmonic guess generator for harmonic and gaussian potential

    //-------------------------------------------------------------------------------//

    // Create the harmonic guesses

    //std::vector<Function<double,NDIM>> guesses = harmonic_guess_generator.create_guesses(num_levels, 1);    // Create the guesses for harmonic potential

    //-------------------------------------------------------------------------------//

    // Plot the potential
    if (NDIM == 1)
        plot1D("potential.dat", V);
    else if (NDIM == 2)
        plot2D("potential2D.dat", V);
    else if (NDIM == 3)
        plot3D("potential3D.dat", V);

    
    //--------------------------------------------------------------------------------//
    // only for harmonic guesses
    /*
    // Plot guesses
    for (int i = 0; i < guesses.size(); i++) {
        char filename[512];
        snprintf(filename, 512, "g-%1d.dat", i);
        if (NDIM == 1)
            plot1D(filename, guesses[i]);
        else if (NDIM == 2)
            plot2D(filename, guesses[i]);
    }
    */
    
    //--------------------------------------------------------------------------------//
    
    // Create Eigensolver
    Eigensolver<double, NDIM> solver(world);

    //-------------------------------------------------------------------------------//

    // Solve the eigenvalue problem for potential V and guesses
    //std::vector<Function<double, NDIM>> y = solver.solve(V, guesses, num_levels, max_iter);  // here only for harmonic potential
    std::vector<Function<double, NDIM>> y = solver.solve(V, num_levels, max_iter); // Solve the eigenvalue problem for potential V 

    // plot the eigenfunctions
    for (int i = 0; i < num_levels; i++) {
        char filename[256];
        snprintf(filename, 256, "Psi_%1d.dat", i);
        double en = solver.energy(world, y[i], V);
        std::cout << "Energy: " << en << std::endl;

        // for hamonic oscillator
        // V(r) = 0.5 * r * r = E
        // inverse function: r = sqrt(2 * E)
        /*
        if(y[i](std::sqrt(2* en)) < 0) {
            y[i] *= -1;
        }
        */
        
        if (NDIM == 1)
        // for harmonic oscillator: 0.75 * y[i] + en for optical reasons 
        // before it was: y[i] + en
            plot1D(filename, y[i] + en); 
        else if (NDIM == 2)
            plot2D(filename, y[i]);
        else if (NDIM == 3)
            plot3D(filename, y[i]);
    }


    //----------Only for comparison--------------------------------------------------------------------------------------------------------//
    
    // calculate the Taylor series of potential
    /*
    TaylorSeriesGenerator<double, NDIM> taylor_series_generator(world);
    //Vector<double, NDIM> x0(0.0);
    Vector<double, NDIM> x0(-0.75);
    std::pair<Function<double, NDIM>, std::vector<double>> taylor_series = taylor_series_generator.create_taylorseries(world, V, x0, 2);
    Function<double, NDIM> approximation = taylor_series.first;
    std::vector<double> quadratic_coefficients = taylor_series.second;

    std::cout << "Size of quadratic coefficients: " << quadratic_coefficients.size() << std::endl;

    if (NDIM == 1)
        plot1D("approximation.dat", approximation);
    else if (NDIM == 2)
        plot2D("approximation2D.dat", approximation);

    std::cout << "Taylor series created" << std::endl;

    Vector<double, NDIM> x1(0.75);
    std::pair<Function<double, NDIM>, std::vector<double>> taylor_series1 = taylor_series_generator.create_taylorseries(world, V, x1, 2);
    Function<double, NDIM> approximation1 = taylor_series1.first;
    std::vector<double> quadratic_coefficients1 = taylor_series1.second;

    for(double quadratic_coefficient : quadratic_coefficients) {
        std::cout << "Quadratic coefficient: " << quadratic_coefficient << std::endl;
    }

    if (NDIM == 1)
        plot1D("approximation1.dat", approximation1);
    else if (NDIM == 2)
        plot2D("approximation2D1.dat", approximation1);

    std::cout << "Taylor series created" << std::endl;
    */
    //--------------------------------------------------------------------------------//
    
    // create Approximation of potential
    /*
    PotentialGenerator<double, NDIM> potential_generator(world);
    //Function<double, NDIM> approximation = potential_generator.create_potential(6.75, 0.6);
    //Function<double, NDIM> approximation = potential_generator.create_potential(7, 1);
    Function<double,NDIM> approximation = potential_generator.create_potential(0.85, 5.6);

    if (NDIM == 1)
        plot1D("approximation.dat", approximation);
    else if (NDIM == 2)
        plot2D("approximation2D.dat", approximation);
    
    */
    //--------------------------------------------------------------------------------//
    /*
    // create the guesses for the taylor series
    HarmonicGuessGenerator<double, NDIM> harmonic_guess_generator(world);
    // quadratic_coefficients[0] is the coefficient of the quadratic term of the taylor series
    //std::vector<Function<double,NDIM>> approximation_guesses = harmonic_guess_generator.create_guesses(2*num_levels, quadratic_coefficients[0]);
    std::vector<Function<double, NDIM>> approximation_guesses = harmonic_guess_generator.create_guesses(num_levels, 0.85);
    
    // Plot guesses
    for (int i = 0; i < approximation_guesses.size(); i++) {
        char filename[512];
        snprintf(filename, 512, "ga-%1d.dat", i);
        if (NDIM == 1)
            plot1D(filename, approximation_guesses[i]);
        else if (NDIM == 2)
            plot2D(filename, approximation_guesses[i]);
    }
    
    
    std::vector<Function<double, NDIM>> approximation_y = solver.solve(V, approximation_guesses, static_cast<int>(approximation_guesses.size()), max_iter); // Solve the eigenvalue problem for potential V

    for(int i = 0; i < static_cast<int>(approximation_guesses.size()); i++) {
        char filename[256];
        snprintf(filename, 256, "PsiApprox_%1d.dat", i);
        double en = solver.energy(world, approximation_y[i], V);
        std::cout << "Energy: " << en << std::endl;

        if (NDIM == 1)
            plot1D(filename, approximation_y[i] + en);
        else if (NDIM == 2)
            plot2D(filename, approximation_y[i]);
    }
    */

    //--------------------------------------------------------------------------------------------------------------------------------------//

    // Finalizing
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}
