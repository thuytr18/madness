#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <madness/constants.h>
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
#include <ostream>
#include <utility>
#include <vector>
#include "hartreefock.h"
#include "potential.h"
#include "guesses.h"
#include "plot.h"
#include "chargeDensityPotential.h"


using namespace madness;

int main(int argc, char** argv) {
    // Initializing
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world,argc,argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    //-------------------------------------------------------------------------------//
    const double thresh = 1e-6; // Threshold
    constexpr int max_iter = 5; // Maximum number of iterations
    constexpr int num_levels = 1; // Number of levels
    constexpr int NDIM = 3; // Dimension

    //-------------------------------------------------------------------------------//
    // Create the operator
    SeparatedConvolution<double, NDIM> op = CoulombOperator(world, 0.001, 1e-4);
    //-------------------------------------------------------------------------------//
    // Set the defaults

    FunctionDefaults<NDIM>::set_k(6);        
    FunctionDefaults<NDIM>::set_thresh(thresh);
    FunctionDefaults<NDIM>::set_cubic_cell(-L, L);  // 1D cubic cell

    //-------------------------------------------------------------------------------//

    // Generator for Hdyrogen atom
    NuclearPotentialGenerator<double, NDIM> hyperbolic_potential_generator(world);    
    Vector<double, NDIM> R{};
    R.fill(0.0);

    //Function<double, NDIM> V = hyperbolic_potential_generator.create_hyperbolicpotential(2, R);
    Function<double, NDIM> V = hyperbolic_potential_generator.create_nuclearpotential(4, R);

    plot3D("potential.dat", V);

    // Generator for initial guess
    std::vector<Function<double, NDIM>> guesses;
    // Generator for gaussian potential
    GaussianPotentialGenerator<double, NDIM> gaussian_potential_generator(world);      
     
    // Parameters mu and sigma for first gaussian potential
    Vector<double, NDIM> mu{};
    mu.fill(0.0);
    Tensor<double> sigma(NDIM, NDIM); // Create a covariance matrix
    for (int i = 0; i < NDIM; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            sigma(i, j) = (i == j) ? 1.0 : 0.0; // Set the diagonal elements to 1
        }
    }
    double a = 1;

    Function<double, NDIM> initial_guess = gaussian_potential_generator.create_gaussianpotential(a, mu, sigma);
    plot3D("guess.dat", initial_guess);
    guesses.push_back(initial_guess);

    Tensor<double> sigma2(NDIM, NDIM); // Create a covariance matrix
    for (int i = 0; i < NDIM; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            sigma2(i, j) = (i == j) ? 5.0 : 0.0; // Set the diagonal elements to 1
        }
    }

    Function<double, NDIM> initial_guess2 = gaussian_potential_generator.create_gaussianpotential(a, mu, sigma2);
    guesses.push_back(initial_guess2);



    //-------------------------------------------------------------------------------//
    // Hartree-Fock
    HartreeFock<double, NDIM> hartree_fock_solver(world);

    // Solve the Hartree-Fock equation
    std::cout << "Hartree Fock with given initial guess" << std::endl;
    std::vector<Function<double, NDIM>> eigenfunctions = hartree_fock_solver.solve(V, guesses, max_iter, op);

    // std::cout << "Hartree Fock with guesses depending on potential" << std::endl;
    // std::vector<Function<double, NDIM>> eigenfunctions_1 = hartree_fock_solver.solve(V, num_levels, max_iter, op);

    for (int i = 0; i < eigenfunctions.size(); i++) {
        char filename[512];
        snprintf(filename, 512, "Psi_%1d.dat", i);
        plot3D(filename, eigenfunctions[i]);
    }

    // Finalizing
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}

