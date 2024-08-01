#include <cmath>
#include <cstddef>
#include <iostream>
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
    const double charge = -2.0; // Charge
    //const int num_levels = std::abs((int) charge / 2); // Number of levels
    const int num_levels = 3; // Number of levels
    constexpr int max_iter = 100; // Maximum number of iterations
    constexpr int NDIM = 1; // Dimension

    std::cout << "Number of levels: " << num_levels << std::endl;

    //-------------------------------------------------------------------------------//
    // Create the operator
    SeparatedConvolution<double, NDIM> op = GaussOperator<NDIM>(world, 1.0);

    //-------------------------------------------------------------------------------//
    // Set the defaults

    FunctionDefaults<NDIM>::set_k(6);        
    FunctionDefaults<NDIM>::set_thresh(thresh);
    FunctionDefaults<NDIM>::set_cubic_cell(-L, L);  // 1D cubic cell

    //-------------------------------------------------------------------------------//

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

    double a = -20.0; // Scaling factor

    // Create the gaussian potential
    Function<double, NDIM> rho = gaussian_potential_generator.create_gaussianpotential(a, mu, sigma);
    double charge_rho = rho.trace(); // Calculate the charge of the gaussian potential
    std::cout << "Charge of rho: " << charge_rho << std::endl;

    // Plot the gaussian potential
    if (NDIM == 1)
        plot1D("rho.dat", rho);
    else if (NDIM == 2)
        plot2D("rho2D.dat", rho);

    double b = a * (charge / charge_rho) ;  // Integral of normal guassian function is sqrt(2*pi), to get a charge of 2, multiply by 2 / sqrt(2 * pi)
    // std::cout << "Scaling factor: " << b << std::endl;

    Function<double, NDIM> charged_rho = gaussian_potential_generator.create_gaussianpotential(b, mu, sigma); // Create the gaussian potential with charge 2
    charge_rho = charged_rho.trace(); // Calculate the charge of the charged gaussian potential
    std::cout << "Charge of charged rho: " << charge_rho << std::endl;

    // Plot the gaussian potential
    if (NDIM == 1)
        plot1D("charged_rho.dat", charged_rho);
    else if (NDIM == 2)
        plot2D("charged_rho2D.dat", charged_rho);

    // Create the potential from the charge density
    ChargeDensityPotential<double, NDIM> charge_density_potential(world);

    Function<double, NDIM> V = charge_density_potential.create_potential(charged_rho, charge, op);
    
    // Plot the potential created from the charge density
    if (NDIM == 1)
        plot1D("potential.dat", V);
    else if (NDIM == 2)
        plot2D("potential2D.dat", V);

    // Create Hartree-Fock solver
    HartreeFock<double, NDIM> hartree_fock_solver(world);

    // Solve the Hartree-Fock equation
    std::vector<Function<double, NDIM>> eigenfunctions = hartree_fock_solver.solve(V, num_levels, max_iter, op);

    // Plot the eigenfunctions
    for (int i = 0; i < num_levels; i++) {
        char filename[256];
        snprintf(filename, 256, "Psi_%1d.dat", i);

        if (NDIM == 1)
            plot1D(filename, eigenfunctions[i]); 
        else if (NDIM == 2)
            plot2D(filename, eigenfunctions[i]);
    }

    // Finalizing
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}
