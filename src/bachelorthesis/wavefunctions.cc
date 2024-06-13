#include <cmath>
#include <cstddef>
#include <iostream>
#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/legendre.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
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


using namespace madness;

// Function to calculate the energy
template <typename T, std::size_t NDIM>
double energy(World& world, const Function<T, NDIM>& phi, const Function<T, NDIM>& V) {
    double potential_energy = inner(phi, V * phi); // <phi|Vphi> = <phi|V|phi>
    double kinetic_energy = 0.0;

    Derivative<T, NDIM> D = free_space_derivative<T, NDIM>(world, 0); // Derivative operator

    Function<T, NDIM> dphi = D(phi);
    kinetic_energy += 0.5 * inner(dphi, dphi);  // (1/2) <dphi/dx | dphi/dx>

    double energy = kinetic_energy + potential_energy;
    return energy;
}


// Function to calculate the Hamiltonian matrix, Overlap matrix and Diagonal matrix
template <typename T, std::size_t NDIM>
std::pair<Tensor<T>, std::vector<Function<T, NDIM>>> diagonalize(World &world, const std::vector<Function<T, NDIM>>& functions, const Function<T, NDIM>& V){
    const int num = functions.size();
    
    auto H = Tensor<T>(num, num); // Hamiltonian matrix
    auto overlap = Tensor<T>(num, num); // Overlap matrix
    auto diag_matrix = Tensor<T>(num, num); // Diagonal matrix

    // Calculate the Hamiltonian matrix
    
    for(int i = 0; i < num; i++) {
        auto energy1 = energy(world, functions[i], V);
        std::cout << energy1 << std::endl;
        for(int j = 0; j < num; j++) {
            double kin_energy = 0.0;
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<T, NDIM> D = free_space_derivative<T,NDIM>(world, axis); // Derivative operator

                Function<T, NDIM> dx_i = D(functions[i]);
                Function<T, NDIM> dx_j = D(functions[j]);

                kin_energy += 0.5 * inner(dx_i, dx_j);  // (1/2) <dphi/dx | dphi/dx>
            }
        
            double pot_energy = inner(functions[i], V * functions[j]); // <phi|V|phi>

            H(i, j) = kin_energy + pot_energy; // Hamiltonian matrix
        }
    }
    std::cout << "H: \n" << H << std::endl;

    // Calculate the Overlap matrix
    overlap = matrix_inner(world, functions, functions);

    // Calculate the Diagonal matrix
    Tensor<T> U;
    Tensor<T> evals;
    // sygvp is a function to solve the generalized eigenvalue problem HU = SUW where S is the overlap matrix and W is the diagonal matrix of eigenvalues of H
    // The eigenvalues are stored in evals and the eigenvectors are stored in U
    sygvp(world, H, overlap, 1, U, evals);
    
    diag_matrix.fill(0.0);
    for(int i = 0; i < num; i++) {
        diag_matrix(i, i) = evals(i); // Set the diagonal elements
    }

    std::cout << "dia_matrix: \n" << diag_matrix << std::endl;

    std::vector<Function<T, NDIM>> y;

    // y = U * functions
    y = transform(world, functions, U);
    
    // std::cout << "U matrix: \n" << U << std::endl;
    // std::cout << "evals: \n" << evals << std::endl;
    return std::make_pair(evals, y);
}


// Function to generate and solve for each energy level
template <typename T, std::size_t NDIM>
Function<T, NDIM> generate_and_solve(World& world, Function<T, NDIM>& V, const Function<T, NDIM> guess_function, int N, const std::vector<Function<T, NDIM>>& prev_phi) {

    // Create the initial guess wave function
    Function<T, NDIM> phi = guess_function;

    phi.scale(1.0/phi.norm2()); // phi *= 1.0/norm
    double E = energy(world, phi, V);

    NonlinearSolverND<NDIM> solver;
    int count_shift = 0; // counter how often the potential was shifted

    for(int iter = 0; iter <= 50; iter++) {
        //char filename[256];
        //snprintf(filename, 256, "phi-%1d.dat", N);
        //plot1D(filename,phi);
        
        // Energy cant be positiv
        // shift potential

        if (E > 0) {
            V = V - DELTA;
            E = energy(world, phi, V);
            count_shift++;
        }

        Function<T, NDIM> Vphi = V*phi;
        Vphi.truncate();
        
        SeparatedConvolution<T,NDIM> op = BSHOperator<NDIM>(world, sqrt(-2*E), 0.01, 1e-7);  

        Function<T, NDIM> r = phi + 2.0 * op(Vphi); // the residual
        T err = r.norm2();

        // Q = 1 - sum(|phi_prev><phi_prev|) = 1 - |phi_0><phi_0| - |phi_1><phi_1| - ...
        // Q*|Phi> = |Phi> - |phi_0><phi_0|Phi> - |phi_1><phi_1|Phi> - ...

        for (const auto& prev_phi : prev_phi) {
            phi -= inner(prev_phi, phi)*prev_phi; 
        }
        phi.scale(1.0/phi.norm2());

        phi = solver.update(phi, r);

        double norm = phi.norm2();
        phi.scale(1.0/norm);  // phi *= 1.0/norm
        E = energy(world,phi,V); 

        if (world.rank() == 0)
            print("iteration", iter, "energy", E, "norm", norm, "error",err);

        if (err < 5e-4) break;
    }

    char filename[256];
    snprintf(filename, 256, "phi-%1d.dat", N);

    if (NDIM == 1)
        plot1D(filename, phi);
    else if (NDIM == 2)
        plot2D(filename,phi);

    if (count_shift != 0) {
        std::cout << "Potential was shifted " << count_shift << " times" << std::endl;
        V = V + count_shift * DELTA;
    }

    print("Final energy without shift: ", E + DELTA);
    return phi;
}


int main(int argc, char** argv) {
    // Initializing
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world,argc,argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    //-------------------------------------------------------------------------------//

    const double thresh = 1e-6; // Threshold
    // Number of levels // for harmonic oscillator: 10, for gaussian potential: 5, for double well potential: 4, for exponential potential: 5
    constexpr int num_levels = 5;  
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

    //-------------------------------------------------------------------------------//

    //Create the potential with potential generator

    //Function<double, NDIM> V = potential_generator.create_harmonicpotential(DELTA);

    // Parameters mu and sigma for first gaussian potential
    Vector<double, NDIM> mu{};
    //mu.fill(0.0);
    mu.fill(-1.5);
    Tensor<double> sigma(NDIM, NDIM); // Create a covariance matrix
    for (int i = 0; i < NDIM; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            sigma(i, j) = (i == j) ? 1.0 : 0.0; // Set the diagonal elements to 1
        }
    }

    //Function<double, NDIM> V = gaussian_potential_generator.create_gaussianpotential(DELTA, 10, mu, sigma);    // Create the gaussian potential

    // Parameters mu1 and sigma1 for second gaussian potential
    Vector<double, NDIM> mu1{};
    mu1.fill(1.5);
    Tensor<double> sigma1(NDIM, NDIM); // Create a covariance matrix
    for (int i = 0; i < NDIM; ++i) {
        for (int j = 0; j < NDIM; ++j) {
            sigma1(i, j) = (i == j) ? 1.0 : 0.0; // Set the diagonal elements to 1
        }
    }
    
    //Function<double, NDIM> V = doublewell_potential_generator.create_doublewellpotential(DELTA, 1, mu, sigma, 1, mu1, sigma1); // Create the double well potential
    Function<double, NDIM> V = exponential_potential_generator.create_exponentialpotential(DELTA, 10.0); // Create the exponential potential
 

    //-------------------------------------------------------------------------------//

    // Create the guess generator 

    //HarmonicGuessGenerator<double, NDIM> harmonic_guess_generator(world);      // Harmonic guess generator for harmonic and gaussian potential
    GuessGenerator<double, NDIM> guess_generator(world);              // Guess generator for all potentials

    //-------------------------------------------------------------------------------//

    // Create the guesses

    //std::vector<Function<double,NDIM>> guesses = harmonic_guess_generator.create_guesses(num_levels);    // Create the guesses for harmonic potential
    std::vector<Function<double,NDIM>> guesses = guess_generator.create_guesses(num_levels, V);   // Create the guesses for all potential

    //-------------------------------------------------------------------------------//

    // Plot the potential
    if (NDIM == 1)
        plot1D("potential.dat", V);
    else if (NDIM == 2)
        plot2D("potential2D.dat", V);

    // Plot guesses
    for (int i = 0; i < guesses.size(); i++) {
        char filename[512];
        snprintf(filename, 512, "g-%1d.dat", i);
        if (NDIM == 1)
            plot1D(filename, guesses[i]);
        else if (NDIM == 2)
            plot2D(filename, guesses[i]);
    }

    //--------------------------------------------------------------------------------//
    
    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> tmp = diagonalize(world, guesses, V);
    std::vector<Function<double, NDIM>> diagonalized_guesses = tmp.second;
    
    // Solve for each energy level and store the eigenfunctions
    std::vector<Function<double, NDIM>> eigenfunctions;

    for (int i = 0; i < num_levels; i++) {
        Function<double, NDIM> phi = generate_and_solve(world, V, diagonalized_guesses[i], i, eigenfunctions);
        eigenfunctions.push_back(phi);
    }

    std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> diagonalized = diagonalize(world, eigenfunctions, V);
    std::cout << "Diagonalize" << std::endl;
    std::vector<Function<double, NDIM>> y = diagonalized.second;
    //auto evals = tmp.first;

    for (int i = 0; i < num_levels; i++) {
        char filename[256];
        snprintf(filename, 256, "Psi_%1d.dat", i);
        double en = energy(world, y[i], V);
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
        // before it was: y[i] + DELTA + en
            plot1D(filename, y[i] + en); 
        else if (NDIM == 2)
            plot2D(filename, y[i]);
    }

    //-------------------------------------------------------------------------------//
    /*
    // calculate the Taylor series of the potential

    TaylorSeriesGenerator<double, NDIM> taylor_series_generator(world);
    Vector<double, NDIM> x0(0.0);
    Function<double, NDIM> taylor_series = taylor_series_generator.create_taylorseries(world, V, x0, 2);

    if (NDIM == 1)
        plot1D("taylor_series.dat", taylor_series);
    else if (NDIM == 2)
        plot2D("taylor_series2D.dat", taylor_series);

    std::cout << "Taylor series created" << std::endl;
    */
    //--------------------------------------------------------------------------------//

    // Finalizing
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}