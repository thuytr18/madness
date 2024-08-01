#ifndef HARTREEFOCK_H
#define HARTREEFOCK_H

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/nonlinsol.h>
#include <madness/mra/operator.h>
#include <madness/mra/vmra.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/tensor_lapack.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <ostream>
#include <vector>
#include "guesses.h"
#include "plot.h"

using namespace madness;

// Class to do the Hartree-Fock Approximation
// Input: World, Function (potential), Vector of Function (Guesses), max iterations, number of guesses

template <typename T, int NDIM>
class HartreeFock {
    public:
        explicit HartreeFock(World& world) : world(world) {}

        // Function to solve the Hartree-Fock equation for a given potential
        std::vector<Function<T, NDIM>> solve(Function<T, NDIM>& V, int num_levels, int max_iter, SeparatedConvolution<T, NDIM>& twoBody_op) {
            // Create the guess generator
            GuessGenerator<double, NDIM> guess_generator(world);              // Guess generator for all potentials
            // Create the guess functions
            std::vector<Function<double,NDIM>> guesses = guess_generator.create_guesses(num_levels, V);

            // plot guess functions
            for (int i = 0; i < guesses.size(); i++) {
                char filename[512];
                snprintf(filename, 512, "g-%1d.dat", i);
                if (NDIM == 1)
                    plot1D(filename, guesses[i]);
                else if (NDIM == 2)
                    plot2D(filename, guesses[i]);
            }

            // store the eigenfunctions in vector eigenfunctions
            std::vector<Function<double, NDIM>> eigenfunctions;
            std::cout << "Start optimization" << std::endl;
            eigenfunctions = optimize(world, V, guesses, max_iter, twoBody_op);

            return eigenfunctions;

        }

        std::vector<Function<T, NDIM>> solve(Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& guesses, int num_levels, int max_iter, SeparatedConvolution<T, NDIM>& twoBody_op) {
            // store the eigenfunctions in vector eigenfunctions
            std::vector<Function<T, NDIM>> eigenfunctions;
            std::cout << "Start optimization" << std::endl;
            eigenfunctions = optimize(world, V, guesses, max_iter, twoBody_op);

            return eigenfunctions;
        }

        // Function to calculate the energy
        double energy(World& world, const Function<T, NDIM>& phi, const Function<T, NDIM>& V) {
            double potential_energy = inner(phi, V * phi); // <phi|Vphi> = <phi|V|phi>
            double kinetic_energy = 0.0;

            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<T, NDIM> D = free_space_derivative<T, NDIM>(world, axis); // Derivative operator

                Function<T, NDIM> dphi = D(phi);
                kinetic_energy += 0.5 * inner(dphi, dphi);  // (1/2) <dphi/dx | dphi/dx>
            }

            double energy = kinetic_energy + potential_energy;
            return energy;
        }

    private:
        World& world;
        
        Function<T, NDIM> calculate_OneBodyHamiltonian(World& world, const Function<T, NDIM>& V, const Function<T, NDIM>& phi) {
            Function<T, NDIM> kin_energy = FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<T, NDIM> D = free_space_derivative<T,NDIM>(world, axis); // Derivative operator

                Function<T, NDIM> d2_x2 = D(D(phi));

                kin_energy -= 0.5 * d2_x2;
            }
        
            Function<T, NDIM> pot_energy = V * phi;
            
            return kin_energy + pot_energy;
        }

        Function<T, NDIM> calculate_OneBodyPotential(World& world, const Function<T, NDIM>& V, const Function<T, NDIM>& phi) {
            return V * phi;
        }

        Function<T, NDIM> calculate_TwoBody(World& world, const std::vector<Function<T, NDIM>>& functions, SeparatedConvolution<T, NDIM>& twoBody_op) {
            // calculate the Coulomb Operator
            const int num = functions.size();
            Function<T, NDIM> sum = FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );

            for (int i = 0; i < num; i++) {
                sum += abs(functions[i]) * abs(functions[i]); // sum of the square of the functions |phi_i|^2
            }

            //SeparatedConvolution<T, NDIM> twoBody_op = GaussOperator<NDIM>(world, 1.0);
            Function<T, NDIM> twoBody = twoBody_op(sum);

            return twoBody;
        }


        Function<T, NDIM> calculate_Exchange(World& world, const std::vector<Function<T, NDIM>>& functions, const Function<T, NDIM>& phi_k, SeparatedConvolution<T, NDIM>& twoBody_op) {
            // calculate the exchange operator
            const int num = functions.size();
            Function<T, NDIM> exchange = FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );

            for (int l = 0; l < num; l++) {
                Function<T, NDIM> sum = phi_k * functions[l];           // phi_k * phi_l

                //SeparatedConvolution<T, NDIM> twoBody_op = GaussOperator<NDIM>(world, 1.0);
                Function<T, NDIM> twoBody = twoBody_op(sum);

                exchange += twoBody;
            }

            return exchange;
        }

        Function<T, NDIM> calculate_Fock(World& world, const Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& functions, const Function<T, NDIM>& phi_k, SeparatedConvolution<T, NDIM>& twoBody_op) {
            // calculate the Fock operator
            Function<T, NDIM> h = calculate_OneBodyHamiltonian(world, V, phi_k);
            
            Function<T, NDIM> twoBody = calculate_TwoBody(world, functions, twoBody_op);
            Function<T, NDIM> J = twoBody * phi_k;  // TwoBody operator operating on phi_k

            Function<T, NDIM> exchange = calculate_Exchange(world, functions, phi_k, twoBody_op); 

            return h + 2.0 * J - exchange;
        }

        Function<T, NDIM> calculate_HFPotential(World& world, const Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& functions, const Function<T, NDIM>& phi_k, SeparatedConvolution<T, NDIM>& twoBody_op) {
            // calculate the Fock operator
            Function<T, NDIM> h = calculate_OneBodyPotential(world, V, phi_k);
            
            Function<T, NDIM> twoBody = calculate_TwoBody(world, functions, twoBody_op);
            Function<T, NDIM> J = twoBody * phi_k;  // TwoBody operator operating on phi_k

            Function<T, NDIM> exchange = calculate_Exchange(world, functions, phi_k, twoBody_op); 

            return h + 2.0 * J - exchange;
        }

        std::vector<Function<T, NDIM>> optimize(World& world, Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& guess_functions, int max_iter, SeparatedConvolution<T, NDIM>& twoBody_op) {
            // optimize the guess function
            std::vector<Function<T, NDIM>> fock_functions;;

            NonlinearSolverND<NDIM> solver; // Nonlinear solver

            for (int i = 0; i < guess_functions.size(); i++) {
                    // calculate the Fock operator
                    std::cout << "Calculate Fock " << i << std::endl;
                    Function<T, NDIM> fock = calculate_Fock(world, V, guess_functions, guess_functions[i], twoBody_op);
                    fock_functions.push_back(fock);
            }

            for (int iter = 0; iter < max_iter; iter++) {
                std::vector<Function<T, NDIM>> eigenfunctions = fock_functions;
                
                for (int i = 0; i < eigenfunctions.size(); i++) {
                    // plot the guess functions
                    char filename[256];
                    snprintf(filename, 256, "phi-%1d-%1d.dat", i, iter);
                    plot1D(filename, eigenfunctions[i]);
                }

                std::cout << "Size of eigenfunctions: " << eigenfunctions.size() << std::endl;
                std::cout << "Size of fock_functions: " << fock_functions.size() << std::endl;
                
                // Optimize every guessfunction with HF potential operator
                for (int i = 0; i < eigenfunctions.size(); i++) {
                    std::cout << "Calculate HF " << i << std::endl;
                    Function<T, NDIM> HF = calculate_HFPotential(world, V, eigenfunctions, eigenfunctions[i], twoBody_op);

                    SeparatedConvolution<T,NDIM> op = BSHOperator<NDIM>(world, 1.0, 0.001, 1e-7);   
                    
                    // fock_functions[i] = - 2.0 * op(HF); 
                    Function<T, NDIM> r = eigenfunctions[i] + 2.0 * op(HF); // the residual
                    std::cout << "Calculate residual " << i << std::endl;
                    double err = r.norm2();

                    // Orthogonalize fock_functions[i] to all previous fock_functions
                    // for (const auto& phi : eigenfunctions) {
                    //     eigenfunctions[i] -= inner(phi, eigenfunctions[i])*phi; 
                    // }
                    for (int j = 0; j < i; j++) {
                        eigenfunctions[i] -= inner(eigenfunctions[j], eigenfunctions[i]) * eigenfunctions[j];
                    }

                    eigenfunctions[i].scale(1.0/eigenfunctions[i].norm2());
                    std::cout << "Orthogonalize " << i << std::endl;
                    eigenfunctions[i] = solver.update(eigenfunctions[i], r);
                    std::cout << "Update " << i << std::endl;
                }             

                std::vector<Function<T, NDIM>> fock_holder;
                for (int i = 0; i < eigenfunctions.size(); i++) {
                    // calculate the Fock operator
                    Function<T, NDIM> fock = calculate_Fock(world, V, eigenfunctions, eigenfunctions[i], twoBody_op);
                    fock_holder.push_back(fock);
                }
                eigenfunctions = fock_holder;

                std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> diagonalized = diagonalize(world, eigenfunctions, V);
                std::cout << "Diagonalize" << std::endl;
                std::vector<Function<T, NDIM>> y = diagonalized.second;

                std::cout << "size y " << y.size() << std::endl;
                std::cout << "size fock_functions " << fock_functions.size() << std::endl;

                std::vector<double> err(y.size());
                for (size_t i = 0; i < y.size(); i++) {
                    err[i] = std::abs(y[i].norm2() - fock_functions[i].norm2());
                }

                double error = *std::max_element(err.begin(), err.end());
                
                fock_functions = y;

                if (world.rank() == 0)
                    print("iteration", iter, "error", error);

                if (error < 1e-6) break;
            }
                
            return fock_functions;
        }
         
         // Function to calculate the Hamiltonian matrix, Overlap matrix and Diagonal matrix
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
};

#endif 