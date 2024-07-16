#ifndef HARTREEFOCK_H
#define HARTREEFOCK_H

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/nonlinsol.h>
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
        std::vector<Function<T, NDIM>> solve(Function<T, NDIM>& V, int num_levels, int max_iter) {
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

            for (int i = 0; i < num_levels; i++) {
                Function<double, NDIM> phi = optimize(world, V, guesses[i], i, eigenfunctions, max_iter);
                eigenfunctions.push_back(phi);
            }

            return eigenfunctions;

        }

        std::vector<Function<T, NDIM>> solve(Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& guesses, int num_levels, int max_iter) {
            // store the eigenfunctions in vector eigenfunctions
            std::vector<Function<T, NDIM>> eigenfunctions;

            for (int i = 0; i < num_levels; i++) {
                Function<T, NDIM> phi = optimize(world, V, guesses[i], i, eigenfunctions, max_iter);
                eigenfunctions.push_back(phi);
            }

            return eigenfunctions;
        }

        // Function to calculate the energy
        double energy(World& world, const Function<T, NDIM>& phi, const Function<T, NDIM>& V) {
            double potential_energy = inner(phi, V * phi); // <phi|Vphi> = <phi|V|phi>
            double kinetic_energy = 0.0;

            Derivative<T, NDIM> D = free_space_derivative<T, NDIM>(world, 0); // Derivative operator

            Function<T, NDIM> dphi = D(phi);
            kinetic_energy += 0.5 * inner(dphi, dphi);  // (1/2) <dphi/dx | dphi/dx>

            double energy = kinetic_energy + potential_energy;
            return energy;
        }


    private:
        World& world;
        
        Function<T, NDIM> calculate_Hamiltonian(World& world, const Function<T, NDIM>& V, const Function<T, NDIM>& phi) {
            // create the Hamiltonian matrix
    
            Function<T, NDIM> kin_energy = FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );
            for (int axis = 0; axis < NDIM; axis++) {
                Derivative<T, NDIM> D = free_space_derivative<T,NDIM>(world, axis); // Derivative operator

                Function<T, NDIM> d2_x2 = D(D(phi));

                kin_energy += 0.5 * d2_x2;
            }
        
            Function<T, NDIM> pot_energy = V * phi;
            
            return kin_energy + pot_energy;
        }

        Function<T, NDIM> calculate_Coulomb(World& world, const std::vector<Function<T, NDIM>>& functions) {
            // calculate the Coulomb Operator
            const int num = functions.size();
            Function<T, NDIM> sum = FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );

            for (int i = 0; i < num; i++) {
                sum += abs(functions[i]) * abs(functions[i]); // sum of the square of the functions |phi_i|^2
            }

            SeparatedConvolution<T, NDIM> coulomb_op = GaussOperator<NDIM>(world, 1.0);
            Function<T, NDIM> coulomb = coulomb_op(sum);

            return coulomb;
        }

        Function<T, NDIM> calculate_Exchange(World& world, const std::vector<Function<T, NDIM>>& functions, const Function<T, NDIM>& phi_k) {
            // calculate the exchange operator
            const int num = functions.size();
            Function<T, NDIM> exchange = FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );

            for (int l = 0; l < num; l++) {
                Function<T, NDIM> sum = phi_k * functions[l];           // phi_k * phi_l

                SeparatedConvolution<T, NDIM> coulomb_op = GaussOperator<NDIM>(world, 1.0);
                Function<T, NDIM> coulomb = coulomb_op(sum);

                exchange += coulomb;
            }

            return exchange;
        }

        Function<T, NDIM> calculate_Fock(World& world, const Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& functions, const Function<T, NDIM>& phi_k) {
            // calculate the Fock operator
            Function<T, NDIM> h = calculate_Hamiltonian(world, V, phi_k);
            
            Function<T, NDIM> coulomb = calculate_Coulomb(world, functions);
            Function<T, NDIM> J = coulomb * phi_k;  // Coulomb operator operating on phi_k

            Function<T, NDIM> exchange = calculate_Exchange(world, functions, phi_k); 

            return h + 2.0 * J - exchange;
        }

        Function<T, NDIM> optimize(World& world, Function<T, NDIM>& V, const Function<T, NDIM> guess_function, int N, const std::vector<Function<T, NDIM>>& prev_phi, int max_iter) {
            // optimize the guess function
            Function<T, NDIM> phi = guess_function;
            phi.scale(1.0 / phi.norm2()); // Normalize initial guess
            double E = energy(world, phi, V);

            NonlinearSolverND<NDIM> solver; // Nonlinear solver

            for (int iter = 0; iter < max_iter; iter++) {
                
                char filename[256];
                snprintf(filename, 256, "phi-%1d-%1d.dat", N, iter);
                plot1D(filename,phi);
                
                Function<T, NDIM> Fock = calculate_Fock(world, V, prev_phi, phi);
                
                SeparatedConvolution<T,NDIM> op = BSHOperator<NDIM>(world, 1.0, 0.001, 1e-7);  

                Function<T, NDIM> phi_next = phi - 2.0 * op(Fock); // the residual

                // Normalize phi_next
                double norm = phi_next.norm2();
                phi_next.scale(1.0 / norm);

                // Orthogonalize phi_next to all previous phi's
                for (const auto& prev : prev_phi) {
                    phi_next -= inner(prev, phi_next) * prev;
                }
            
                phi_next.scale(1.0 / phi_next.norm2());

                // Calculate energy
                E = energy(world, phi_next, V);
                double err = (phi - phi_next).norm2();

                // Update phi for the next iteration
                phi = phi_next;

                if (world.rank() == 0)
                    print("iteration", iter, "energy", E, "norm", norm, "error", err);

                if (err < 5e-4) break;
            }

            char filename[256];
            snprintf(filename, 256, "phi-%1d.dat", N);

            if (NDIM == 1)
                plot1D(filename, phi);
            else if (NDIM == 2)
                plot2D(filename,phi);

            print("Final energy without shift: ", E);

            return phi;
        }
         
};

#endif 