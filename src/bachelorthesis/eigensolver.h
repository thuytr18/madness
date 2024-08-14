#ifndef EIGENSOLVER_H
#define EIGENSOLVER_H

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

// Class to create Guesses, diagonalize, optimize with BSH Operator and diagonalize again
// Input: World, Function (Potential), max Iteration, number of guesses or World, Function (Potential), max Iteration, number of guesses, Vector of Function (Guesses)
// Output: Vector of Function (Diagonalized Eigenfunctions)

template <typename T, std::size_t NDIM>
class Eigensolver {
    public:     
        explicit Eigensolver(World& world) : world(world) {}

        // Function to solve the eigenvalue problem for the given potential
        std::vector<Function<double, NDIM>> solve(Function<T, NDIM>& V, int num_levels, int max_iter) {
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

            // Diagonalize the Hamiltonian matrix
            std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> tmp = diagonalize(world, guesses, V);
            std::vector<Function<double, NDIM>> diagonalized_guesses = tmp.second;

            // store the eigenfunctions in vector eigenfunctions
            std::vector<Function<double, NDIM>> eigenfunctions;

            // Generate and solve the diagonalized guesses and store them in eigenfunctions (Optimization with BSH Operator)
            for (int i = 0; i < num_levels; i++) {
                Function<double, NDIM> phi = optimize(world, V, diagonalized_guesses[i], i, eigenfunctions, max_iter);
                eigenfunctions.push_back(phi);
            }

            // diagonalize the eigenfunctions again
            std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> diagonalized = diagonalize(world, eigenfunctions, V);
            std::cout << "Diagonalize" << std::endl;
            std::vector<Function<double, NDIM>> y = diagonalized.second;

            return y;
        }

        // Function to solve the eigenvalue problem for the given potential with given guesses
        std::vector<Function<double, NDIM>> solve(Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& guesses, int num_levels, int max_iter) {
            // Diagonalize the Hamiltonian matrix
            std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> tmp = diagonalize(world, guesses, V);
            std::vector<Function<double, NDIM>> diagonalized_guesses = tmp.second;

            // store the eigenfunctions in vector eigenfunctions
            std::vector<Function<double, NDIM>> eigenfunctions;

            // Generate and solve the diagonalized guesses and store them in eigenfunctions (Optimization with BSH Operator)
            for (int i = 0; i < num_levels; i++) {
                Function<double, NDIM> phi = optimize(world, V, diagonalized_guesses[i], i, eigenfunctions, max_iter);
                eigenfunctions.push_back(phi);
            }

            // diagonalize the eigenfunctions again
            std::pair<Tensor<double>, std::vector<Function<double, NDIM>>> diagonalized = diagonalize(world, eigenfunctions, V);
            std::cout << "Diagonalize" << std::endl;
            std::vector<Function<double, NDIM>> y = diagonalized.second;

            return y;
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


        // Function to optimize the eigenfunction for each energy level
        Function<T, NDIM> optimize(World& world, Function<T, NDIM>& V, const Function<T, NDIM> guess_function, int N, const std::vector<Function<T, NDIM>>& prev_phi, int max_iter) {

            // Create the initial guess wave function
            Function<T, NDIM> phi = guess_function;

            phi.scale(1.0/phi.norm2()); // phi *= 1.0/norm
            double E = energy(world, phi, V);

            NonlinearSolverND<NDIM> solver;
            int count_shift = 0; // counter how often the potential was shifted

            for(int iter = 0; iter <= max_iter; iter++) {
                
                char filename[256];
                snprintf(filename, 256, "phi-%1d-%1d.dat", N, iter);
                if (NDIM == 1)
                    plot1D(filename,phi);
                else if (NDIM == 2)
                    plot2D(filename,phi);
                else if (NDIM == 3)
                    plot3D(filename,phi);
                
                
                // Energy cant be positiv
                // shift potential

                double shift = 0.0;

                if (E > 0) {
                    shift = -20;
                    //shift = - 1.2 * E;
                    E = energy(world, phi, V + shift);
                    count_shift++;
                }

                Function<T, NDIM> Vphi = (V + shift) * phi;
                Vphi.truncate();
                
                SeparatedConvolution<T,NDIM> op = BSHOperator<NDIM>(world, sqrt(-2*E), 0.001, 1e-7);  

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
            else if (NDIM == 3)
                plot3D(filename,phi);

            if (count_shift != 0) {
                std::cout << "Potential was shifted " << count_shift << " times" << std::endl;
            }

            print("Final energy without shift: ", E);
            return phi;
        }

};

#endif