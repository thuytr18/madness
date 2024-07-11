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
        // Function to solve the Hartree-Fock equation for a given potential
        std::vector<Function<T, NDIM>> solve(Function<T, NDIM>& V, int num_levels, int max_iter) {

        }

    private:
        World& world;

        Tensor<T> calculate_HartreeFockOp(World& world, const Function<T, NDIM>& V, const std::vector<Function<T, NDIM>>& functions) {
            // create the Hamiltonian matrix
            const int num = functions.size();

            auto H = Tensor<T>(num, num);

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

            // calculate the Coulomb Operator

            double sum = 0.0;

            for (int i = 0; i < num; i++) {
                sum += abs(functions[i]) * abs(functions[i]);
            }

            SeparatedConvolution<T, NDIM> coulomb_op = GaussOperator<NDIM>(world, 1.0);
            Function<T, NDIM> coulomb = coulomb_op(sum);

            // calculate the exchange operator

            double exchange = 0.0;

            for (int i = 0; i < num; i++) {
                for (int j = 0; j < num; j++) {
                    double result = functions[i] * functions[j];
                    coulomb_op = GaussOperator<NDIM>(world, 1.0);
                    Function<T, NDIM> result1  = coulomb_op(result);
                    exchange += result1;
                }
            }

            Tensor<T> hartree_fock_op = H + 2 * coulomb - exchange;
        }
}

#endif 