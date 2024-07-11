#ifndef CHARGEDENSITYPOTENTIAL_H
#define CHARGEDENSITYPOTENTIAL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
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

using namespace madness;

// Class to create the potential from the charge density
// Input: World, Function (potential), 
// Output: Potential

template <typename T, int NDIM>
class ChargeDensityPotential {
    public: 
        explicit ChargeDensityPotential(World& world) : world(world) {}

        Function<T, NDIM> create_potential(Function<T, NDIM> charge_density) {
            // Check if the charge density is correct
            // charge density should have a charge of 2.0 and a tolerance of 1e-6
            if (!check_charge(charge_density, 2.0, 1e-6)) {
                throw std::runtime_error("Charge density should have a charge of 2.0.");
            }

            // Create the potential with the coulomb potential from the charge density
            SeparatedConvolution<T, NDIM> coulomb_op = GaussOperator<NDIM>(world, 1.0);
            Function<T, NDIM> potential = coulomb_op(charge_density);

            return potential;
        }

    private:
        World& world;

        bool check_charge(Function<T, NDIM> charge_density, double c, double tol) {

            std::cout << "Checking charge density" << std::endl;
            std::cout << charge_density.trace() << std::endl;

            double charge = charge_density.trace();

            if (std::abs(charge - c) < tol) {
                return true;
            } else {
                std::cout << "Charge Density should have a charge of " << c << std::endl;
                return false;
            }
        }
}; 


#endif