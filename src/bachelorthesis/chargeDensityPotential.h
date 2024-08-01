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
#include <string>
#include <vector>

using namespace madness;

// Class to create the potential from the charge density
// Input: World, Function (potential), 
// Output: Potential

template <typename T, int NDIM>
class ChargeDensityPotential {
    public:
        explicit ChargeDensityPotential(World& world) : world(world) {}

        Function<T, NDIM> create_potential(Function<T, NDIM> charge_density, double charge, SeparatedConvolution<T, NDIM>& op) {
            // Check if the charge density is correct
            // charge density should have a charge of 2.0 and a tolerance of 1e-6
            std::cout << "Checking charge density" << std::endl;
            if (!check_charge(charge_density, charge, 1e-6)) {
                std::cout << "Charge Density is incorrect" << std::endl;
            }

            // Create the potential with the given potential from the charge density
            // potential should be negative
            std::cout << "Charge is correct. Creating potential from charge density" << std::endl;
            
            return op(charge_density);
        }

    private:
        World& world;

        bool check_charge(Function<T, NDIM> charge_density, double c, double tol) {
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