#ifndef DIAGONALIZED_H
#define DIAGONALIZED_H

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/vmra.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/tensor_lapack.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <ostream>
#include <vector>

using namespace madness;

// Class to create Guesses, diagonalize, optimize with BSH Operator and diagonalize again
// Input: World, Function (Potential), max Iteration, number of guesses or World, Function (Potential), max Iteration, number of guesses, Vector of Function (Guesses)
// Output: Vector of Function (Diagonalized Eigenfunctions)

#endif