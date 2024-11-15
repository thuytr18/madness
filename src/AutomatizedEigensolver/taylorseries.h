#ifndef TAYLORSERIES_H
#define TAYLORSERIES_H

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

// helper function for TaylorSeriesGenerator

// Function to calculate the tensor product of two vectors of functions
template <typename T>
std::vector<T> tensor_product(const std::vector<T>& v, const std::vector<T>& w) {
    int N = v.size();
    int M = w.size();
    std::vector<T> result(N*M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result[i*M + j] = v[i] * w[j];
        }
    }
    return result;
}

// Function to calculate the gradient of a vector of functions and put in another vector of functions
template <typename T, std::size_t NDIM>
std::vector<Function<T, NDIM>> my_gradient(const std::vector<Function<T, NDIM>>& f) {
    int N = f.size();
    std::vector<Function<T, NDIM>> result(N*NDIM);
    // iterate over the row of the vector of functions f
    for (int i = 0; i < N; i++) {
        grad(f[i]);     // calculate the gradient of each function in f
        for (int j = 0; j < NDIM; j++) {
            result[i*NDIM + j] = grad(f[i])[j];  // put the gradient in the vector of functions result
        }
    }
    return result;
}


// Class to generate the Taylor series of given function

template<typename T, std::size_t NDIM>
class TaylorSeriesGenerator {
    public:
        class TaylorSeriesFunctor : public FunctionFunctorInterface<T, NDIM> {
        public:
            TaylorSeriesFunctor();

            explicit TaylorSeriesFunctor(World& world, Function<T, NDIM>& f, const Vector<T, NDIM>& x0, const int& order): world(world), f(f), x0(x0), order(order) {
                std::vector<Function<T, NDIM>> gradient = {f}; 

                // calculate the gradient of the function
                for(int ord = 0; ord < order; ord++) {
                    gradient = my_gradient(gradient);
                }

                // calculate the factorial of the order
                int fac = factorial(order);

                // evaluate the gradient at the point x0
                gradient_values.resize(gradient.size());
                for(int i = 0; i < gradient.size(); i++) {
                    gradient_values[i] = gradient[i](x0) / fac; // evaluate and divide by factorial
                    std::cout << "Gradient value: " << gradient_values[i] << std::endl;
                    if(order == 2) {
                        quadratic_coefficients.push_back(gradient_values[i]);
                    }
                }
            }

            World& world;
            Function<T, NDIM> f;
            const Vector<T, NDIM> x0;
            const int order;
            std::vector<T> gradient_values;
            std::vector<T> quadratic_coefficients;

            /// explicit construction
            double operator ()(const Vector<T, NDIM>& r) const override {
                std::vector<T> diff(NDIM);
                for(int i = 0; i < NDIM; i++) {
                    diff[i] = r[i] - x0[i];
                }

                std::vector<T> monomials = {1.0};
                for(int ord = 0; ord < order; ord++) {
                    monomials = tensor_product(monomials, diff);
                }

                std::flush(std::cout);
                T taylor_series = 0.0;
                for(int i = 0; i < gradient_values.size(); i++) {
                    std::flush(std::cout);
                    taylor_series += monomials[i] * gradient_values[i];
                }
                return taylor_series;
            }

            std::vector<T> get_quadratic_coefficients() {
                for(int i = 0; i < quadratic_coefficients.size(); i++) {
                    std::cout << "Quadratic coefficient " << quadratic_coefficients[i] << std::endl;
                }
                return quadratic_coefficients;
            }
        };  

        explicit TaylorSeriesGenerator(World& world) : world(world) {
        }

        // Function to create Taylor series
         std::pair<Function<T, NDIM>, std::vector<T>> create_taylorseries(World& world, Function<T, NDIM>& f, Vector<T, NDIM>& x0, int order) {
            // creates every term of order of taylor series
            // saves them in a vector
            std::vector<Function<T, NDIM>> taylor_series;
            std::vector<T> quadratic_coefficients;

            for(int ord = 0; ord <= order; ord++) {
                TaylorSeriesFunctor taylorseries_function(world, f, x0, ord);
                taylor_series.push_back(FunctionFactory<T, NDIM>(world).functor(taylorseries_function));  // create taylor series function

                if (ord == 2) {
                    quadratic_coefficients = taylorseries_function.get_quadratic_coefficients();
                }
            }
            // iterates over the terms of the taylor series and adds them up
            Function<T, NDIM> taylor =FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );

            for(int i = 0; i < taylor_series.size(); i++) {
                taylor += taylor_series[i];
            }

            return std::make_pair(taylor, quadratic_coefficients);
        }

        private:
            World& world;

            static int factorial(int n) {
                if(n == 0) {
                    return 1;
                }
                return n * factorial(n-1);
            }
};

#endif 