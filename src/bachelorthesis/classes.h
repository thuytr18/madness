#ifndef CLASSES_H
#define CLASSES_H

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

const double DELTA = 20.0;

// Class to generate a harmonic potential
template<typename T, std::size_t NDIM>
class HarmonicPotentialGenerator {
    public:
        class HarmonicPotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                HarmonicPotentialFunctor();

                explicit HarmonicPotentialFunctor(const int &DELTA): DELTA(DELTA) {
                }

                const double DELTA;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    double potential = 0.0;
                    for(int i = 0; i < NDIM; i++) {
                        potential += 0.5 * (r[i] * r[i]); 
                    }

                    return potential - DELTA;  // shifted potential
                }
        };

        explicit HarmonicPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_harmonicpotential(double DELTA) {
            HarmonicPotentialFunctor harmonic_potential_function(DELTA);
            return FunctionFactory<T, NDIM>(world).functor(harmonic_potential_function);  // create potential function
        }

    private:
        World& world;
};

// Class to generate a gaussian potential
template<typename T, std::size_t NDIM>
class GaussianPotentialGenerator {
    public:
        class GaussianPotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                GaussianPotentialFunctor();

                explicit GaussianPotentialFunctor(const double &DELTA, const double& a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma): 
                    DELTA(DELTA), a(a), mu(mu), sigma(sigma) {
                }

                const double DELTA;
                const double a;
                const Vector<T, NDIM> mu;
                const Tensor<T> sigma;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    Vector<T, NDIM> diff;
                    for (int i = 0; i < NDIM; i++) {
                        diff[i] = r[i] - mu[i];
                    } 
                    
                    Tensor<double> sigma_inv = inverse(sigma); // inverse of sigma (covariance matrix
                    double sum = 0.0;

                    for (int i = 0; i < NDIM; ++i) {
                        for (int j = 0; j < NDIM; ++j) {
                            sum += diff[i] * sigma_inv(i, j) * diff[j];
                        }
                    }

                    double potential = - a * exp(- 0.5 * sum);

                    return potential; 
                }
        };

        explicit GaussianPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_gaussianpotential(double DELTA, double a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma) {
            GaussianPotentialFunctor gaussian_potential_function(DELTA, a, mu, sigma);
            return FunctionFactory<T, NDIM>(world).functor(gaussian_potential_function);  // create potential function
        }

    private:
        World& world;
};

// Class to generate a gaussian potential
template<typename T, std::size_t NDIM>
class DoubleWellPotentialGenerator {
    public:
        class DoubleWellPotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                DoubleWellPotentialFunctor();

                explicit DoubleWellPotentialFunctor(const double &DELTA, const double& a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma, 
                                                    const double& b, const Vector<T, NDIM>& mu2, const Tensor<T>& sigma2): 
                    DELTA(DELTA), a(a), mu(mu), sigma(sigma), b(b), mu2(mu2), sigma2(sigma2){
                }

                const double DELTA;
                const double a;
                const Vector<T, NDIM> mu;
                const Tensor<T> sigma;
                const double b;
                const Vector<T, NDIM> mu2;
                const Tensor<T> sigma2;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    Vector<T, NDIM> diff1;
                    for (int i = 0; i < NDIM; i++) {
                        diff1[i] = r[i] - mu[i];
                    } 
                    
                    Tensor<double> sigma_inv = inverse(sigma); // inverse of sigma (covariance matrix
                    double sum1 = 0.0;

                    for (int i = 0; i < NDIM; ++i) {
                        for (int j = 0; j < NDIM; ++j) {
                            sum1 += diff1[i] * sigma_inv(i, j) * diff1[j];
                        }
                    }

                    double potential1 = - a * exp(- 0.5 * sum1);

                    Vector<T, NDIM> diff2;
                    for (int i = 0; i < NDIM; i++) {
                        diff2[i] = r[i] - mu2[i];
                    } 
                    
                    Tensor<double> sigma2_inv = inverse(sigma2); // inverse of sigma (covariance matrix
                    double sum2 = 0.0;

                    for (int i = 0; i < NDIM; ++i) {
                        for (int j = 0; j < NDIM; ++j) {
                            sum2 += diff2[i] * sigma2_inv(i, j) * diff2[j];
                        }
                    }

                    double potential2 = - b * exp(- 0.5 * sum2);

                    return potential1 + potential2;
                }
        };

        explicit DoubleWellPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_doublewellpotential(double DELTA, double a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma, 
                                                    double b, const Vector<T, NDIM>& mu2, const Tensor<T>& sigma2) {
            DoubleWellPotentialFunctor doublewell_potential_function(DELTA, a, mu, sigma, b, mu2, sigma2);
            return FunctionFactory<T, NDIM>(world).functor(doublewell_potential_function);  // create potential function
        }

    private:
        World& world;
};

//----------------------------------------------------------------------------------------------------------------------------------------------//

// Class to generate harmonic guesses
template<typename T, std::size_t NDIM>
class HarmonicGuessGenerator {
    public:
        class HarmonicGuessFunctor : public FunctionFunctorInterface<T, NDIM> {
        public:
            HarmonicGuessFunctor();

            explicit HarmonicGuessFunctor(const int &order): order(order){
            }
            
            const int order;

            /// explicit construction
            double operator ()(const Vector<T, NDIM>& r) const override {
                return exp(-r[0]*r[0])*std::pow(r[0], order);
            }
        };

        explicit HarmonicGuessGenerator(World& world) : world(world) {
        }

        // Function to create guesses
        std::vector<Function<T, NDIM>> create_guesses(int num) {
            std::vector<Function<T, NDIM>> guesses;
            for(int i = 0; i < num; i++) {
                HarmonicGuessFunctor guessfunction(i);
                Function<T, NDIM> guess_function = FunctionFactory<T, NDIM>(world).functor(guessfunction);  // create guess function
                guesses.push_back(guess_function); // add guess function to list
            }
            return guesses; // return list of guess functions
        }

        private:
            World& world;
};

template<typename T, std::size_t NDIM>
class GuessGenerator {
    public:
        class GuessFunctor : public FunctionFunctorInterface<T, NDIM> {
        public:
            GuessFunctor();

            explicit GuessFunctor(const Vector<int, NDIM>& order, Function<T, NDIM>& V): order(order), V(V){
            }
            
            const Vector<int, NDIM> order;
            Function<T, NDIM> V;

            /// explicit construction
            double operator ()(const Vector<T, NDIM>& r) const override {
                double monomial = 1.0;
                for (int dim = 0; dim < NDIM; dim++) {
                    monomial *= std::pow(r[dim], order[dim]); 
                }
                return monomial * V(r);
            }
        };

        explicit GuessGenerator(World& world) : world(world) {
        }

        // Function to create guesses
        std::vector<Function<T, NDIM>> create_guesses(int num, Function<T, NDIM>& V) {
            std::vector<Function<T, NDIM>> guesses;
            int count = 0;
            int order = 0;
            Vector<int, NDIM> orders(0);
            // iterates over the orders of the monomials
            while (count < num){
                orders.fill(0);     // array to store the orders of the monomials
                orders[0] = order;    // set the order of the first monomial 
                std::cout << "Order of the first monomial:" << std::endl;
                std::cout << orders[0] << std::endl;
                while (true) {
                    GuessFunctor guessfunction(orders, V);
                    Function<T, NDIM> guess_function = FunctionFactory<T, NDIM>(world).functor(guessfunction);  // create guess function
                    guesses.push_back(guess_function); // add guess function to list

                    count++;
                    std::cout << "Counter: "<< count << " and current num: " << num << std::endl;

                    if(count >= num) {
                        std::cout << "Return guesses" << std::endl; 
                        return guesses;
                    }

                    std::cout << "Get the other orders: " << std::endl;

                    int first_nonzero = 0; // index of the first non-zero in the array

                    std::cout << "First non-zero: " << first_nonzero << std::endl;
                    std::cout << "order with first non-zero: " << orders[first_nonzero] << std::endl;

                    bool all_zero = true;

                    for (int j = 0; j < orders.size(); j++) {
                        if (orders[j] != 0) {
                            all_zero = false;
                            break;
                        }
                    }

                    if (all_zero) {
                        std::cout << "All zero" << std::endl;
                        break;
                    }

                    while (first_nonzero < NDIM && orders[first_nonzero] != 0) {
                        first_nonzero++;
                        std::cout << "First non-zero: " << first_nonzero << std::endl;
                    }

                    if (first_nonzero >= NDIM-1) {
                        std::cout << "Break" << std::endl;
                        break;
                    }

                    orders[first_nonzero] -= 1;
                    orders[first_nonzero + 1] += 1;

                    if (first_nonzero != 0) {
                        orders[0] = orders[first_nonzero];
                        orders[first_nonzero] = 0;   
                    }

                    for (int j = 0; j < orders.size(); j++) {
                        std::cout << orders[j] << " ";
                    }
                    std::cout << std::endl;
                    
                }
                ++order;
            }
            return guesses; // return list of guess functions
        }

        private:
            World& world;
};

//----------------------------------------------------------------------------------------------------------------------------------------------//
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

                std::cout << "gradient" << std::endl;
                // calculate the gradient of the function
                for(int ord = 0; ord < order; ord++) {
                    std::cout << "before my_gradient" << std::endl;
                    gradient = my_gradient(gradient);
                    std::cout << "Gradient " << std::endl;
                }

                // calculate the factorial of the order
                std::cout << "Factorial" << std::endl;
                int fac = factorial(order);

                // evaluate the gradient at the point x0
                gradient_values.resize(gradient.size());
                for(int i = 0; i < gradient.size(); i++) {
                    gradient_values[i] = gradient[i](x0) / fac; // evaluate and divide by factorial
                    std::cout << "Gradient: " << gradient_values[i] << std::endl;
                }
            }

            World& world;
            Function<T, NDIM> f;
            const Vector<T, NDIM> x0;
            const int order;
            std::vector<T> gradient_values;

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
        };  

        explicit TaylorSeriesGenerator(World& world) : world(world) {
        }

        // Function to create Taylor series
        Function<T, NDIM> create_taylorseries(World& world, Function<T, NDIM>& f, Vector<T, NDIM>& x0, int order) {
            // creates every term of order of taylor series
            // saves them in a vector
            std::vector<Function<T, NDIM>> taylor_series;
            for(int ord = 0; ord <= order; ord++) {
                std::cout << "Order: " << order << std::endl;
                std::cout << "ord: " << ord << std::endl;
                TaylorSeriesFunctor taylorseries_function(world, f, x0, ord);
                taylor_series.push_back(FunctionFactory<T, NDIM>(world).functor(taylorseries_function));  // create taylor series function
                std::cout << "ord: " << ord << std::endl;
            }
            // iterates over the terms of the taylor series and adds them up
            Function<T, NDIM> taylor =FunctionFactory<T, NDIM>(world).functor([] (const Vector<T, NDIM>& r) {return 0.0;} );

            for(int i = 0; i < taylor_series.size(); i++) {
                taylor += taylor_series[i];
            }
            return taylor;
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