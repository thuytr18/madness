#ifndef CLASSES_H
#define CLASSES_H

#include <cmath>
#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/tensor_lapack.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>

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

                    return potential - DELTA;  // shifted potential
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

                    return potential1 + potential2 - DELTA;  // shifted potential
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

            explicit GuessFunctor(const int &order, Function<T, NDIM>& V): order(order), V(V){
            }
            
            const int order;
            Function<T, NDIM> V;

            /// explicit construction
            double operator ()(const Vector<T, NDIM>& r) const override {
                return std::pow(r[0], order) * V(r);
            }
        };

        explicit GuessGenerator(World& world) : world(world) {
        }

        // Function to create guesses
        std::vector<Function<T, NDIM>> create_guesses(int num, Function<T, NDIM>& V) {
            std::vector<Function<T, NDIM>> guesses;
            for(int i = 0; i < num; i++) {
                GuessFunctor guessfunction(i, V);
                Function<T, NDIM> guess_function = FunctionFactory<T, NDIM>(world).functor(guessfunction);  // create guess function
                guesses.push_back(guess_function); // add guess function to list
            }
            return guesses; // return list of guess functions
        }

        private:
            World& world;
};

#endif 