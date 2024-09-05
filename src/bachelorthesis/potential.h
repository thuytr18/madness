#ifndef POTENTIAL_H
#define POTENTIAL_H

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

// Class to generate a harmonic potential
template<typename T, std::size_t NDIM>
class HarmonicPotentialGenerator {
    public:
        class HarmonicPotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                HarmonicPotentialFunctor();

                explicit HarmonicPotentialFunctor(const double& potential_shift): potential_shift(potential_shift) {
                }

                const double potential_shift;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    double potential = 0.0;
                    for(int i = 0; i < NDIM; i++) {
                        potential += 0.5 * (r[i] * r[i]); 
                    }

                    return potential - potential_shift;  
                }
        };

        explicit HarmonicPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_harmonicpotential(double potential_shift) {
            HarmonicPotentialFunctor harmonic_potential_function(potential_shift);
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

                explicit GaussianPotentialFunctor(const double& a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma): 
                    a(a), mu(mu), sigma(sigma) {
                }

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

                    double potential = a * exp(- 0.5 * sum);

                    return potential; 
                }
        };

        explicit GaussianPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_gaussianpotential(double a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma) {
            GaussianPotentialFunctor gaussian_potential_function(a, mu, sigma);
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

                explicit DoubleWellPotentialFunctor(const double& a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma, 
                                                    const double& b, const Vector<T, NDIM>& mu2, const Tensor<T>& sigma2): 
                    a(a), mu(mu), sigma(sigma), b(b), mu2(mu2), sigma2(sigma2){
                }

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

                    double potential1 = a * exp(- 0.5 * sum1);

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

                    double potential2 = b * exp(- 0.5 * sum2);

                    return potential1 + potential2;
                }
        };

        explicit DoubleWellPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_doublewellpotential(double a, const Vector<T, NDIM>& mu, const Tensor<T>& sigma, 
                                                    double b, const Vector<T, NDIM>& mu2, const Tensor<T>& sigma2) {
            DoubleWellPotentialFunctor doublewell_potential_function(a, mu, sigma, b, mu2, sigma2);
            return FunctionFactory<T, NDIM>(world).functor(doublewell_potential_function);  // create potential function
        }

    private:
        World& world;
};


template <typename T, std::size_t NDIM>
class ExponentialPotentialGenerator {
    public:
        class ExponentialPotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                ExponentialPotentialFunctor();

                explicit ExponentialPotentialFunctor(const double& a, const double& b): a(a), b(b){
                }

                const double a;
                const double b;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    double sum = 0.0;
                    for(int i = 0; i < NDIM; i++) {
                        sum += b * b * r[i] * r[i]; 
                    }
                    
                    sum = sqrt(sum);

                    double potential = a * exp(- sum);

                    return potential;
                }
        };

        explicit ExponentialPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_exponentialpotential(double a, double b) {
            ExponentialPotentialFunctor exponential_potential_function(a, b);
            return FunctionFactory<T, NDIM>(world).functor(exponential_potential_function);  // create potential function
        }

    private:
        World& world;
};


template <typename T, std::size_t NDIM>
class MorsePotentialGenerator {
    public:
        class MorsePotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                MorsePotentialFunctor();

                explicit MorsePotentialFunctor(const double& D, const double& a, const double& R): D(D), a(a), R(R) {
                }

                const double D;
                const double a;
                const double R;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    double sum = 0.0;
                    for(int i = 0; i < NDIM; i++) {
                        sum += r[i] * r[i]; 
                    }
                    
                    sum = sqrt(sum);

                    double ex = exp(-a * (sum - R));

                    double potential = D * std::pow((1 - ex), 2);

                    return potential - D; 
                }
        };

        explicit MorsePotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_morsepotential(double D, double a, double R){
            MorsePotentialFunctor morse_potential_function(D, a, R);
            return FunctionFactory<T, NDIM>(world).functor(morse_potential_function);  // create potential function
        }

    private:
        World& world;
};

// Class to generate a harmonic potential
template<typename T, std::size_t NDIM>
class PotentialGenerator {
    public:
        class PotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                PotentialFunctor();

                explicit PotentialFunctor(const double& a, const double& b): a(a), b(b) {
                }

                const double a;
                const double b;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    double potential = 0.0;
                    for(int i = 0; i < NDIM; i++) {
                        potential += a * (r[i] * r[i]); 
                    }

                    return potential - b;  
                }
        };

        explicit PotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_potential(double a, double b) {
            PotentialFunctor potential_function(a, b);
            return FunctionFactory<T, NDIM>(world).functor(potential_function);  // create potential function
        }

    private:
        World& world;
};

template<typename T, std::size_t NDIM>
class NuclearPotentialGenerator {
    public:
        class NuclearPotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                NuclearPotentialFunctor();

                explicit NuclearPotentialFunctor(double a, const Vector<T, NDIM>& R): a(a), R(R) {
                }

                double a;
                const Vector<T, NDIM> R;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    Vector<T, NDIM> diff = r - R;

                    double sum = 0.0;
                    for(int i = 0; i < NDIM; i++) {
                        sum += diff[i] * diff[i]; 
                    }
                    
                    sum = sqrt(sum);

                    double potential = -a / sum;

                    return potential; 
                }
            };

            explicit NuclearPotentialGenerator(World& world) : world(world) {
            }

            // Function to create potential
            Function<T, NDIM> create_nuclearpotential(double a, const Vector<T,NDIM>& R){
                NuclearPotentialFunctor nuclear_potential_function(a, R);
                return FunctionFactory<T, NDIM>(world).functor(nuclear_potential_function);  // create potential function
            }

    private:
        World& world;
};


#endif 