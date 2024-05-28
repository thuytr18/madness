#ifndef CLASSES_H
#define CLASSES_H

#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>

using namespace madness;

const double DELTA = 20.0;

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

                explicit GaussianPotentialFunctor(const double &DELTA, const double& a): DELTA(DELTA), a(a){
                }

                const double DELTA;
                const double a;

                /// explicit construction
                double operator ()(const Vector<T, NDIM>& r) const override {
                    double sum = 0.0;
                    for(int i = 0; i < NDIM; i++) {
                        sum += a * (r[i] * r[i]); 
                    }
                    double potential = -exp(-sum);

                    return potential - DELTA;  // shifted potential
                }
        };

        explicit GaussianPotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_gaussianpotential(double DELTA, double a) {
            GaussianPotentialFunctor gaussian_potential_function(DELTA, a);
            return FunctionFactory<T, NDIM>(world).functor(gaussian_potential_function);  // create potential function
        }

    private:
        World& world;
};


#endif 