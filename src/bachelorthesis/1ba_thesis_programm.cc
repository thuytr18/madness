#include <cmath>
#include <cstddef>
#include <iostream>
#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/legendre.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
#include <madness/mra/operator.h>
#include <madness/mra/vmra.h>
#include <madness/tensor/gentensor.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <madness/world/worldmpi.h>
#include <madness/tensor/tensor.h>
#include <utility>
#include <vector>

using namespace madness;

//--------------------------------------------------------------------------------------------------------------------//
// Helper struct for the for_ loop
template<std::size_t N>     
struct num { static const constexpr auto value = N; }; 

template <class F, std::size_t... Is>   // Loop over the integers 0,1,2,...,N-1
void for_(F func, std::index_sequence<Is...>)
{
  using expander = int[];   
  (void)expander{1, ((void)func(num<Is>{}), 1)...};
}

template <std::size_t N, typename F>  
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}
//--------------------------------------------------------------------------------------------------------------------//

const double L = 10.0;  // Length of the 1D cubic cell
const double DELTA = 20.0;

//--------------------------------------------------------------------------------------------------------------------//
// Convenience routine for plotting
/*
void plot(const char* filename, const Function<double,1>& f) {
  Vector<double,1>lo(0.0), hi(0.0);
  lo[0] = -L;
  hi[0] = L;
  plot_line(filename,401,lo,hi,f);
}
*/

template <typename T, std::size_t NDIM>
void plot(const char* filename, const Function<T, NDIM>& f) {
    Vector<T, NDIM> lo{}, hi{};
    for (std::size_t i = 0; i < NDIM; ++i) {
        lo[i] = -L;
        hi[i] = L;
    }
    plot_line(filename, 401, lo, hi, f);
}
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
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

// Class to generate a potential
template<typename T, std::size_t NDIM>
class PotentialGenerator {
    public:
        class PotentialFunctor: public FunctionFunctorInterface<T, NDIM> {
            public:
                PotentialFunctor();

                explicit PotentialFunctor(const int &DELTA): DELTA(DELTA) {
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

        explicit PotentialGenerator(World& world) : world(world) {
        }

        // Function to create potential
        Function<T, NDIM> create_potential(double DELTA) {
            PotentialFunctor potential_function(DELTA);
            return FunctionFactory<T, NDIM>(world).functor(potential_function);  // create potential function
        }

    private:
        World& world;
};
//--------------------------------------------------------------------------------------------------------------------//

// Function to calculate the energy
template <typename T, std::size_t NDIM>
double energy(World& world, const Function<T, NDIM>& phi, const Function<T, NDIM>& V) {
    double potential_energy = inner(phi, V * phi); // <phi|Vphi> = <phi|V|phi>
    double kinetic_energy = 0.0;

    Derivative<T, NDIM> D = free_space_derivative<T, NDIM>(world, 0); // Derivative operator

    Function<T, NDIM> dphi = D(phi);
    kinetic_energy += 0.5 * inner(dphi, dphi);  // (1/2) <dphi/dx | dphi/dx>

    double energy = kinetic_energy + potential_energy;
    return energy;
}


// Function to calculate the Hamiltonian matrix, Overlap matrix and Diagonal matrix
template <typename T, std::size_t NDIM>
std::pair<Tensor<T>, std::vector<Function<T, NDIM>>> diagonalize(World &world, const std::vector<Function<T, NDIM>>& functions, const Function<T, NDIM>& V){
    const int num = functions.size();
    
    auto H = Tensor<T>(num, num); // Hamiltonian matrix
    auto overlap = Tensor<T>(num, num); // Overlap matrix
    auto diag_matrix = Tensor<T>(num, num); // Diagonal matrix

    // Calculate the Hamiltonian matrix
    Derivative<T, NDIM> D = free_space_derivative<T,NDIM>(world, 0); // Derivative operator

    for(int i = 0; i < num; i++) {
        auto energy1 = energy(world, functions[i], V);
        std::cout << energy1 << std::endl;
        for(int j = 0; j < num; j++) {
            Function<T, NDIM> dx_i = D(functions[i]);
            Function<T, NDIM> dx_j = D(functions[j]);

            double kin_energy = 0.5 * inner(dx_i, dx_j);  // (1/2) <dphi/dx | dphi/dx>
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
    sygvp(world, H, overlap, 1, U, evals); // Solve the generalized eigenvalue problem

    diag_matrix.fill(0.0);
    for(int i = 0; i < num; i++) {
        diag_matrix(i, i) = evals(i); // Set the diagonal elements
    }

    std::cout << "dia_matrix: \n" << diag_matrix << std::endl;

    std::vector<Function<T, NDIM>> y;

    y = transform(world, functions, U);
    
    // std::cout << "U matrix: \n" << U << std::endl;
    // std::cout << "evals: \n" << evals << std::endl;
    return std::make_pair(evals, y);
}


// Function to generate and solve for each energy level
template <typename T, std::size_t NDIM>
Function<T, NDIM> generate_and_solve(World& world, const Function<T, NDIM>& V, const Function<T, NDIM> guess_function, int N, const std::vector<Function<T, NDIM>>& prev_phi) {

    // Create the initial guess wave function
    Function<T, NDIM> phi = guess_function;

    phi.scale(1.0/phi.norm2()); // phi *= 1.0/norm
    double E = energy(world, phi, V);

    NonlinearSolverND<NDIM> solver;

    for(int iter = 0; iter <= 50; iter++) {
        char filename[256];
        snprintf(filename, 256, "phi-%1d.dat", N);
        plot(filename,phi);

        Function<T, NDIM> Vphi = V*phi;
        Vphi.truncate();
        SeparatedConvolution<T,NDIM> op = BSHOperator<NDIM>(world, sqrt(-2*E), 0.01, 1e-7);  

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

    print("Final energy without shift: ", E + DELTA);
    return phi;
}


int main(int argc, char** argv) {
    // Initializing
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world,argc,argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    const double thresh = 1e-6; // Threshold
    constexpr int num_levels = 10; // Number of levels

    // Set the defaults
    FunctionDefaults<1>::set_k(6);        
    FunctionDefaults<1>::set_thresh(thresh);
    FunctionDefaults<1>::set_cubic_cell(-L, L);  // 1D cubic cell

    // Create the guess generator and potential generator

    HarmonicGuessGenerator<double, 1> guess_generator(world);
    PotentialGenerator<double, 1> potential_generator(world);

    // Create the guesses and potential
    std::vector<Function<double,1>> guesses = guess_generator.create_guesses(num_levels);
    Function<double, 1> V = potential_generator.create_potential(DELTA);

    // Plot the potential
    plot("potential.dat", V + DELTA);

    // Plot guesses
    for (int i = 0; i < guesses.size(); i++) {
        char filename[512];
        snprintf(filename, 512, "g-%1d.dat", i);
        plot(filename, guesses[i]);
    }

    // Solve for each energy level and store the eigenfunctions
    std::vector<Function<double, 1>> eigenfunctions;

    for (int i = 0; i < num_levels; i++) {
        Function<double, 1> phi = generate_and_solve(world, V, guesses[i], i, eigenfunctions);
        eigenfunctions.push_back(phi);
    }

    std::pair<Tensor<double>, std::vector<Function<double, 1>>> tmp = diagonalize(world, eigenfunctions, V);
    std::vector<Function<double, 1>> y = tmp.second;
    //auto evals = tmp.first;

    for (int i = 0; i < num_levels; i++) {
        char filename[256];
        snprintf(filename, 256, "Psi_%1d.dat", i);
        double en = energy(world, y[i], V);

        // for hamonic oscillator
        // V(r) = 0.5 * r * r = E
        // inverse function: r = sqrt(2 * E)
        if(y[i](std::sqrt(2* (en+DELTA))) < 0) {
            y[i] *= -1;
        }

        plot(filename, 0.75*y[i] + DELTA + en);
    }

    // Finalizing
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}