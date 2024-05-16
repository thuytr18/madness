#include <cstddef>
#include <iostream>
#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
#include <madness/mra/legendre.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
#include <madness/mra/operator.h>
#include <madness/tensor/gentensor.h>
#include <madness/world/world.h>
#include <madness/world/worldmpi.h>
#include <madness/tensor/tensor.h>
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

constexpr double L = 10.0;  // Length of the 1D cubic cell

//--------------------------------------------------------------------------------------------------------------------//
// Convenience routine for plotting
void plot(const char* filename, const Function<double,1>& f) {
  Vector<double,1>lo(0.0), hi(0.0);
  lo[0] = -L;
  hi[0] = L;
  plot_line(filename,401,lo,hi,f);
}
//--------------------------------------------------------------------------------------------------------------------//

// The initial guess wave function
template <int N>
double guessFunction (const Vector<double, 1>& r) {
    return exp(-r[0]*r[0])*std::pow(r[0], N);
}

template<typename T, std::size_t NDIM, int num_guesses>
class GuessFactory {

    typedef Vector<double, NDIM> coordT; ///< Type of vector holding coordinates

    protected: 
        World& _world;
    public:

        explicit GuessFactory(World& world) : 
            _world(world)  {
        }

        std::vector<Function<T, NDIM>> create_guesses() {
            std::vector<Function<T, NDIM>> guesses;
            for_<num_guesses>([&](auto i) {
                Function<double, NDIM> guess_function = FunctionFactory<T, NDIM>(_world).f(&guessFunction<i.value>);  // create guess function
                guesses.push_back(guess_function); // add guess function to list
            });
            return guesses; // return list of guess functions
        }
};

// Function to create guesses
template<typename T, std::size_t NDIM, int N, int M>
std::vector<Function<T, NDIM>> create_guesses(World& world) {
    std::vector<Function<T, NDIM>> guesses;
    constexpr int num = N + M;
    for_<num>([&](auto i) {
        Function<double, NDIM> guess_function = FunctionFactory<T, NDIM>(world).f(&guessFunction<i.value>);  // create guess function
        guesses.push_back(guess_function); // add guess function to list
    });

    return guesses; // return list of guess functions
}

// Function to calculate the Hamiltonian matrix
template <typename T, std::size_t NDIM>
Tensor<T> calculate_Hamiltonian(World &world, const std::vector<Function<T, NDIM>>& guesses, const Function<T, NDIM>& V){
    const int num = guesses.size();
    Tensor<T> H(num, num); // Hamiltonian matrix

    Derivative<double,1> D = free_space_derivative<double,1>(world, 0); // Derivative operator

    for(int i = 0; i < num; i++) {
        for(int j = 0; j < num; j++) {
            Function<T, NDIM> dx_i = D(guesses[i]);
            Function<T, NDIM> dx_j = D(guesses[j]);

            double kin_energy = 0.5 * inner(dx_i, dx_j);  // (1/2) <dphi/dx | dphi/dx>
            double pot_energy = inner(guesses[i], V * guesses[j]); // <phi|V|phi>

            H(i, j) = kin_energy + pot_energy; // Hamiltonian matrix
        }
    }
    return H;
}

// Function to calculate the Overlap matrix
template <typename T, std::size_t NDIM>
Tensor<T> calculate_Overlap(World &world, const std::vector<Function<T, NDIM>>& guesses) {
    const int num = guesses.size();
    Tensor<T> overlap(num, num); // Overlap matrix

    for(int i = 0; i < num; i++) {
        for(int j = 0; j < num; j++) {
            overlap(i, j) = guesses[i].inner(guesses[j]); // <phi_i|phi_j>
        }
    }

    return overlap;
}

// Function to calculate the diagonal matrix
template <typename T, std::size_t NDIM>
Tensor<T> calculate_DiagonalMatrix(World &world, const std::vector<Function<T, NDIM>>& guesses, const Function<T, NDIM>& V) {
    Tensor<T> H = calculate_Hamiltonian(world, guesses, V); // Hamiltonian matrix
    Tensor<T> overlap = calculate_Overlap(world, guesses); // Overlap matrix

    Tensor<T> U;
    Tensor<T> evals;
    sygvp(world, H, overlap, 1, U, evals); // Solve the generalized eigenvalue problem

    Tensor<T> diag_matrix(guesses.size(), guesses.size()); // Diagonal matrix
    diag_matrix.fill(0.0);
    for(int i = 0; i < guesses.size(); i++) {
        diag_matrix(i, i) = evals(i); // Set the diagonal elements
    }

    return diag_matrix;
}

// Function to calculate the U matrix
template <typename T, std::size_t NDIM>
Tensor<T> calculate_UMatrix(World &world, const std::vector<Function<T, NDIM>>& guesses, const Function<T, NDIM>& V, int N) {
    Tensor<T> H = calculate_Hamiltonian(world, guesses,V);
    Tensor<T> overlap = calculate_Overlap(world, guesses);

    Tensor<double> U;
    Tensor<double> evals;
    sygvp(world, H, overlap, 1, U, evals);

    // |x> = U|x>
    std::vector<Function<double,1>> y(N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
        if (!j) {
            y[i] = U(i, j)*guesses[j];
        } else {
            y[i] = y[i] + U(i, j)*guesses[j];
        }
        }
    }
    
    std::cout << "U matrix: \n" << U << std::endl;
    return y;
}


int main(int argc, char** argv) {
    // Initializing
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world,argc,argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    const double thresh = 1e-5; // Threshold
    constexpr int num_levels = 10; // Number of levels

    // Set the defaults
    FunctionDefaults<1>::set_k(6);        
    FunctionDefaults<1>::set_thresh(thresh);
    FunctionDefaults<1>::set_cubic_cell(-L, L);  // 1D cubic cell

    std::vector<Function<double,1>> guesses = create_guesses<double, 1, 5, 5>(world);

    // Plotting the guesses -----------------------------------------------
    for(int i = 0; i < guesses.size(); i++) {
        char filename[256];
        snprintf(filename, 256, "guess-%1d.dat", i);
        plot(filename, guesses[i]);
    }
    //---------------------------------------------------------------------

    // Finalizing
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}