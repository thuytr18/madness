#include <cstddef>
#include <iostream>
#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
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

constexpr double L = 10.0;  // Length of the 1D cubic cell
const double DELTA = 20.0;

//--------------------------------------------------------------------------------------------------------------------//
// Convenience routine for plotting
void plot(const char* filename, const Function<double,1>& f) {
  Vector<double,1>lo(0.0), hi(0.0);
  lo[0] = -L;
  hi[0] = L;
  plot_line(filename,401,lo,hi,f);
}
//--------------------------------------------------------------------------------------------------------------------//

template<typename T, std::size_t NDIM>
class HarmonicGuessFunctor : public FunctionFunctorInterface<T, NDIM> {
public:
	HarmonicGuessFunctor();

	HarmonicGuessFunctor(const int &order): order(order){
	}
	
	const int order;

	/// explicit construction
	double operator ()(const Vector<T, NDIM>& r) const {
        return exp(-r[0]*r[0])*std::pow(r[0], order);
    }
};

// Function to create guesses
template<typename T, std::size_t NDIM>
std::vector<Function<T, NDIM>> create_guesses(World& world, int num) {
    std::vector<Function<T, NDIM>> guesses;
    for(int i = 0; i < num; i++) {
        HarmonicGuessFunctor<T, NDIM> guessfunction(i);
        Function<T, NDIM> guess_function = FunctionFactory<T, NDIM>(world).functor(guessfunction);  // create guess function
        guesses.push_back(guess_function); // add guess function to list
    }
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
    overlap = matrix_inner(world, guesses, guesses);

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
std::pair<Tensor<T>, std::vector<Function<T, NDIM>>> calculate_UMatrix(World &world, const std::vector<Function<T, NDIM>>& guesses, const Function<T, NDIM>& V, int N) {
    Tensor<T> H = calculate_Hamiltonian(world, guesses,V);
    Tensor<T> overlap = calculate_Overlap(world, guesses);

    Tensor<T> U;
    Tensor<T> evals;
    sygvp(world, H, overlap, 1, U, evals);

    // |x> = U|x>
    std::vector<Function<T, NDIM>> y(N);

    // to do: clean
    auto y_2 = transform(world, guesses, U);
    auto H_1 = calculate_Hamiltonian(world, y,V);
    auto H_2 = calculate_Hamiltonian(world, y_2,V);

    auto S = matrix_inner(world, y_2, y);
    
    std::cout << "S matrix: \n" << S << std::endl;
    std::cout << "U matrix: \n" << U << std::endl;
    std::cout << "H1 matrix: \n" << H_1 << std::endl;
    std::cout << "H2 matrix: \n" << H_2 << std::endl;
    std::cout << "evals: \n" << evals << std::endl;
    return std::make_pair(evals, y);
}

double potential(const Vector<double,1>& r) {
  return 0.5*(r[0]*r[0]) - DELTA; 
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

    std::vector<Function<double,1>> guesses = create_guesses<double, 1>(world, 10);
     // Create the initial potential 
  Function<double,1> V = FunctionFactory<double, 1>(world).f(potential);

    auto tmp = calculate_UMatrix(world, guesses, V, 5);
    auto evals = tmp.first;

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