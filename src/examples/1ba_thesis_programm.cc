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

// The initial guess wave function
template <int N>
struct GuessFunction {
  double operator()(const Vector<double, 1>& r) const {
    return exp(-r[0]*r[0])*std::pow(r[0], N);
  }
};

template<typename T, std::size_t NDIM>
class GuessFactory {

    typedef Vector<double, NDIM> coordT; ///< Type of vector holding coordinates

    protected: 
        World& _world;
        int _num_guesses;
    public:

        GuessFactory(World& world, int num_guesses) : 
            _world(world),
            _num_guesses(num_guesses)  {
        }

        std::vector<Function<T, NDIM>> create_guesses() {
            std::vector<Function<T, NDIM>> guesses;
            for_<_num_guesses>([&](auto i) {
                GuessFunction<i.value> guess;   // initialize guess function
                Function<T, NDIM> guess_function = FunctionFactory<T, NDIM>(_world).f(GuessFunction<i>());  // create guess function
                guesses.push_back(guess_function); // add guess function to list
            });
            return guesses; // return list of guess functions
        }
};


// Function to find the eigenfunction 
std::vector<Function<double, 1>> find_eigenfunction(int N, int M, const std::vector<Function<double, 1>>& guess) {
}


int main(int argc, char** argv) {
    // Initializing
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world,argc,argv);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

    std::cout << "Hello, World!" << std::endl;


    // Finalizing
    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}