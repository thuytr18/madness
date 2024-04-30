#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
#include <madness/mra/legendre.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
#include <madness/mra/operator.h>
#include <madness/world/worldmpi.h>

using namespace madness;

const double L = 10.0; 
//const double DELTA = 3*L*L/2; // Use this to make fixed-point iteration converge
const double DELTA = 7.0;

// The initial guess wave function
template <int N>
struct GuessFunction {
  double operator()(const Vector<double, 1>& r) const {
    return exp(-r[0]*r[0])*std::pow(r[0], N);
  }
};

template<std::size_t N>     
struct num { static const constexpr auto value = N; };  // Helper struct for the for_ loop

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

// The shifted potential
double potential(const Vector<double,1>& r) {
  return 0.5*(r[0]*r[0]) - DELTA; 
}

// Convenience routine for plotting
void plot(const char* filename, const Function<double,1>& f) {
  Vector<double,1>lo(0.0), hi(0.0);
  lo[0] = -L;
  hi[0] = L;
  plot_line(filename,401,lo,hi,f);
}

double energy(World& world, const Function<double,1>& phi, const Function<double,1>& V) {
  double potential_energy = inner(phi,V*phi); // <phi|Vphi> = <phi|V|phi>
  double kinetic_energy = 0.0;
  
  Derivative<double,1> D = free_space_derivative<double,1>(world, 0);
  Function<double,1> dphi = D(phi);
  kinetic_energy += 0.5*inner(dphi,dphi);  // (1/2) <dphi/dx | dphi/dx>

  double energy = kinetic_energy + potential_energy;
  //print("kinetic",kinetic_energy,"potential", potential_energy, "total", energy);
  return energy;
}


// function to generate and solve for each energy level
template <int N>
void generate_and_solve(World& world, const Function<double, 1>& V) {
  auto guess_function = [](const Vector<double, 1>& r) {
      GuessFunction<N> functor;
      return functor(r);
  };

  // Create the initial guess wave function
  Function<double, 1> phi = FunctionFactory<double, 1>(world).f(guess_function); 

  phi.scale(1.0/phi.norm2()); // phi *= 1.0/norm
  double E = energy(world,phi,V); 

  NonlinearSolverND<1> solver;

  for(int iter = 0; iter <= 20; iter++) {
    char filename[256];
    snprintf(filename, 256, "phi-%1d.dat", N);
    plot(filename,phi);

    Function<double, 1> Vphi = V*phi;
    Vphi.truncate();
    SeparatedConvolution<double,1> op = BSHOperator<1>(world, sqrt(-2*E), 0.01, 1e-5);  

    Function<double,1> r = phi + 2.0 * op(Vphi); // the residual
    double err = r.norm2();

    phi = solver.update(phi, r);

    double norm = phi.norm2();
    phi.scale(1.0/norm);  // phi *= 1.0/norm
    E = energy(world,phi,V);

    if (world.rank() == 0)
          print("iteration", iter, "energy", E, "norm", norm, "error",err);

    if (err < 5e-4) break;
  }

  print("Final energy without shift: ", E + DELTA);

}

int main(int argc, char** argv) {

  // Initializing
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);
  startup(world,argc,argv);
  if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

  const double thresh = 1e-5;
  constexpr int num_levels = 5; // Number of levels

  // Set the defaults
  FunctionDefaults<1>::set_k(6);        
  FunctionDefaults<1>::set_thresh(thresh);
  FunctionDefaults<1>::set_cubic_cell(-L, L);  // 1D cubic cell

  // Create the initial potential 
  Function<double,1> V = FunctionFactory<double, 1>(world).f(potential);
  plot("potential.dat", V);
  
  for_<num_levels>([&] (auto i) {      
    generate_and_solve<i.value>(world, V);
  });

  
  if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
  finalize();
  return 0;
}