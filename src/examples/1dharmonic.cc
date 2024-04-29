#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
#include <madness/mra/legendre.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>
#include <madness/mra/operator.h>
#include <madness/world/worldmpi.h>

using namespace madness;

const double L = 7.0; 
//const double DELTA = 3*L*L/2; // Use this to make fixed-point iteration converge
const double DELTA = 3.5;

// The initial guess wave function
double guess(const Vector<double,1>& r) {
  return exp(-(r[0]*r[0])/2.0);
}

// The shifted potential
double potential(const Vector<double,1>& r) {
  return 0.5*(r[0]*r[0]) - DELTA; 
}

// Convenience routine for plotting
void plot(const char* filename, const Function<double,1>& f) {
  Vector<double,1>lo(0.0), hi(0.0);
  lo[0] = -L; hi[0] = L;
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

int main(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);
  startup(world,argc,argv);
  if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());

  const double thresh = 1e-5;           
  FunctionDefaults<1>::set_k(6);        
  FunctionDefaults<1>::set_thresh(thresh);
  FunctionDefaults<1>::set_cubic_cell(-L, L);  // 1D cubic cell

  Function<double,1> phi = FunctionFactory<double, 1>(world).f(guess); 
  Function<double,1> V = FunctionFactory<double, 1>(world).f(potential);
  plot("potential.dat", V);

  phi.scale(1.0/phi.norm2());  // phi *= 1.0/norm
  double E = energy(world,phi,V); // Compute the energy of the initial guess

  // The nonlinear solver
  NonlinearSolver solver;
  
  for (int iter=0; iter<100; iter++) {
    char filename[256];
    snprintf(filename, 256, "phi-%1d.dat", iter);
    plot(filename,phi);

    Function<double, 1> Vphi = V*phi;
    Vphi.truncate();
    SeparatedConvolution<double,1> op = BSHOperator<1>(world, sqrt(-2*E), 0.01, thresh);  

    Function<double,1> r = phi + 2.0 * op(Vphi); // the residual
    double err = r.norm2();

    phi = phi - r; 

    double norm = phi.norm2();
    phi.scale(1.0/norm);  // phi *= 1.0/norm
    E = energy(world,phi,V);

    if (world.rank() == 0)
        print("iteration", iter, "energy", E, "norm", norm, "error",err);

    if (err < 5e-4) break;
  }

  print("Final energy without shift", E);

  if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
  finalize();
  return 0;
}