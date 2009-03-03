#define WORLD_INSTANTIATE_STATIC_TEMPLATES

#include <mra/mra.h>
#include <iostream>
#include <cmath>
#include <complex>

extern "C" void readinput_();
extern "C" void wann_init1_();
extern "C" void wann_unk_(int* n,double* vpl,double* vrc,double* val);

using namespace madness;

typedef SharedPtr< FunctionFunctorInterface< std::complex<double> ,3> > cfunctorT;
typedef FunctionFactory<std::complex<double>,3> cfactoryT;
typedef Function<std::complex<double>,3> cfunctionT;
typedef SharedPtr< FunctionFunctorInterface<double,3> > functorT;
typedef FunctionFactory<double,3> factoryT;
typedef Function<double,3> functionT;
typedef Vector<double,3> coordT;

//***************************************************************************
double abs(double x) {return x;}
//***************************************************************************

//***************************************************************************
double real(double x) {return x;}
//***************************************************************************

//***************************************************************************
template <typename Q, int NDIM>
struct abs_square_op
{
  typedef typename TensorTypeData<Q>::scalar_type resultT;
  Tensor<resultT> operator()(const Key<NDIM>& key, const Tensor<Q>& t) const
  {
    Tensor<resultT> result(t.ndim, t.dim);
    BINARY_OPTIMIZED_ITERATOR(Q, t, resultT, result, resultT d = abs(*_p0); *_p1 = d*d);
    return result;
  }
  template <typename Archive>
  void serialize(Archive& ar) {}
};
//***************************************************************************

//***************************************************************************
template<typename Q, int NDIM>
Function<typename TensorTypeData<Q>::scalar_type,NDIM> abs_square(const Function<Q,NDIM>& func)
{
  return unary_op(func, abs_square_op<Q,NDIM>());
}
//***************************************************************************

std::complex<double> func(int n, const Vector<int,3>& nk, double xx, double yy, double zz)
{
  // initialize value to zero
  double val[2];
  val[0] = 0.0; val[1] = 0.0;
  // number of k-points
  int nkpt = nk[0] * nk[1] * nk[2];
  //compute \sum_{k}u_{nk}(r)
  double vkl[3];
  // translate point
  double vr[3];
  vr[0] = xx; vr[1] = yy; vr[2] = zz;
  for (int j1 = 0; j1 < nk[0]; j1++)
  {
    for (int j2 = 0; j2 < nk[1]; j2++)
    {
      for (int j3 = 0; j3 < nk[2]; j3++)
      {
        // temp variable
        double val0[2];
        //k-point in lattice coordinates
        vkl[0] = 1.0 * j1 / nk[0];
        vkl[1] = 1.0 * j2 / nk[1];
        vkl[2] = 1.0 * j3 / nk[2];
        //get value of u_{nk}(r)
//        printf("%10.2f%10.2f%10.2f%15.7d\n", vr[0], vr[1], vr[2], n);
//        printf("%10.2f%10.2f%10.2f%15.7d\n", xx, yy, zz, n);
        wann_unk_(&n, &vkl[0], &vr[0], &val0[0]);
        val[0] += val0[0];
        val[1] += val0[1];
      }
    }
  }
  val[0]=val[0]/nkpt;
  val[1]=val[1]/nkpt;

  return std::complex<double>(val[0], val[1]);
}

template<typename T, int NDIM>
class Wannier: public FunctionFunctorInterface<T, NDIM>
{
public:
  typedef Vector<double, NDIM> coordT;
  int _n;
  coordT _center;
  Vector<int,3> _nk;

  Wannier(int n, const coordT& center, const Vector<int,3>& nk) :
    _n(n), _center(center), _nk(nk) {}

  T operator()(const coordT& x) const
  {
    return func(_n, _nk, x[0], x[1], x[2]);
  }
};

void test_wannier(World& world)
{
    //k-mesh division
    Vector<int,3> nk(4);
    //cener point of the box
    Vector<double,3> center(0.0);
    //index of Wannier function
    int n=2;

    readinput_();
    wann_init1_();

    // Function defaults
    int funck = 6;
    double thresh = 1e-4;
    double bsize = 6.0;
    FunctionDefaults<3>::set_k(funck);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_cubic_cell(-bsize, bsize);

    cfunctionT w = cfactoryT(world).functor(cfunctorT(new Wannier<std::complex<double>,3>(n, center, nk)));
//    cfunctionT w = cfactoryT(world);

    {
      w.reconstruct();
      if (world.rank() == 0)  printf("\n");
      double L = bsize / 2;
      int npts = 5;

      for (int i = 0; i < npts; i++)
      {
        for (int j = 0; j < npts; j++)
        {
          for (int k = 0; k < npts; k++)
          {
            double x = (i+0.5) * (2.0*bsize/npts) - bsize;
            double y = (j+0.5) * (2.0*bsize/npts) - bsize;
            double z = (k+0.5) * (2.0*bsize/npts) - bsize;
            coordT p(0.0);
            p[0] = x; p[1] = y; p[2] = z;
            double fval = abs(func(n, nk, p[0], p[1], p[2]));
            double fval2 = abs(w(p));
            double fdiff = fval2 - fval;
            if (world.rank() == 0)
              printf("%10.2f%10.2f%10.2f%15.8f%15.8f%15.8f\n", x, y, z, fval2, fval, fdiff);
          }
        }
      }
    }

    double wnorm = w.norm2();
    if (world.rank() == 0)
      printf("Normalization of wannier function is: %14.8f\n\n", wnorm);
//    // Plot to OpenDX
//    vector<long> npt(3,101);
//    plotdx(w, "wannier2.dx", FunctionDefaults<3>::get_cell(), npt);

}

void test_wannier3(World& world)
{
    //k-mesh division
    Vector<int,3> nk(4);
    //cener point of the box
    Vector<double,3> center(0.0);

    readinput_();
    wann_init1_();

    // Function defaults
    int funck = 6;
    double thresh = 1e-4;
    double bsize = 6.0;
    FunctionDefaults<3>::set_k(funck);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_cubic_cell(-bsize, bsize);

    std::vector<cfunctionT> w = zero_functions<std::complex<double>,3>(world,5);

    for (int n = 1; n <= 5; n++)
    {
      w[n-1] = cfactoryT(world).functor(cfunctorT(new Wannier<std::complex<double>,3>(n, center, nk)));
      double wnorm = w[n-1].norm2();
      if (world.rank() == 0)
        printf("Normalization of wannier function #%d is: %14.8f\n", n, wnorm);
    }
    if (world.rank() == 0) printf("\n\n");

    for (int i = 1; i <= 5; i++)
    {
      for (int j = 1; j < i; j++)
      {
        double tmp = real(inner(w[i],w[j]));
        if (world.rank() == 0)
          printf("Inner product (%d,%d) = %15.8f\n", i, j, tmp);
      }
    }
}

void compute_U(World& world)
{
    //k-mesh division
    Vector<int,3> nk(4);
    //cener point of the box
    Vector<double,3> center(0.0);

    readinput_();
    wann_init1_();

    // Function defaults
    int funck = 7;
    double thresh = 1e-5;
    double bsize = 6.0;
    FunctionDefaults<3>::set_k(funck);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_cubic_cell(-bsize, bsize);

    SeparatedConvolution<double,3> cop = CoulombOperator<double,3>(world,
        FunctionDefaults<3>::get_k(), 1e-8, thresh * 0.1);

    // Vector of cfunctionT's
    std::vector<cfunctionT> w;
    // Compress wannier functions and add to vector
    for (int n = 1; n <= 5; n++)
    {
      cfunctionT w_ = cfactoryT(world).functor(cfunctorT(new Wannier<std::complex<double>,3>(n, center, nk)));
      w_.truncate();
      w.push_back(w_);
    }

    for (int n = 1; n <= 5; n++)
    {
      functionT tmp_n = abs_square(w[n-1]);
      tmp_n.truncate();
      functionT tmp_n_apply = apply(cop, tmp_n);
      tmp_n_apply.truncate();
      for (int p = 1; p <= n; p++)
      {
        functionT tmp_p = abs_square(w[p-1]);
        tmp_p.truncate();
        // Compute inner product
        double ip = abs(inner(w[n-1],w[p-1]));
        // Compute U
        double U = inner(tmp_p, tmp_n_apply);
        if (world.rank() == 0) printf("%4.1d%4.1d%15.8f%15.8f\n", n, p, ip, U);
      }
    }
    if (world.rank() == 0) printf("\n\n");

}

void test_wannier2(World& world)
{
    //k-mesh division
    Vector<int,3> nk(4);
    //cener point of the box
    Vector<double,3> center(0.0);
    //index of Wannier function
    int n=3;

    readinput_();
    wann_init1_();

    if (world.rank() == 0)  printf("\n");
    double bsize = 6.0;
    int npts = 30000;

    for (int k = 0; k < npts; k++)
    {
      double z = (k+0.5) * (2.0*bsize/npts) - bsize;
      double fval = abs(func(n, nk, 0.0, 0.0, z));
      if (world.rank() == 0)
        printf("%20.12f%20.12f\n", z, fval);
    }
}

#define TO_STRING(s) TO_STRING2(s)
#define TO_STRING2(s) #s

//*****************************************************************************
int main(int argc, char** argv)
{
  initialize(argc, argv);
  World world(MPI::COMM_WORLD);
  if (world.rank() == 0)
  {
    print("");
    print("--------------------------------------------");
    print("   MADNESS", " multiresolution testsuite");
    print("--------------------------------------------");
    print("");
    print("   number of processors ...", world.size());
    print("    processor frequency ...", cpu_frequency());
    print("            host system ...", TO_STRING(HOST_SYSTEM));
    print("             byte order ...", TO_STRING(MADNESS_BYTE_ORDER));
    print("          configured by ...", MADNESS_CONFIGURATION_USER);
    print("          configured on ...", MADNESS_CONFIGURATION_HOST);
    print("          configured at ...", MADNESS_CONFIGURATION_DATE);
    print("                    CXX ...", MADNESS_CONFIGURATION_CXX);
    print("               CXXFLAGS ...", MADNESS_CONFIGURATION_CXXFLAGS);
#ifdef WORLD_WATCHDOG
    print("               watchdog ...", WATCHDOG_BARK_INTERVAL,
        WATCHDOG_TIMEOUT);
#endif
#ifdef OPTERON_TUNE
    print("             tuning for ...", "opteron");
#elif defined(CORE_DUO_TUNE)
    print("             tuning for ...", "core duo");
#else
    print("             tuning for ...", "core2");
#endif
#ifdef BOUNDS_CHECKING
    print(" tensor bounds checking ...", "enabled");
#endif
#ifdef TENSOR_INSTANCE_COUNT
    print("  tensor instance count ...", "enabled");
#endif
    print(" ");
  }

  try
  {
    printf("WSTHORNTON: Starting up the world ... \n");

    startup(world,argc,argv);
    if (world.rank() == 0) print("Initial tensor instance count", BaseTensor::get_instance_count());
    compute_U(world);
  }
  catch (const MPI::Exception& e)
  {
    //        print(e);
    error("caught an MPI exception");
  }
  catch (const madness::MadnessException& e)
  {
    print(e);
    error("caught a MADNESS exception");
  }
  catch (const madness::TensorException& e)
  {
    print(e);
    error("caught a Tensor exception");
  }
  catch (const char* s)
  {
    print(s);
    error("caught a string exception");
  }
  catch (const std::string& s)
  {
    print(s);
    error("caught a string (class) exception");
  }
  catch (const std::exception& e)
  {
    print(e.what());
    error("caught an STL exception");
  }
  catch (...)
  {
    error("caught unhandled exception");
  }

  if (world.rank() == 0)
    print("entering final fence");
  world.gop.fence();
  if (world.rank() == 0)
    print("done with final fence");
  if (world.rank() == 0)
    print("Final tensor instance count", BaseTensor::get_instance_count());

  finalize();
  return 0;
}
