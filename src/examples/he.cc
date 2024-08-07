#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <madness/mra/lbdeux.h>

using namespace madness;

static const double L = 30.0;   // box size
static const long k = 6;        // wavelet order
static const double thresh = 1e-4; // precision

static const double   rcut = 0.01; // Smoothing distance in 1e potential
static const double r12cut = 0.01; // Smoothing distance in 2e potential
static const double   dcut = 0.01; // Smoothing distance in wave function
static const double d12cut = 0.01; // Smoothing distance in wave function

static const long oi = 0;        // orbital (mcquarrie, hylleraas-3term)
static const long vi = 1;        // potential (all, pot1/2, eri, Vpsi_eri)

// Smoothed 1/r potential (c is the smoothing distance)
static double u(double r, double c) {
    r = r/c;
    double r2 = r*r, pot;
    if (r > 6.5){
        pot = 1.0/r;
    } else if (r > 1e-2) {
        pot = erf(r)/r + exp(-r2)*0.56418958354775630;
    } else{
        pot = 1.6925687506432689-r2*(0.94031597257959381-r2*(0.39493270848342941-0.12089776790309064*r2));
    }

    return pot/c;
}

void distances(const coord_6d& r, double& r1, double& r2, double& r12) {
    const double x1=r[0], y1=r[1], z1=r[2];
    const double x2=r[3], y2=r[4], z2=r[5];
    const double xx=x1-x2, yy=y1-y2, zz=z1-z2;
    r1 = sqrt(x1*x1 + y1*y1 + z1*z1);
    r2 = sqrt(x2*x2 + y2*y2 + z2*z2);
    r12 = sqrt(xx*xx + yy*yy + zz*zz);
}



static double V(const coord_6d& r) {
    double r1, r2, r12;
    distances(r, r1, r2, r12);
    if (vi==0) {
	return -2.0*u(r1,rcut) - 2.0*u(r2,rcut) + 1.0*u(r12,r12cut);
    } else if (vi==1) {
	return -2.0*u(r1,rcut) - 2.0*u(r2,rcut) ;
    } else if (vi==2) {
	return  1.0*u(r12,r12cut);
    } else if (vi==3) {		// for Vpsi
	return  1.0*u(r12,r12cut);
    } else {
    	MADNESS_EXCEPTION("unknown vi",0);
	return 0.0;
    }
}


struct LBCost {
    double leaf_value;
    double parent_value;
    LBCost(double leaf_value=1.0, double parent_value=1.0)
        : leaf_value(leaf_value)
        , parent_value(parent_value)
    {}

    double operator()(const Key<6>& key, const FunctionNode<double,6>& node) const {
        if (key.level() <= 1) {
            return 100.0*(leaf_value+parent_value);
        }
        else if (node.is_leaf()) {
            return leaf_value;
        }
        else {
            return parent_value;
        }
    }
};


struct true_op {
    bool operator()(FunctionImpl<double,6>* impl, const Key<6>& key, const FunctionNode<double,6>& t) const {
        return true;
    }

    template <typename Archive> void serialize (Archive& ar) {}
};

static double f6d(const coord_6d& r) {
    double r1, r2, r12;
    distances(r, r1, r2, r12);
    r1 = sqrt(r1*r1 + dcut*dcut); // Smooth cusps just a little
    r2 = sqrt(r2*r2 + dcut*dcut);
    r12 = sqrt(r12*r12 + d12cut*d12cut);
    double val;
    if (oi==0) {		// McQuarrie
        val=exp(-(27.0/16.0)*(r1+r2));
    } else if (oi==1) {		// Hylleraas-3-term
	val=-exp(-1.81607*(r1 + r2)) * (
	-1.33083943395992
	-0.388320016632985 * r12
	-0.174093511691879 *  ( r1*r1  + r2*r2  -2.0 * r1 * r2 )
	);
    } else if (oi==2) {		// Hylleraas-2-term
        val=exp(-1.8*(r1 + r2))*(1.0 + 0.5*r12);
    } else {
    	MADNESS_EXCEPTION("unknown oi",0);
    }
    return val;

}

/*
static double Vpsi(const coord_6d& r) {
    return f6d(r) * V(r);
}
*/

class YetAnotherWrapperClass {
    const real_function_6d& f;
    real_function_6d eri;
    const Tensor<double>& qx;

public:
    YetAnotherWrapperClass(const real_function_6d& f)
        : f(f)
		, eri(TwoElectronFactory(f.get_impl()->world).dcut(dcut*dcut).thresh(thresh*0.1).k(k))
//        , eri(ERIFactory(f.get_impl()->world).dcut(dcut*dcut).thresh(thresh*0.1).k(k))
        , qx(FunctionCommonData<double,6>::get(k).quad_x)
    {
//        eri=ERIFactory<double,6>(f.get_impl()->world).dcut(dcut*dcut).thresh(thresh).k(k);
    }

    void operator()(const Key<6>& key, Tensor<double>& t) const {
    	const long npt=k+3;
        Tensor<double> v(npt,npt,npt,npt,npt,npt);

        f.get_impl()->fcube(key, V, qx, v);
//        v+=f.get_impl()->coeffs2values(
//		key,eri.get_impl()->get_functor()->coeff(key).full_tensor());

        t.emul(v);

//        f.get_impl()->fcube(key, V, qx, v);
//        t=v;
    }
};

real_function_6d multiply_by_V(const real_function_6d& psi) {
    real_function_6d Vpsi = copy(psi);
    Vpsi.unaryop(YetAnotherWrapperClass(Vpsi));
    return Vpsi;
}

double energy(World& world, const real_function_6d& psi) {
    double kinetic_energy = 0.0, potential_energy = 0.0, overlap;
    for (int axis=0; axis<1; axis++) { // Note use of spherical symmetry
        real_derivative_6d D = free_space_derivative<double,6>(world, axis);
        real_function_6d dpsi = D(psi);
        double dd = 0.5*inner(dpsi,dpsi);
        kinetic_energy += dd;
    }
    kinetic_energy *= 6.0; // Spherical symmetry

    potential_energy = inner(psi,multiply_by_V(psi));
//    real_function_6d vpsi = real_factory_6d(world).f(Vpsi);
//    potential_energy = inner(psi,vpsi);

    overlap = inner(psi,psi);

    double total = (kinetic_energy  + potential_energy)/overlap;
    double size = psi.size();
    if (world.rank() == 0) {
        printf("        k %ld\n", k);
        printf("   thresh %.1e\n", thresh);
        printf("   t-mode %d\n", FunctionDefaults<6>::get_truncate_mode());
        printf("   t-on-p %d\n", FunctionDefaults<6>::get_truncate_on_project());
        printf("   smooth %.1e %.1e %.1e %.1e\n", rcut, r12cut, dcut, d12cut);
        printf("        L %12.6f\n", L);
        print ("   tensor ", FunctionDefaults<6>::get_tensor_type());
        printf("       oi %ld\n", oi);
        printf("       vi %ld\n", vi);
        printf("\n");
        printf("   #coeff %12.2e\n", size);
        printf("        S %12.6f\n", overlap);
        printf("       KE %12.6f\n", kinetic_energy/overlap);
        printf("       PE %12.6f\n", potential_energy/overlap);
        printf("    Total %12.6f\n", total);
    }

    return total;
}

int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());
    startup(world,argc,argv);
    std::cout.precision(6);

    FunctionDefaults<6>::set_k(k);
    FunctionDefaults<6>::set_thresh(thresh);
    FunctionDefaults<6>::set_truncate_mode(0);
    FunctionDefaults<6>::set_truncate_on_project(true);
    FunctionDefaults<6>::set_project_randomize(true);
    FunctionDefaults<6>::set_cubic_cell(-L/2,L/2);
    FunctionDefaults<3>::set_cubic_cell(-L/2,L/2);
    FunctionDefaults<6>::set_tensor_type(TT_2D);
//    FunctionDefaults<6>::set_tensor_type(TT_FULL);
    
    if (world.rank() == 0) printf("compressing PSI at time %.1f\n", wall_time());
    real_function_6d psi = real_factory_6d(world).f(f6d);
//    psi.refine_general(true_op());
    if (world.rank() == 0) printf("computing norm at time %.1f\n", wall_time());
    if (world.rank() == 0) printf("load balancing using psi at time %.1f\n", wall_time());
    LoadBalanceDeux<6> lb(world);
    lb.add_tree(psi,LBCost(1.0,1.0));
    FunctionDefaults<6>::redistribute(world, lb.load_balance(2.0,false));


//     if (world.rank() == 0) print("compressing V");
//     real_function_6d potn = real_factory_6d(world).f(V);


    if (world.rank() == 0) printf("computing energy at time %.1f\n", wall_time());
    energy(world, psi);

    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
    finalize();
    return 0;
}
