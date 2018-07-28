#include <iostream>
#include <stdio.h>
#include <fstream>
#include <complex>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
#include <map>
#include <mpi.h>
//#include </opt/intel/compilers_and_libraries_2017.1.126/mac/mkl/include/mkl.h>
#include <mkl.h>
using namespace std;

//Superradiance

//4th order Runge-Kutta; forward z, forward t

//constants & parameters
double omega = 2.*M_PI*(1720.5299*pow(10.,6.));   //(10.**(6.)))
double eps = 8.85*pow(10.,-12.);     //10.**(-12.)
double c = 3.*pow(10.,8.); //(10.**8.)
double k0 = omega/c;
double hbar = (6.626*pow(10.,-34.))/(2.*M_PI);
double n_mean = pow(10.,5.); //mean inverted population density (in molecules/m^3)
//this density is used for all samples
double lambdaOH = c/(1720.5299*pow(10.,6.));  //wavelength
double T_1_ = 9.349*pow(10.,-12.); // = gamma
//T_1_ = (omega**3.)*(d**2.)/(3.*np.pi*eps*hbar*(c**3.))
//DIPOLE = 0.5*d/((eps*np.pi)**0.5)
double Tsp = 1./T_1_;
double TR_mean = 0.1*pow(10.,-6.); //mean superradiance characteristic time
double T1_mean = 100000.*pow(10.,-6.);//100.*TR_mean;
double T2_mean = 100000.*pow(10.,-6.); // #dephasing timescale
double L_mean = (Tsp/TR_mean)*(8.*M_PI)/(3.*n_mean*(pow(lambdaOH,2.)));
double radius_mean = sqrt(lambdaOH*L_mean/M_PI);
double A_mean = M_PI*(pow(radius_mean,2.));
double phi_diffraction_mean = pow(lambdaOH,2.)/A_mean;
double V_mean = (M_PI*pow(radius_mean,2.))*L_mean;
double NN_mean = n_mean*V_mean; //total inverted population
double d = pow((3.*eps*c*hbar*pow(lambdaOH,2.)*T_1_/(4.*M_PI*omega)),0.5);  //dipole transition element
const complex<double> Im1(0.0,1.0); //imaginary number definition

int Ngridz = 5000;
double *z = new double[Ngridz];

int Ngridt = 5000;;
double *t = new double[Ngridt];
double tmax = 5000.;
double dt = tmax/double(Ngridt);

//complex<int> **ary = new complex<int>*[Ngridt];
//run loop to initialize
//for (int i = 0; i < Ngridt; ++i)
//{
//    ary[i] = new complex<int>[Ngridz];
//}
    
//complex<double> *Ep = new complex<double>[Ngridz * Ngridt];
//complex<double> *Pp = new complex<double>[Ngridz * Ngridt];
//complex<double> *N = new complex<double>[Ngridz * Ngridt];
//access with:
//ary[y*sizeX + x]
//assign with
//ary[y*sizeX+x] = complex<int>(int,int);
//clean up
//delete[] ary;
//access with
//ary[i][j];
//assign index with
//ary[i][j] = complex<int>(int,int);

void printProgBar( double percent, int i , int total){
    std::string bar;
    
    if(total < 100){
        total = 100; //just to stop error from popping up if testing a low number of cycles
    }
    
    if( i == 0){
        bar = "[" ;
        std::cout<< bar ;
    }
    else if( i == (total-1) ){
        bar = "]" ;
        std::cout<< bar ;
    }
    else if( ((int(i))%(int(0.1*total))) == 0){//only really works if cycle_STOP grater than 100 and divisble by 10
        std::cout << (i*100.0/(total*1.0)) << "% complete\n";
    }
    else{
        bar = ">" ;
    }
//    std::cout.width( 3 );
}

double T_time_scale = 0.3;

struct Functions{
    static complex<double> dN(complex<double> **N, complex<double> **P, complex<double> **E, int k, int i, complex<double> kN, complex<double> kP, complex<double> kE, double T1, double theta0_used){
//        if (i > int(T_time_scale*Ngridt))
//            T1 = 1000000.00000000000001;
        complex<double> kNN = (Im1*((P[k][i]+kP)*(E[k][i]) - conj(E[k][i])*conj(P[k][i]+kP)) - (N[k][i]+kN)/(T1/TR_mean));
//        if (i == int(T_time_scale*Ngridt))
//            kNN = 0.18*(0.5*cos(theta0_used) - N[k][i]);
        return kNN;
    };
    
    static complex<double> dP(complex<double> **N, complex<double> **P, complex<double> **E, int k, int i, complex<double> kN, complex<double> kP, complex<double> kE, double T2, double theta0_used){
//        if (i > int(T_time_scale*Ngridt))
//            T2 = 1000000.00000000000001;
        complex<double> kPP = ((2.*Im1)*(conj(E[k][i]))*(N[k][i]+kN) - (P[k][i]+kP)/(T2/TR_mean));
//        if (i == int(T_time_scale*Ngridt))
//            kPP = 1.0*(0.5*sin(theta0_used) - P[k][i]);
        return kPP;
    };
    
    static complex<double> dE(complex<double> **N, complex<double> **P, complex<double> **E, int k, int i, complex<double> kN, complex<double> kP, complex<double> kE, double constant3, double Ldiff, double Lp){
        complex<double> kEE = ((Im1*constant3)*conj(P[k][i]) - (E[k][i]+kE)/(Ldiff/Lp));
        return kEE;
    };
};



int main(int argc, const char * argv[]) {
    
    complex<double> **EpM = new complex<double>*[Ngridz];
    //run loop to initialize
    for (int i = 0; i < Ngridz; ++i)
    {
        EpM[i] = new complex<double>[Ngridt];
    }
    
    complex<double> **PpM = new complex<double>*[Ngridz];
    //run loop to initialize
    for (int i = 0; i < Ngridz; ++i)
    {
        PpM[i] = new complex<double>[Ngridt];
    }
    
    complex<double> **NM = new complex<double>*[Ngridz];
    //run loop to initialize
    for (int i = 0; i < Ngridz; ++i)
    {
        NM[i] = new complex<double>[Ngridt];
    }
    //code to plot histogram of the Gaussian distributed TR
    
    int cycle_STOP = 1; //number of cycles/superradiant samples being used
    double *TR_list = new double[cycle_STOP];
    double *n_list = new double[cycle_STOP];
    double *T1_list = new double[cycle_STOP];
    double *T2_list = new double[cycle_STOP];
    int cycles = 0;
    double *INTENSITY_T = new double[Ngridt];
    double *EpM_T = new double[Ngridt];
    double *PpM_T = new double[Ngridt];
    double *NM_T = new double[Ngridt];
    while (cycles<cycle_STOP){
        
        //        std::default_random_engine generator;
        //        std::normal_distribution<double> distribution(mu1,sigma1);
        //        double TR = distribution(generator);
        
        double mu1 = TR_mean;
        double sigma1 = 0.001*mu1;
        std::random_device rd;
        std::mt19937 e2(rd());
        std::normal_distribution<> dist(mu1, sigma1);
        double TR_actual = abs((dist(e2)));
        TR_list[cycles] = TR_actual;
        
        double mu_n = n_mean;
        double sigma_n = 0.001*mu_n;
        std::random_device rd1;
        std::mt19937 e12(rd1());
        std::normal_distribution<> dist1(mu_n, sigma_n);
        double n_actual = abs((dist1(e12)));
        n_list[cycles] = n_actual;
        
        double mu_T1 = T1_mean;
        double sigma_T1 = 0.000000001*mu_T1;
        std::random_device rd2;
        std::mt19937 e12T1(rd2());
        std::normal_distribution<> distT1(mu_T1, sigma_T1);
        double T1_actual = abs((distT1(e12T1)));
        T1_list[cycles] = T1_actual;
        
        double mu_T2 = T2_mean;
        double sigma_T2 = 0.000000001*mu_T2;
        std::random_device rd22;
        std::mt19937 e12T2(rd22());
        std::normal_distribution<> distT2(mu_T2, sigma_T2);
        double T2_actual = abs((distT2(e12T2)));
        T2_list[cycles] = T2_actual;
        
        //plotting histogram of Gaussian generator
        //            std::map<int, int> hist;
        //            for (int n = 0; n < 100000; ++n) {
        //                cout << (dist(e2)) << "\n";
        //                ++hist[std::round(dist(e2))];
        //            }
        //
        //            for (auto p : hist) {
        //                std::cout << std::fixed << std::setprecision(1) << std::setw(2)
        //                << p.first << ' ' << std::string(p.second/200, '*') << '\n';
        //            }
        
        double L_actual = (Tsp/TR_actual)*(8.*M_PI)/(3.*n_actual*(lambdaOH*lambdaOH));
        double Lp = 1.;//L_mean; //scaling of z-axis
        double zmax = (L_actual/1.); //scaling is 1. to 1. for z-axis
        //lengths can be different but the t-axis needs to be constant!!!!! although TR varies!
        double dz = zmax/double(Ngridz);
//        double zmax = (L/Lp);
//        double dz = zmax/double(Ngridz);
//        double *z = new double[Ngridz];
        double radius = pow((lambdaOH*L_actual/M_PI),0.5);
        double A = M_PI*(radius*radius);
        double phi_diffraction = (lambdaOH*lambdaOH)/A;
        double V = (M_PI*(radius*radius))*L_actual;
        double NN = n_actual*V; //total inverted population;
        double Fn = 1.; //0.025 # Fresnel number = A/(lambdaOH*L)
//        double Lp = L_mean; // this is L': scaling factor for length; z' = z/L' (z' is domain used)
        double Ldiff = Fn*L_actual/0.35; //estimated phenomenological diffraction lost term
        double constant3 = (omega*TR_mean*NN*(d*d))*Lp/(2.*c*hbar*eps*V);
        
        double theta0_act = 2./pow(NN,0.5); //initial Bloch angle = 2/np.sqrt(NN)
        double TD = 0.25*(log(theta0_act/(2.*M_PI)))*(log(theta0_act/(2.*M_PI)));//*TR
        double T0 = log(theta0_act/2.); //pg.30 Benedict et al. discuss area of initial pulse
        
        int k = 0;
//        cout << k;
        while (k < (Ngridz - 1)){
            //not making theta0 fluctuate for every point since evolution largely the same
            //if sigma_theta0 is small enough...
            PpM[k][0] = 0.5*sin(theta0_act); //*(np.e**((-(z-0.5*L)**2)/L**2))
            NM[k][0] = 0.5*cos(theta0_act); //*(np.e**((-(z-0.5*L)**2)/L**2))
            z[k+1] = z[k] + dz;
            double hz = z[k+1]-z[k];//hz is positive ; reintroducing here in case I want to use adaptive mesh in future
            
            int i = 0;
            while (i < (Ngridt - 1)){
                t[i+1] = t[i] + dt;
                double ht = t[i+1] - t[i];
                
                //        Initial condition
                
                EpM[0][i] = 0.; //E_sp2 #E boundary condition
                
                if (k == 0){
                    PpM[0][i] = 0.5*sin(theta0_act)*(exp(-t[i]*TR_mean/T2_actual));
                    NM[0][i] = 0.5*cos(theta0_act)*(exp(-t[i]*TR_mean/T1_actual));
                }
                if (i == 0){
                    EpM[k][0] = 0.;
                }
                
                complex<double> kEb1 = Functions::dE(NM,PpM,EpM,k,i,0.,0.,0.,constant3,Ldiff,Lp);
                complex<double> kEb2 = Functions::dE(NM,PpM,EpM,k,i,0.,0.,0.5*hz*kEb1,constant3,Ldiff,Lp);
                complex<double> kEb3 = Functions::dE(NM,PpM,EpM,k,i,0.,0.,0.5*hz*kEb2,constant3,Ldiff,Lp);
                complex<double> kEb4 = Functions::dE(NM,PpM,EpM,k,i,0.,0.,hz*kEb3,constant3,Ldiff,Lp);
                
                complex<double> kNb1 = Functions::dN(NM,PpM,EpM,k,i,0.,0.,0.,T1_actual,theta0_act);
                complex<double> kPb1 = Functions::dP(NM,PpM,EpM,k,i,0.,0.,0.,T2_actual,theta0_act);
                complex<double> kNb2 = Functions::dN(NM,PpM,EpM,k,i,0.5*ht*kNb1,0.5*ht*kPb1,0.,T1_actual,theta0_act);
                complex<double> kPb2 = Functions::dP(NM,PpM,EpM,k,i,0.5*ht*kNb1,0.5*ht*kPb1,0.,T2_actual,theta0_act);
                complex<double> kNb3 = Functions::dN(NM,PpM,EpM,k,i,0.5*ht*kNb2,0.5*ht*kPb2,0.,T1_actual,theta0_act);
                complex<double> kPb3 = Functions::dP(NM,PpM,EpM,k,i,0.5*ht*kNb2,0.5*ht*kPb2,0.,T2_actual,theta0_act);
                complex<double> kNb4 = Functions::dN(NM,PpM,EpM,k,i,ht*kNb3,ht*kPb3,0.,T1_actual,theta0_act);
                complex<double> kPb4 = Functions::dP(NM,PpM,EpM,k,i,ht*kNb3,ht*kPb3,0.,T2_actual,theta0_act);
                
                EpM[k+1][i] = EpM[k][i] + (hz/6.)*(kEb1 + 2.*kEb2 + 2.*kEb3 + kEb4);
                NM[k][i+1] = NM[k][i] + (ht/6.)*(kNb1 + 2.*kNb2 + 2.*kNb3 + kNb4);
                PpM[k][i+1] = PpM[k][i] + (ht/6.)*(kPb1 + 2.*kPb2 + 2.*kPb3 + kPb4);
                
//                cout << kEb4 << "\n";
                
                if (k == 0){
                    PpM[0][i] = 0.5*sin(theta0_act)*(exp(-t[i]*TR_mean/T2_actual));
                    NM[0][i] = 0.5*cos(theta0_act)*(exp(-t[i]*TR_mean/T1_actual));
                }
                
                if (i == 0){
                    EpM[k][0] = 0.;
                }
                
//                cout << k << "\n" << i << "\n" << EpM[k][i] << "\n" << kEb1 << "\n";
                
                i = i + 1;
            }
            
            k = k + 1;
        }
        
//        cout << "constant3 = " << constant3 << "\n L_actual = " << L_actual;
        
//        (omega*TR*NN*(d*d))*Lp/(2.*c*hbar*eps*V);
        int jj_z,ii_t;
        for(jj_z=(Ngridz-2);jj_z<(Ngridz-1);jj_z++) //does not quite end properly
        {
            for (ii_t=0; ii_t<(Ngridt-1); ii_t++) { //does not quite end properly
                
                INTENSITY_T[ii_t] = INTENSITY_T[ii_t] + real(0.5*c*eps*(EpM[jj_z][ii_t])*conj(EpM[jj_z][ii_t]))/(((d*TR_mean)/hbar)*((d*TR_mean)/hbar));
                EpM_T[ii_t] = imag(EpM_T[ii_t] + EpM[jj_z][ii_t]);
                PpM_T[ii_t] = real(PpM_T[ii_t] + PpM[jj_z][ii_t]);
                NM_T[ii_t] = real(NM_T[ii_t] + NM[jj_z][ii_t]);
                //only works if Ngridz == Ngridt; otherwise need to include
                // writes final line of z at end to output file
            }
        }
        cycles = cycles + 1; //should perhaps put this at end of loop
        double percent = 100.*double(cycles)/double(cycle_STOP);
        printProgBar(percent, cycles, cycle_STOP);
//        cout << "\n";
    }
    
    //Writing the above complex array into .txt file
    FILE *fp;
    
    ofstream outfile ("/Users/Abhilash/Desktop/test.txt");
    fp=fopen("/Users/Abhilash/Desktop/test.txt", "w");
    fprintf(fp,"time\tz\tE\tN\tP\n");
    int jj,ii;
    for(jj=(Ngridz-2);jj<(Ngridz-1);jj++) //does not quite end properly
    {
        for (ii=0; ii<(Ngridt-1); ii++) { //does not quite end properly
            //                fprintf(fp,"%g+%gi\t",real(EpM[jj][ii]),imag(EpM[jj][ii]));
            fprintf(fp,"%.10f\t%le\t%le\t%le\t%le\t%le\n",t[ii]*TR_mean,z[ii],EpM_T[ii],NM_T[ii],PpM_T[ii],INTENSITY_T[ii]); //only works if Ngridz == Ngridt; otherwise need to include
//            cout << t[ii]*TR_mean << "\n";
            // writes final line of z at end to output file
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
    cout << "DONE";
    return 0;
};
