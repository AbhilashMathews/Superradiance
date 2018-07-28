# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 13:34:46 2016

@author: Abhilash
"""

""" Superradiance code employing a 4th order Runge-Kutta method: forward z, forward t.
    This is similar to the method of lines and only valid if no backward 
    propagating waves are considered. This code uses the slowly varying envelope 
    approximation for the electric dipole case when focusing on the 1720 MHz transition
    in OH and all parameters are in SI units and variables are scaled as per: 
    E' = (d*TR/hbar)E, N' = (V/N)N, P' = (V/Nd)P, t' = t/TR, z' = z."""

import matplotlib #no display environment on SHARCNET so need to import
matplotlib.use('pdf') #must also comment out plt.show() in SHARCNET
import os #this line and the two above are used so compatible with SHARCNET 
matplotlib.use('pdf') 
from random import random, randint, uniform 
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from tqdm import tqdm #must import this non-standard package
          
#constants & parameters
omega = 2.*np.pi*(1720.5299*(10.**(6.)))
eps = 8.85*10.**(-12.)
c = 3.*(10.**8.)
k0 = omega/c
hbar = (6.626*10.**(-34.))/(2.*np.pi)
n_mean = (10.**(5.)) #mean inverted population density (in molecules/m^3)
lambdaOH = c/(1720.5299*(10.**(6.))) #wavelength
T_1_ = 9.349*(10.**(-12.))# = gamma 
Tsp = 1./T_1_
TR_mean = 1.0*(10.**(-6.))#mean superradiance characteristic time
L_mean = (Tsp/TR_mean)*(8.*np.pi)/(3.*n_mean*(lambdaOH**2.))
radius_mean = np.sqrt(lambdaOH*L_mean/np.pi)
A_mean = np.pi*(radius_mean**2.)
phi_diffraction_mean = (lambdaOH**2.)/A_mean # (15) from Dicke superradiance II paper (Rajabi & Houde, 2016)
V_mean = (np.pi*(radius_mean**2.))*L_mean
NN_mean = n_mean*V_mean #total inverted population
d = (3.*eps*c*hbar*(lambdaOH**2.)*T_1_/(4.*np.pi*omega))**0.5  #dipole transition element
Lp = 1. #scaling of z-axis   
#z-space
Ngridz = 1000 #number of steps in z-axis
zmax = (L_mean/Lp)
dz = zmax/Ngridz 
z = np.linspace(0.0,zmax,Ngridz+1) 
#time
Ngridt = 1000 #number of steps in t-axis; should be larger than tmax (i.e. tmax/TR if tmax in SI units)
tmax = 1000. #time limit; is actually tmax*TR_mean in SI units (re-scaled later in code)
dt = tmax/Ngridt #this should be less than or equal to TR
t = np.linspace(0.0,tmax,Ngridt+1) #this is tau/TR; len(t) = interval

#arrays storing values for each single realization              
Ep = np.zeros(((Ngridz+1),len(t)),dtype=np.complex_) #number of rows = len(z)
Pp = np.zeros(((Ngridz+1),len(t)),dtype=np.complex_)
N = np.zeros(((Ngridz+1),len(t)),dtype=np.complex_) #using np.complex_ or complex seems to work; check higher precision arithmetic packages

#array storing values for different realizations by summing from all realizations 
#since the t-axis is scaled by TR_mean for all realizations, they can be added together in time
#although if L is changing, then only intensities at end of sample for each realization should be added   
IntensitySUM = np.zeros(((Ngridz+1),len(t)),dtype=np.complex_)

T_time_scale = 0.5 #uncommenting this line and lines 73-75 and lines 79-80 
#allows for arbitrary "additional pump(s)" that can be arbitrarily added
#to arrive and restore N and P as in lines 73-75 and lines 79-80, respectively
def dN(N,P,E,k,i,kN,kP,kE,TR,T1):
    kNN = ((1j)*((P[k,i]+kP)*(E[k,i]) - np.conj(E[k,i])*np.conj(P[k,i]+kP)) \
        - (N[k,i]+kN)/(T1/TR_mean)) #this must be TR_mean if constant time-axis used
#    if (i >= int(0.2*Ngridt)) and (i <= int(0.21*Ngridt)):
#    if i == int(T_time_scale*Ngridt):
#        kNN = 0.7*(0.5*np.cos(theta0) - N[k,i])
    return kNN
def dP(N,P,E,k,i,kN,kP,kE,TR,T2):      
    kPP = ((2j)*(np.conj(E[k,i]))*(N[k,i]+kN) - (P[k,i]+kP)/(T2/TR_mean)) #this must be TR_mean if constant time-axis used
#    if (i >= int(0.2*Ngridt)) and (i <= int(0.21*Ngridt)):
#    if i == int(T_time_scalen(theta0) - P[k,i])
    return kPP
def dE(N,P,E,k,i,kN,kP,kE,constant3,Ldiff,Lp):
    kEE = ((1j*constant3)*np.conj(P[k,i]) - (E[k,i]+kE)/(Ldiff/Lp)) 
    return kEE

#used to hold values for different realizations for plotting at end of code, if desired   
TR_list = []

cycle_STOP = 100 #number of cycles/superradiant realizations

#this provides a progress bar but only updates per cycle
#i.e. if only doing one cycle, will not see progress update
#although can manually change this easily so that it updates per gridstep
progress = tqdm(total=100.)
total = 100.
progress_cycles = (total)/(cycle_STOP)

#these arrays store values strictly at z = L
TOTAL_E = np.zeros((cycle_STOP+1,len(t)),dtype=np.complex_)
TOTAL_I = np.zeros((cycle_STOP+1,len(t)))
TOTAL_N = np.zeros((cycle_STOP+1,len(t)),dtype=np.complex_)
TOTAL_P = np.zeros((cycle_STOP+1,len(t)),dtype=np.complex_)

cycles = 0
while cycles < cycle_STOP:
    mu1 = TR_mean
    sigma1 = 0.0000000001*mu1 #standard deviation of T_R
    TR = np.random.normal(mu1, sigma1) #Gaussian distributed TR
    TR_list.append(TR) 
    T1 = 10000.*(10.**(-6.)) #relaxation time - SI units
    T2 = 10000.*(10.**(-6.)) #dephasing time - SI units
    n_actual = (Tsp/TR)*(8.*np.pi)/(3.*L_mean*(lambdaOH**2.)) #assuming constant length although can remove
    #this assumption my using different value for "L_mean"
    NN = n_actual*V_mean #total inverted population for this realization
    Fn = 1. # = (radius**2.)/(lambdaOH*L) 
    Ldiff = Fn*L_mean/0.35
    constant3 = (omega*TR_mean*NN*(d**2.))*Lp/(2.*c*hbar*eps*V_mean) #the TR_mean has to be the outside one
    #since it is used for scaling of t-axis; only NN varies here
    
    #this simply makes entries for arrays all zeros again
    EpM = Ep
    PpM = Pp
    NM = N
    
    theta0_mean = 2./np.sqrt(NN) #initial Bloch angle = 2/np.sqrt(NN)
    sigma_theta0 = 0.0001*theta0_mean
    SIGMA_T0 = int(sigma_theta0/theta0_mean) #Gaussian distributed theta0
    theta0_k = (np.random.normal(theta0_mean, sigma_theta0))
    theta0_i = theta0_k#theta_mean#np.abs(np.random.normal(theta0_mean, sigma_theta0))

    theta0 = 2./np.sqrt(NN)
    TD = 0.25*(np.log(theta0/(2.*np.pi)))**2.#*TR
    T0 = np.log(theta0/2.) #pg.30 Benedict et al. discuss area of initial pulse
    if cycle_STOP == 1:
        cycles = cycles + 1
    progress.update(progress_cycles)
    k = 0
    while k < (len(z) - 1):
    
#----------------calculating NE, PpE, and EpE----------------

        hz = z[k+1]-z[k] #hz is positive

        i = 0
        while i < (len(t) - 1):
            ht = t[i+1] - t[i]

#        Initial conditions   
            PpM[k,0] = 0.5*np.sin(theta0_k)#*np.sin(z[:]/(0.001*Lp))#(np.e**((-(z[:])**2.)/(0.001*L_mean)**2.))
            NM[k,0] = 0.5*np.cos(theta0_k)#the above lines commented out is just one way to alter initial conditions
            if i == 0:  
                EpM[k,0] = 0.
#        Boundary conditions
            EpM[0,i] = 0. #E boundary condition
            if k == 0:
                PpM[0,i] = 0.5*np.sin(theta0_i)*(np.e**(-t[i]*TR_mean/T2))
                NM[0,i] = 0.5*np.cos(theta0_i)*(np.e**(-t[i]*TR_mean/T1))
                
#           4th-order Runge-Kutta
#            in time
            kEb1 = dE(NM,PpM,EpM,k,i,0.,0.,0.,constant3,Ldiff,Lp)
            kEb2 = dE(NM,PpM,EpM,k,i,0.,0.,0.5*hz*kEb1,constant3,Ldiff,Lp)
            kEb3 = dE(NM,PpM,EpM,k,i,0.,0.,0.5*hz*kEb2,constant3,Ldiff,Lp)
            kEb4 = dE(NM,PpM,EpM,k,i,0.,0.,hz*kEb3,constant3,Ldiff,Lp)
#            in z
            kNb1 = dN(NM,PpM,EpM,k,i,0.,0.,0.,TR,T1)
            kPb1 = dP(NM,PpM,EpM,k,i,0.,0.,0.,TR,T2)
            kNb2 = dN(NM,PpM,EpM,k,i,0.5*ht*kNb1,0.5*ht*kPb1,0.,TR,T1)
            kPb2 = dP(NM,PpM,EpM,k,i,0.5*ht*kNb1,0.5*ht*kPb1,0.,TR,T2)
            kNb3 = dN(NM,PpM,EpM,k,i,0.5*ht*kNb2,0.5*ht*kPb2,0.,TR,T1)
            kPb3 = dP(NM,PpM,EpM,k,i,0.5*ht*kNb2,0.5*ht*kPb2,0.,TR,T2)
            kNb4 = dN(NM,PpM,EpM,k,i,ht*kNb3,ht*kPb3,0.,TR,T1)
            kPb4 = dP(NM,PpM,EpM,k,i,ht*kNb3,ht*kPb3,0.,TR,T2)
        
            EpM[k+1,i] = EpM[k,i] + (hz/6.)*(kEb1 + 2.*kEb2 + 2.*kEb3 + kEb4)
            NM[k,i+1] = NM[k,i] + (ht/6.)*(kNb1 + 2.*kNb2 + 2.*kNb3 + kNb4)
            PpM[k,i+1] = PpM[k,i] + (ht/6.)*(kPb1 + 2.*kPb2 + 2.*kPb3 + kPb4)
        
#        Initial conditions   
            PpM[k,0] = 0.5*np.sin(theta0_k)#*np.sin(z[:]/(0.001*Lp))#(np.e**((-(z[:])**2.)/(0.001*L_mean)**2.))
            NM[k,0] = 0.5*np.cos(theta0_k)#the above lines commented out is just one way to alter initial conditions
            if i == 0:  
                EpM[k,0] = 0.
#        Boundary conditions
            EpM[0,i] = 0. #E boundary condition
            if k == 0:
                PpM[0,i] = 0.5*np.sin(theta0_i)*(np.e**(-t[i]*TR_mean/T2))
                NM[0,i] = 0.5*np.cos(theta0_i)*(np.e**(-t[i]*TR_mean/T1))
                
            i = i + 1
        k = k + 1
    
        if k == (len(z)-1): #this is only because i dont have extra iteration for N and P in z
            if i == (len(t)-1): #and this smooths N, P, and E at grid boundaries
                EpM[:,i] = EpM[:,i-1] #for E at at end of time
                NM[k,:] = NM[k-1,:] #for N and P at end of sample 
                PpM[k,:] = PpM[k-1,:] #i.e. at end of z-axis
                
    #index for the below arrays corresponds to which superradiant sample it is (i.e. which cycle)
    #and only stores the values at the end of the sample
                Ep_FINAL = ((EpM[int(Ngridz-1.),:])/(d*TR_mean/hbar)) #the electric field is in SI units
                I_SR = ((0.5*c*eps*((Ep_FINAL*np.conj(Ep_FINAL)))).real) # = Itot
                TOTAL_E[cycles] = Ep_FINAL 
                TOTAL_I[cycles] = I_SR
                TOTAL_N[cycles] = NM[int(Ngridz-1.),:] #still scaled by N/V
                TOTAL_P[cycles] = PpM[int(Ngridz-1.),:] #still scaled by d*N/V
                IntensitySUM = IntensitySUM + ((0.5*c*eps*(((EpM/(d*TR_mean/hbar))*np.conj((EpM/(d*TR_mean/hbar)))))).real)
    
    #could technically start a second loop, and even more, to continue
    #finding solution inside the boundary of the grid to ensure an
    #"stable" solution is reached; unless pathological boundary/initial conditions are applied
    #all solutions should be stable (as observed when tested); increasing the number of grid points is analogous
    #although this alternative would allow for E to have an updated dependence on N and P 
    #as they vary in z, too, since this is missing in its RK4 step
    cycles = cycles + 1
progress.close()

I_nc = NN_mean*hbar*omega*(1./(A_mean*Tsp))*(phi_diffraction_mean/(4.*np.pi))
#the above and below lines should be equivalent expressions
I_nc = (2./3.)*hbar*omega/(A_mean*TR_mean) #should technically be multiplied by "cycles" since there are that many superradiant samples
IntensityRatio = IntensitySUM.real/(NN_mean*I_nc) #summed intensity for all z scaled by N*I_n

newpath = r'OH_1720MHz_SVEA_superradiance_theta0_sigma='+str(SIGMA_T0)+'_TR='+str(TR_mean)+'_n='+str(n_mean)+'_tmax=TR*'+str(tmax)+'_T1='+str(T1)+'_T2='+str(T2)+'_L='+str(L_mean)+'_zmax='+str(zmax)+'_Ngridz='+str(Ngridz)+'_Ngridt='+str(Ngridt)+'_cycles='+str(cycles)+'_T_time_scale='+str(T_time_scale)
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
z = z*Lp #ensures the axis scales are in SI units
t = t*TR_mean

j = 0
IntensityFINAL = np.zeros(len(t))
while j < len(TOTAL_I[:,0]):
    IntensityFINAL += TOTAL_I[j,:] 
    j = j + 1
#IntensityRatio = ((0.5*c*eps*((E_SR*np.conj(E_SR)))).real)#TOTAL_E

t = TR_mean*np.linspace(0.0,tmax,Ngridt+1)

plt.figure(1)
#histogram of TR for different realizations
n, bins, patches = plt.hist(TR_list, 50, normed=1, facecolor='green', alpha=0.75)
# add a 'best fit' line
y = mlab.normpdf( bins, mu1, sigma1)
plt.xlabel(r"$ \tau (s) $")
plt.ylabel('Relative Frequency of T_R (not normalized)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
l = plt.plot(bins, y, 'r--', linewidth=1) 
plt.savefig(str(newpath)+'/Histogram_TR.pdf')
#plt.show()

plt.figure(2)
fig, ax = plt.subplots(figsize=(20, 20))
plt.subplot(221)
plt.plot(t,IntensityRatio[0,:],'k:',linewidth = 1.5)
plt.ylabel(r"$ \frac {I_{SR}}{NI_{nc}} $")
plt.subplot(222)
plt.plot(t,IntensityRatio[int(0.33*len(z)),:],'k:',linewidth = 1.5)
plt.subplot(223)
plt.plot(t,IntensityRatio[int(0.66*len(z)),:],'k:',linewidth = 1.5)
plt.subplot(224)
plt.plot(t,IntensityRatio[len(z)-1,:],'k:',linewidth = 1.5)
plt.xlabel(r"$ \tau (s)$")
plt.savefig(str(newpath)+'/I_vs_t.pdf')
#plt.show()

plt.figure(3)
fig, ax = plt.subplots(figsize=(20, 20))
plt.subplot(221)
plt.plot(z,IntensityRatio[:,0],'k:',linewidth = 1.5)
plt.ylabel(r"$ \frac {I_{SR}}{NI_{nc}}$")
plt.subplot(222)
plt.plot(z,IntensityRatio[:,int(0.33*len(t))],'k:',linewidth = 1.5)
plt.subplot(223)
plt.plot(z,IntensityRatio[:,int(0.66*len(t))],'k:',linewidth = 1.5)
plt.subplot(224)
plt.plot(z,IntensityRatio[:,len(t)-1],'k:',linewidth = 1.5)
plt.xlabel('z (m)')
plt.savefig(str(newpath)+'/I_vs_z.pdf')
#plt.show()
