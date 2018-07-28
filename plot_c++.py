#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 01:46:48 2017

@author: Abhilash
"""
import os #these 3 lines used so compatible with SHARCNET
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange 
from scipy.fftpack import rfftfreq, rfft
from pylab import *
from random import random, randint, uniform
import matplotlib as mpl
matplotlib.use('pdf') #must also comment out plt.show()

omega = 2.*np.pi*(1720.5299*(10.**(6.)))
eps = 8.85*10.**(-12.)
c = 3.*(10.**8.)
k0 = omega/c
hbar = (6.626*10.**(-34.))/(2.*np.pi)
n_mean = (10.**(5.)) #mean inverted population density (in molecules/m^3)
#this density is used for all samples
lambdaOH = c/(1720.5299*(10.**(6.))) #wavelength
T_1_ = 9.349*(10.**(-12.))#1.56*(10.**(-9.)) # = gamma
#T_1_ = (omega**3.)*(d**2.)/(3.*np.pi*eps*hbar*(c**3.))
#DIPOLE = 0.5*d/((eps*np.pi)**0.5)
Tsp = 1./T_1_
TR_mean = 0.1*(10.**(-6.))#1.1*36. #mean superradiance characteristic time
TR = TR_mean
#TR = (Tsp*8.*np.pi/(n_mean*lambdaOH*A))
#F = (A/(2.*np.pi))/(L*lambdaOH)
L_mean = (Tsp/TR_mean)*(8.*np.pi)/(3.*n_mean*(lambdaOH**2.))
radius_mean = np.sqrt(lambdaOH*L_mean/np.pi)
A_mean = np.pi*(radius_mean**2.)
phi_diffraction_mean = (lambdaOH**2.)/A_mean # (15) from OH paper
V_mean = (np.pi*(radius_mean**2.))*L_mean #these mean spatial values are constants in programs
#and are actuallly used
NN_mean = n_mean*V_mean #total inverted population
d = (3.*eps*c*hbar*(lambdaOH**2.)*T_1_/(4.*np.pi*omega))**0.5 
theta0 = 2./np.sqrt(NN_mean)
tau_D = TR*0.25*(np.abs(np.log(theta0/(2.*np.pi)))**2.)

output_file = open('/Users/Abhilash/Desktop/test.txt','r')
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()])
    
newpath = r'/Users/Abhilash/Desktop/sigma=0.1mean_newOH_multiple_cycles=500_different_t_reinvert_1720MHz_SVEA_superradiance_T1=0.04_T2=0.004_TR='+str(TR)+'_TR='+str(TR_mean)
if not os.path.exists(newpath):
    os.makedirs(newpath) 


t = output_data[::6]
z = output_data[1::6]
E = output_data[2::6]
N = output_data[3::6]
P = output_data[4::6]
IntensityFINAL = np.array(output_data[5::6])
x_TEST = t
y_TEST = IntensityFINAL

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.set_title("E vs t")    
ax1.set_xlabel(r"$ \tau (s)$")
ax1.set_ylabel('E.imag scaled')
ax1.plot(t,E, c='r', label='E at z = L')
leg = ax1.legend()
plt.show()

fig = plt.figure(2)
ax1 = fig.add_subplot(111)
ax1.set_title("N vs t")    
ax1.set_xlabel(r"$ \tau (s)$")
ax1.set_ylabel('N.real scaled')
ax1.plot(t,N, c='r', label='N at z = L')
leg = ax1.legend()
plt.show()

fig = plt.figure(3)
ax1 = fig.add_subplot(111)
ax1.set_title("P vs t")    
ax1.set_xlabel(r"$ \tau (s)$")
ax1.set_ylabel('P.real scaled')
ax1.plot(t,P, c='r', label='P at z = L')
leg = ax1.legend()
plt.show()

YY = np.array(E)/((d*TR_mean)/hbar)

fig = plt.figure(4)
ax1 = fig.add_subplot(111)
ax1.set_title("E vs t (SI units)")    
ax1.set_xlabel(r"$ \tau (s)$")
ax1.set_ylabel('E.imag (N/C)')
ax1.plot(t,YY, c='r', label='E at z = L')
leg = ax1.legend()
plt.show()

Ep_FINAL = YY
ht = 1. #scaled so it works in axis...
#IntensityFINAL = ((0.5*c*eps*((Ep_FINAL*np.conj(Ep_FINAL)))).real)
    
plt.figure(20)
plot(t,IntensityFINAL)
plt.xlabel(r"$ \tau (s) $")
plt.ylabel('Intensity (W)') 
#plt.savefig(str(newpath)+'/Intensity_vs_tau.pdf')
plt.show()

#lags1, c1, line1, b1 = acorr(((Ep_FINAL - np.mean(Ep_FINAL))), usevlines=False, normed=True, maxlags=1000, lw=2) 
#plt.figure(21)
#plt.plot(((lags1*(ht*TR_mean))),c1)
#plt.xlabel('t (s)')
#plt.savefig(str(newpath)+'/autocorrelation_of_E_z=L_time.pdf')
#plt.show()
#lags2, c2, line2, b2 = acorr((fft(c1)), usevlines=False, normed=True, maxlags=1000, lw=2)
#freqs = np.fft.fftfreq(len(c1), ht*TR_mean)
#plt.figure(22)
#plt.plot(((lags2*(freqs[2]-freqs[1]))),c2)
#plt.xlabel('f (Hz)')
##plt.xlim([-20000,20000])
#plt.savefig(str(newpath)+'/autocorrelation_of_fft_of_autocorrelation_of_E_z=L.pdf')
#plt.show()
#Y3 = (fft(c1))
#freqs = np.fft.fftfreq(len(c1), ht*TR_mean)
#plt.figure(23)
#plt.plot(freqs,Y3)
#plt.xlabel('f (Hz)')
##plt.xlim([-20000,20000])
#plt.savefig(str(newpath)+'/fft_of_autocorrelation_of_E_z=L_power_spectrum.pdf')
#plt.show()
#Ep_FINAL = YY
#Y4 = (fft(Ep_FINAL))
#freqs = np.fft.fftfreq(len(Ep_FINAL), ht*TR_mean)
#plt.figure(23)
#plt.plot(freqs,Y4)
#plt.xlabel('f (Hz)')
##plt.xlim([-20000,20000])
#plt.savefig(str(newpath)+'/fft_E_z=L.pdf')
#plt.show()


I_SR_peak = np.max(IntensityFINAL)#*2.*L_mean*n_mean*hbar*omega/(3.*TR)
dist = 10.**9.        # distance in parsec
phi_source = (A_mean/((3.09*(10.**16.))*dist))/((3.09*(10.**16.))*dist)
Int_flux = I_SR_peak*phi_source/phi_diffraction_mean #in W/m^2
Power_output_max = I_SR_peak*A_mean

mpl.rcParams['text.usetex'] = False  # not really needed
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18} 
#IntensityRatio1 = Int_flux*IntensityFINAL/np.max(IntensityFINAL)
#IntensityRatio1 = IntensityRatio1/(10.**(-26.)) #/(7000./131.)
#t1 = np.array(t) 
#t1 = t1*1000.
#plt.figure(24)
#plt.plot(t1,((IntensityRatio1[:]/0.01) + 1000.),'k',linewidth = 3.0)
#plt.xlabel("$ \\tau \ \mathrm{(ms)}$",**font)
#plt.ylabel(r"$ S_{\nu} \ \mathrm{(Jy)}$",**font)
#plt.xlim([0.,30.])
#plt.ylim([-0.25,1.5])
#plt.savefig(str(newpath)+'/Superradiance_2233.png')
#plt.show()

#t = np.array(output_data[::6])
#def spectrum(sig, t):
#    f = rfftfreq(sig.size, d=t[1]-t[0])
#    y = rfft(sig)
#    return f, y
#
#plt.figure(25)
#f, y = spectrum(Ep_FINAL, np.array(t))
#plt.subplot(211)
#plt.plot(np.array(t),Ep_FINAL)
#plt.grid(True)
#plt.subplot(212)
#y_a = np.abs(y)#/float(2.*len(y))#/N*2 # Scale accordingly
#plt.plot(f, y_a)
##plt.xlim([0.0,10000.])
#plt.grid(True)
#plt.show()
#
#Y10 = np.abs((fft(c1)))
#freqs = np.fft.fftfreq(len(c1), ht*TR_mean)
#plt.figure(26)
#plt.plot(freqs,Y10)
#plt.xlabel('f (Hz)')
##plt.xlim([-20000,20000])
#plt.savefig(str(newpath)+'/fft_of_autocorrelation_of_E_z=L_abs_power_spectrum.pdf')
#plt.show()

#X_new_t = lags1[(len(lags1)/2):(len(lags1))]
#Y_new_t = c1[(len(lags1)/2):(len(lags1))]
#plt.plot(X_new_t,Y_new_t)
#plt.show()
#Y11 = ((fft(Y_new_t)))
#freqs = np.fft.fftfreq(len(Y_new_t), ht*TR_mean)
#plt.figure(27)
#plt.plot(X_new_t,Y_new_t)
#plt.xlabel('f (Hz)')
#plt.xlim([-20000,20000])
#plt.savefig(str(newpath)+'/fft_of_autocorrelation__different_of_E_z=L_abs_power_spectrum.pdf')
#plt.show()
#
#Y_E = (fft(Ep_FINAL))
#freqs_E = np.fft.fftfreq(len(Ep_FINAL), ht*TR_mean)
#plt.plot(freqs_E,Y_E)
#plt.show()
#
#freqs_EE = freqs[0:(len(freqs_E)/2)]
#Y_EE = Y_E[0:(len(freqs_E)/2)]
#Y_EEE = Y_EE*np.conj(Y_EE)
#lagsE, cE, lineE, bE = acorr(((Y_EEE)), usevlines=False, normed=True, maxlags=1000, lw=2) 
#plt.figure(27)
#plt.plot(((lagsE*(freqs_EE[1]-freqs_EE[0]))),cE)
#plt.xlabel('F (Hz)')
#plt.xlim([-20000,20000])
#plt.savefig(str(newpath)+'/autocorrelation_of_E_z=L_time.pdf')
#plt.show()

#f, psd = welch(Ep_FINAL,fs=1./(t[1]-t[0]),  # sample rate
#    window='hanning',   # apply a Hanning window before taking the DFT
#    nperseg=256,        # compute periodograms of 256-long segments of x
#    detrend='constant')
#plt.plot(f,psd)

#T = 10
#f = 2
#f1 = 5
#dt = 0.001
#
#t = np.linspace(0, T, T/dt)
#x = 0.4 * np.cos(2*np.pi*f*t) + np.cos(2*np.pi*f1*t)
#
#
#from scipy.fftpack import fft, fftfreq
#import matplotlib.pyplot as plt
#
#X = fft(x)
#freq = fftfreq(x.size, d=dt)
#
## Only keep positive frequencies.
#keep = freq>=0
#X = X[keep]
#freq = freq[keep]
#
#ax1 = plt.subplot(111)
#ax1.plot(freq, np.absolute(X)/3000.)
#ax1.set_xlim(0,60)
#plt.show()

#X = fft(Ep_FINAL)
#freq = fftfreq(Ep_FINAL.size, d=TR)
#
## Only keep positive frequencies.
#keep = freq>=0
#X = X[keep]
#freq = freq[keep]
#
#ax1 = plt.subplot(111)
#ax1.plot(freq, np.absolute(X)/3000.)
#ax1.set_xlim(0,5000)
#plt.show()

#Fourier transform of autocorrelation
lags1, c1, line1, b1 = acorr(((Ep_FINAL - np.mean(Ep_FINAL))), usevlines=False, normed=True, maxlags=3998, lw=2)
Y3 = (fft(np.real(c1)))
freqs = np.fft.fftfreq(len(c1), ht*TR_mean)
freqs1 = np.fft.fftshift(freqs)
Y4 = fftshift(fft(ifftshift(c1)))
plt.figure(23)
plt.plot(freqs1,Y4)
plt.xlabel('f (Hz)')
plt.xlim([-100000,100000])
plt.show()

plt.figure(figsize=(8,6))
plt.plot(freqs,np.abs(Y3),'k',color = '0.7',linewidth = 2.0,label=r"$ \mathrm{Power}$")
plt.xlabel("$ \mathrm{f \ (Hz)}$",**font)
plt.ylabel("$ \\frac {\mathrm{V}^2}{\mathrm{m}^2 \mathrm{Hz}}$",**font)
plt.xlim([-100000,100000])
plt.savefig('/Users/Abhilash/Desktop/Power_Spectrum.png')
plt.show()

#plt.figure(23)
#plt.plot(x_TEST[::10],Ep_FINAL[::10])
#plt.xlabel('t (s)')
#plt.ylabel('Intensity')
#plt.savefig('/Users/Abhilash/Desktop/OUTPUT.png')


#FRB 110220
output_file = open('/Users/Abhilash/Downloads/FRB_110220_better.txt','r')
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()]) 
x_FRB = (np.array(output_data[::2]) - 51.5)
y_FRB = output_data[1::2]

output_file = open('/Users/Abhilash/Desktop/FRB110220_7percent_15micros_TR_15000micros_T1andT2_1000cycles.txt','r')
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()]) 
    
t2 = 1000.*np.array(output_data[::6])
z2 = output_data[1::6]
E2 = output_data[2::6]
N2 = output_data[3::6]
P2 = output_data[4::6]
IntensityFINAL2 = np.array(output_data[5::6])
    
plt.figure(figsize=(8,6))
plt.plot(x_FRB,y_FRB,'k',linewidth = 3.0,label=r"$ \mathrm{FRB \ 110220}$")
plt.plot(t2,(IntensityFINAL2/(0.017/1.38)),'r',linewidth = 3.0,label=r"$ \mathrm{Superradiance}$")
plt.xlim([0.,33.])
plt.xlabel("$ \\tau \ \mathrm{(ms)}$",**font)
plt.ylabel(r"$ S_{\nu} \ \mathrm{(Jy)}$",**font)
plt.legend(frameon=False,prop={'size':14})
plt.savefig('/Users/Abhilash/Desktop/FRB110220_fit_GOOD.png')
plt.show() 


#FRB 150418
output_file = open('/Users/Abhilash/Downloads/FRB-v2.txt','r') #Keane's data
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()]) 
x_FRB = (np.array(output_data[2::8]))*(64.*(10.**(-6.)))*1000.
y_FRB = output_data[3::8]

output_file = open('/Users/Abhilash/Desktop/FRB150418_1percent_1_4micros_TR_700micros_T1andT2_1000cycles.txt','r')
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()]) 
    
t2 = 1000.*np.array(output_data[::6])
z2 = output_data[1::6]
E2 = output_data[2::6]
N2 = output_data[3::6]
P2 = output_data[4::6]
IntensityFINAL2 = np.array(output_data[5::6])
    
plt.figure(figsize=(8,6))
plt.plot((x_FRB-17.3),y_FRB,'k',linewidth = 3.0,label=r"$ \mathrm{FRB \ 150418}$")
plt.plot(((t2+0.1)-0.3),(IntensityFINAL2/(0.000155/1.6)),'g',linewidth = 3.0,label=r"$ \mathrm{Superradiance}$")
plt.xlim([0.,2.5])
plt.xlabel("$ \\tau \ \mathrm{(ms)}$",**font)
plt.ylabel(r"$ S_{\nu} \ \mathrm{(Jy)}$",**font)
plt.legend(frameon=False,prop={'size':14})
plt.savefig('/Users/Abhilash/Desktop/FRB150418_fit_GOOD.png')
plt.show() 

#plt.plot(t2,IntensityFINAL2)
#plt.show()
 
#FRB multiple-peak

output_file = open('/Users/Abhilash/Desktop/single_pulse_FINAL_0percent_14micros_TR_40000micros_T1and4000micros_T2_good.txt','r')
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()])
    
newpath = r'/Users/Abhilash/Desktop/sigma=0.1mean_newOH_multiple_cycles=500_different_t_reinvert_1720MHz_SVEA_superradiance_T1=0.04_T2=0.004_TR='+str(TR)+'_TR='+str(TR_mean)
if not os.path.exists(newpath):
    os.makedirs(newpath) 


t = output_data[::6]
z = output_data[1::6]
E = output_data[2::6]
N = output_data[3::6]
P = output_data[4::6]
IntensityFINAL = np.array(output_data[5::6])

plt.figure(figsize=(8,6))
plt.plot(1000.*np.array(t),IntensityFINAL,'k',linewidth = 3.0,label=r"$ \mathrm{(Superradiance)}$")
#plt.xlim([0.,2.5])
plt.xlabel("$ \\tau \ \mathrm{(ms)}$",**font)
plt.ylabel(r"$ S_{\nu} \ \mathrm{(arb. units)}$",**font)
#plt.legend(frameon=False,prop={'size':14})
plt.savefig('/Users/Abhilash/Desktop/multiple_peak_fit_GOOD.png')
plt.show() 

#FRB Exponential
output_file = open('/Users/Abhilash/Desktop/FRB131104_5percent_5micros_TR_10ms_T1andT2_1000cycles.txt','r')
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()]) 
    
t2 = np.array(output_data[::6])
z2 = output_data[1::6]
E2 = output_data[2::6]
N2 = output_data[3::6]
P2 = output_data[4::6]
IntensityFINAL2 = np.array(output_data[5::6])

output_file = open('/Users/Abhilash/Downloads/FRB131104.txt','r')
output_data = []
next(output_file)
for line in output_file:
    output_data.extend([float(i) for i in line.split()]) 
x_FRB = (np.array(output_data[0::2]))
y_FRB = output_data[1::2]

YH5 = np.e**(-((((np.array(x_TEST)-tau_D))/(tau_D))**0.5))

plt.figure(figsize=(8,6))
plt.plot((np.array(x_TEST)*1000.),y_TEST/1.63,'k',color = '0.7',linewidth = 3.0,label=r"$ \mathrm{Superradiance}$")
plt.plot((np.array(x_TEST)*1000.),YH5*1.02,'k--',linewidth = 3.0,label=r"$ e^{-[(\tau - \tau_D)/\tau_D]^{0.5}}$")
##plt.plot((x_FRB-30.5),y_FRB,'k',linewidth = 3.0,label='FRB 131104')
#plt.xlim([0.,40.])
#plt.ylim([0.,1.])
plt.xlabel("$ \\tau \ \mathrm{(ms)}$",**font)
plt.ylabel(r"$ \mathrm{Normalized \ Intensity}$",**font)
plt.legend(frameon=False,prop={'size':16}) 
plt.savefig('/Users/Abhilash/Desktop/exponential_GOOD.png')
plt.show()

#CONSIDER INCLUDING THIS FREQUENCY AUTOCORRELATION AS DONE IN RAVI!
lags1, c1, line1, b1 = acorr(((Ep_FINAL - np.mean(Ep_FINAL))), usevlines=False, normed=True, maxlags=3998, lw=2)
Y3 = (fft(np.real(c1)))
freqs = np.fft.fftfreq(len(c1), ht*TR_mean)
freqs1 = np.fft.fftshift(freqs)
Y4 = (fft(ifftshift(c1)))
lags1, c1, line1, b1 = acorr(((Y4)), usevlines=False, normed=True, maxlags=3998, lw=2)
X_axis = ((lags1*(freqs1[2]-freqs1[1])))
plt.figure(23)
plt.plot(X_axis,c1, 'k',color = '0.7',linewidth = 3.0)
plt.xlabel('f (Hz)')
plt.xlim([-100000,100000])
plt.savefig('/Users/Abhilash/Desktop/frequency_spectrum_autocorr_GOOD.png')
plt.show()