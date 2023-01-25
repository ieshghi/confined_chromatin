#  Code based on "shtns_test" as provided by the SHTNS project
#  Copyright (c) 2010-2018 Centre National de la Recherche Scientifique.
#  written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
#
#  nathanael.schaeffer@univ-grenoble-alpes.fr
#
#  All dynamics, radial Bessel function expansions, and movies were written by 
#  Iraj Eshghi (New York University Center for Soft Matter Research)
#
#  Contact: iraj.eshghi@nyu.edu
#
 
import numpy as np 
import matplotlib.pyplot as plt
import shtns        
#import expansions as ex #import module where we define bessel expansions
import sim_utils as utils
import pickle

def initialize(nlat,nphi,nr,rmax,sh): #put initial conditions here
    wr = np.zeros((nlat,nphi,nr))
    wth = np.zeros((nlat,nphi,nr))
    wphi = np.zeros((nlat,nphi,nr))
    pr = np.zeros((nlat,nphi,nr))
    pth = np.zeros((nlat,nphi,nr))
    pphi = np.zeros((nlat,nphi,nr))
    mr = np.zeros((nlat,nphi,nr))
    mth = np.zeros((nlat,nphi,nr))
    mphi = np.zeros((nlat,nphi,nr))
    
    cost = sh.cos_theta
    phi = sh.phi
    r = sh.r 
    PH,CO,AR = np.meshgrid(phi,cost,r)

    mphi = 0.1*np.random.rand(CO.shape)
    mr = AR*(1-AR)*np.sin**2(AR) 

    return wr,wth,wphi,pr,pth,pphi,mr,mth,mphi
 
lmax = 20           # maximum degree of spherical harmonic representation.
mmax = 15            # maximum order of spherical harmonic representation.  
nmax = 30           # maximum order of radial Bessel function representation
nr = 100            # number of points in radial grid 
rmax = 1            # radius of the sphere

r = np.linspace(0,1,nr)*rmax #radial positions

#sh = shtns.sht(lmax, mmax)  # create sht object with given lmax and mmax (orthonormalized)
sh = utils.picklable_sht(lmax, mmax)  # create sht object with given lmax and mmax (orthonormalized)
 
nlat, nphi = sh.set_grid()  # build default grid (gauss grid, phi-contiguous)
phi = np.linspace(0,2*np.pi,nphi) #assume uniform grid in \phi direction

el = sh.l                   # array of size sh.nlm giving the spherical harmonic degree l for any sh coefficient
em = sh.m
sh.r = r
sh.phi = phi
sh.nmax = nmax
cost = sh.cos_theta
sh.nvals = range(1,nmax+1)
#Initialize arrays
wr,wth,wphi,pr,pth,pphi,mr,mth,mphi = initialize(nlat,nphi,nr,rmax,sh)

zers = np.loadtxt('zerovals.txt') #load zeros of the spherical bessel functions

#### convert w init conds to harmonics

wanlm,wbnlm = utils.my_spat_to_sh(wth,wphi,sh,zers) 
manlm,mbnlm = utils.my_spat_to_sh(mth,mphi,sh,zers) 
panlm,pbnlm = utils.my_spat_to_sh(pth,pphi,sh,zers)

#### Physics parameters

ld = 0.1 # Osmotic length
ls = 0.001 # Screening length
lam = 0.01 # Mesh size
eps = 10 # Critical forcing parameter (eps > 0, ordered phase spontaneously forms in infinite domain)
dt = 0.03 # Time step
nt = 300 # Number of time points

#Store history of parameters here
wanlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)
manlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)
wbnlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)
mbnlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)

### Time evolution arrays
zers_keep = utils.shaperight(zers[:lmax,:nmax],sh).T #keep only the relevant zeros
evol_curl = dt*(eps+1)/(1+(zers_keep*lam/rmax)**2)

bigA = (2*ld**2*zers_keep**2/rmax**2)/(1+ls**2*zers_keep**2/rmax**2)
bigB = (eps+1)/(1+ls**2*zers_keep**2/rmax**2)

evol_long_pn = dt*bigB/(1+dt*bigA/2)
evol_long_wn = (1-dt*bigA/2)/(1+dt*bigA/2) #crank-nicolson scheme

### Time evolution loop
for i in range(nt):
    print('Timestep: '+str(i))
    wanlm_hist[:,:,i] = wanlm.T
    manlm_hist[:,:,i] = manlm.T
    wbnlm_hist[:,:,i] = wbnlm.T
    mbnlm_hist[:,:,i] = mbnlm.T
    mr_eq,mth_eq,mphi_eq = utils.meq(wr,wth,wphi)

    pth = 2*(mth_eq -mth).copy()
    pphi = 2*(mphi_eq -mphi).copy()
    panlm,bpnlm = utils.my_spat_to_sh(pth,pphi,sh,zers)

    if i == 0: #for the first time step we use explicit Euler
        mth += dt*pth
        mphi += dt*pphi
        wanlm += evol_curl*panlm
        wbnlm = evol_long_pn*pbnlm + evol_long_wn*wbnlm
    else: #adams-bashforth two-step
        mth += dt*(3/2*pth-1/2*pth_laststep)
        mphi += dt*(3/2*pphi-1/2*pphi_laststep)
        wanlm += dt*evol_curl*(3/2*panlm-1/2*panlm_laststep)
        wbnlm = evol_long_pn*(3/2*pbnlm - 1/2*pbnlm_laststep) + evol_long_wn*(3/2*wbnlm - 1/2*wbnlm_laststep)


    wth,wphi,wr = utils.my_sh_to_spat(wanlm,wbnlm,sh,zers)
    pth_laststep = pth.copy()
    pphi_laststep = pphi.copy()
    panlm_laststep = panlm.copy()
    pbnlm_laststep = pbnlm.copy()
    wbnlm_laststep = wbnlm.copy()
    manlm,mbnlm = utils.my_spat_to_sh(mth,mphi,sh,zers)

### Save output to a file
with open('output_files/wanlm_hist.pickle', 'wb') as f:
    pickle.dump(wanlm_hist,f)
    pickle.dump(manlm_hist,f)
    pickle.dump(wbnlm_hist,f)
    pickle.dump(mbnlm_hist,f)
    pickle.dump(sh,f)
    pickle.dump(r,f)
