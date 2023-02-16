#  Spherical harmonics expansions done with SHTNS library
#  Copyright (c) 2010-2018 Centre National de la Recherche Scientifique.
#  written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
#
#  nathanael.schaeffer@univ-grenoble-alpes.fr
#
#  All dynamics, radial Bessel function expansions, and movies were written by 
#  Iraj Eshghi (New York University Center for Soft Matter Research)
#
#  iraj.eshghi@nyu.edu
#
 
import numpy as np 
import matplotlib.pyplot as plt
import shtns        
import sim_utils as utils
import pickle

def main(simpars,physpars,initarrs,sh,zers = None,pool = None): 
    lmax = simpars.lmax  # maximum degree of spherical harmonic representation.
    mmax = simpars.mmax  # maximum order of spherical harmonic representation.  
    nmax = simpars.nmax  # maximum order of radial Bessel function representation
    nr = simpars.nr      # number of points in radial grid 
    rmax = simpars.rmax  # radius of the sphere
    iflinear = simpars.iflinear #whether to include the nonlinearity in m_eq

    r = np.linspace(0,1,nr)*rmax #radial positions
    
    #sh = shtns.sht(lmax, mmax)  # create sht object with given lmax and mmax (orthonormalized)
     
    nlat, nphi = sh.set_grid()  # build default grid (gauss grid, phi-contiguous)
    phi = np.linspace(0,2*np.pi,nphi,endpoint = False) #assume uniform grid in \phi direction
    
    el = sh.l                   # array of size sh.nlm giving the spherical harmonic degree l for any sh coefficient
    em = sh.m
    sh.r = r
    sh.phi = phi
    sh.nmax = nmax
    cost = sh.cos_theta
    #Initialize arrays
    wr = initarrs[0]
    wth = initarrs[1]
    wphi = initarrs[2]
    mr = initarrs[3]
    mth = initarrs[4]
    mphi = initarrs[5]
    
    if zers is None:
        zers = np.loadtxt('../zerovals.txt') #load zeros of the spherical bessel functions
        zers_keep = utils.shaperight(zers[:lmax,:nmax],sh).T #keep only the relevant zeros
    else:
        zers_keep = zers
    
    #### convert w init conds to harmonics
    
    wanlm,wbnlm = utils.my_spat_to_sh(wth.copy(),wphi.copy(),wr.copy(),sh,simpars,zers,pool) 
    manlm,mbnlm = utils.my_spat_to_sh(mth.copy(),mphi.copy(),mr.copy(),sh,simpars,zers,pool) 
    
    #### Physics parameters

    ld = physpars.ld    # Osmotic length
    ls = physpars.ls    # Screening length
    lam = physpars.lam  # Mesh size
    eps = physpars.eps  # Critical forcing parameter (eps > 0, ordered phase spontaneously forms in infinite domain)
                        
    dt = simpars.dt     # Time step
    nt = simpars.nt     # Number of time points
    
    #Store history of parameters here
    wanlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)
    manlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)
    wbnlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)
    mbnlm_hist = np.zeros((sh.nlm,nmax,nt),dtype = complex)
    
    ### Time evolution arrays
    evol_curl = dt*(eps+1)/(1+(zers_keep*lam/rmax)**2)
    
    bigA = (2*ld**2*zers_keep**2/rmax**2)/(1+ls**2*zers_keep**2/rmax**2)
    bigB = (eps+1)/(1+ls**2*zers_keep**2/rmax**2)
    
    evol_long_pn = dt*bigB/(1+dt*bigA/2)
    evol_long_wn = (1-dt*bigA/2)/(1+dt*bigA/2) #crank-nicolson scheme
    
    ### Time evolution loop
    for i in range(nt):
        print('Timestep: '+str(i))
        wanlm_hist[:,:,i] = wanlm.copy()
        manlm_hist[:,:,i] = manlm.copy()
        wbnlm_hist[:,:,i] = wbnlm.copy()
        mbnlm_hist[:,:,i] = mbnlm.copy()
        mr_eq,mth_eq,mphi_eq = utils.meq(wr.copy(),wth.copy(),wphi.copy(),iflinear)

        pth = 2*(mth_eq -mth).copy()
        pphi = 2*(mphi_eq -mphi).copy()
        pr = 2*(mr_eq -mr).copy()
        panlm,pbnlm = utils.my_spat_to_sh(pth,pphi,pr,sh,simpars,zers,pool)
    
        if i == 0: #for the first time step we use explicit Euler
            mth += dt*pth
            mphi += dt*pphi
            mr += dt*pr
            wanlm += evol_curl*panlm
            wbnlm = evol_long_pn*pbnlm + evol_long_wn*wbnlm
        else: #adams-bashforth two-step
            mth += dt*(3/2*pth-1/2*pth_laststep)
            mphi += dt*(3/2*pphi-1/2*pphi_laststep)
            mr += dt*(3/2*pr-1/2*pr_laststep)
            wanlm += evol_curl*(3/2*panlm - 1/2*panlm_laststep)
            wbnlm = evol_long_pn*(3/2*pbnlm - 1/2*pbnlm_laststep) + evol_long_wn*wbnlm
    
        wth,wphi,wr = utils.my_sh_to_spat(wanlm.copy(),wbnlm.copy(),sh,simpars,zers,pool)
        manlm,mbnlm = utils.my_spat_to_sh(mth.copy(),mphi.copy(),mr.copy(),sh,simpars,zers,pool)
        pth_laststep = pth.copy()
        pphi_laststep = pphi.copy()
        pr_laststep = pr.copy()
        panlm_laststep = panlm.copy()
        pbnlm_laststep = pbnlm.copy()

    return wanlm_hist,wbnlm_hist,manlm_hist,mbnlm_hist,sh,r
    
def save_out(wanlm,wbnlm,manlm,mbnlm,sh,r,simpars,name='wanlm_hist'):
    ### Save output to a file
    with open('../output_files/'+str(name)+'.pickle', 'wb') as f:
        pickle.dump(wanlm,f)
        pickle.dump(manlm,f)
        pickle.dump(wbnlm,f)
        pickle.dump(mbnlm,f)
        pickle.dump(sh,f)
        pickle.dump(r,f)
        pickle.dump(simpars,f)
