import sys
sys.path.insert(0,'../')
import sim_utils
import spherical_integrate
import plotting_utils
import shtns
import numpy as np
import multiprocessing as mlp
import time
import datetime

###debugging
#import matplotlib.pyplot as plt
###

lmax = 40
mmax = 40
nmax = 20
nr = 100
rmax = 1
dt = 0.05
nt = 100
phi0 = 0.5
eps = -1
lam = 0.001
ls = 0.001
ld = 0.1/ls**4
iflinear = 0

simpars = sim_utils.SimPars(lmax,mmax,nmax,nr,rmax,dt,nt,np.linspace(0,1,nr)*rmax,iflinear) #store parameters for the run in one object

physpars = sim_utils.PhysPars(eps,lam,ls,ld) #store physical parameters

PH,CO,AR,sh = sim_utils.make_coords(simpars)
cost = sh.cos_theta
print('Coords made.')

wr,wth,wphi,mr,mth,mphi = sim_utils.initialize_arrays(simpars,sh)
#Initial conditions here

X = AR*np.cos(PH)*np.sqrt(1-CO**2)
Y = AR*np.sin(PH)*np.sqrt(1-CO**2)
Z = AR*CO

rho_init = np.exp(-((X-0.5)**2 + Y**2 + Z**2)/0.01)

#rho_init = np.exp(-(AR-0.5)**2/(0.1)**2)*np.exp(-CO**2/0.1**2)*np.exp(-(PH-np.pi)**2/0.01**2)
print('Arrays initialized')
besselzers = np.loadtxt('../zerovals.txt')
print('Zeros loaded')
#
ncpu = 40
p = mlp.Pool(ncpu)
print('Pool started')
sh.nmax = nmax

zers_keep = sim_utils.shaperight(besselzers[:lmax,:nmax],sh).T

gradphi_to_w = -ls**2*np.sqrt(ld)
bnlm = gradphi_to_w*sim_utils.my_analys(rho_init,sh,simpars,besselzers,p)
anlm = 0*bnlm
print('Initial conditions calculated')

wth,wphi,wr = sim_utils.my_sh_to_spat(anlm,bnlm,sh,simpars,besselzers,p)
print('Initial conditions converted to spatial rep.')

initarrs = (wr,wth,wphi,mr,mth,mphi,sh) 
wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh,besselzers,p)
#spherical_integrate.save_out(wa,wb,ma,mb,sh,r,simpars,'corner_blob')
#wa,wb,ma,mb,sh,r,simpars = plotting_utils.load_w_hist('corner_blob')
plotting_utils.density_movie(wb,sh,r,simpars,rho_init,phi0,besselzers,undersamp = 1,name = 'density_mov_betterunits_fine',minmax = [np.min(rho_init),np.max(rho_init)],pool = p)
#plotting_utils.animate_soln_arrows(wa,wb,sh,r,simpars,besselzers,undersamp = 10,fname = 'diffusive_arrows',spatial_undersamp = 10)
