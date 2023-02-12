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


lmax = 10
mmax = 10
nmax = 20
nr = 100
rmax = 1
dt = 0.01
nt = 50
phi0 = 0.5
eps = -1
lam = 0.001
ls = 0.001
ld = 0.1
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

rho_init = np.exp(-((X)**2 + Y**2 + Z**2)/0.01)

sh.nmax = nmax
#rho_init = np.exp(-(AR-0.5)**2/(0.1)**2)*np.exp(-CO**2/0.1**2)*np.exp(-(PH-np.pi)**2/0.01**2)
print('Arrays initialized')
besselzers = np.loadtxt('../zerovals.txt')
zers = sim_utils.shaperight(besselzers[:lmax,:nmax],sh)
print('Zeros loaded')
#
ncpu = 6
p = mlp.Pool(ncpu)
print('Pool started')
sh.nmax = nmax

zers_keep = sim_utils.shaperight(zers[:lmax,:nmax],sh)

#gradphi_to_w = -ls**2*np.sqrt(ld)/(1+ls**2*zers_keep**2/rmax**2) 
bnlm = sim_utils.my_analys(rho_init,sh,simpars,zers,p)
anlm = 0*bnlm
print('Initial conditions calculated')

wth,wphi,wr = sim_utils.my_sh_to_spat(anlm,bnlm,sh,simpars,zers,p)
print('Initial conditions converted to spatial rep.')

initarrs = (wr,wth,wphi,mr,mth,mphi,sh) 
wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh,zers,p)
#spherical_integrate.save_out(wa,wb,ma,mb,sh,r,simpars,'corner_blob')
#wa,wb,ma,mb,sh,r,simpars = plotting_utils.load_w_hist('corner_blob')
plotting_utils.density_movie(wb,sh,r,simpars,rho_init,phi0,zers,undersamp = 1,name = 'density_mov_centered',minmax = [np.min(rho_init),np.max(rho_init)],pool = p)
#plotting_utils.animate_soln(wa,wb,ma,mb,sh,r,simpars,zers,undersamp = 1,fname = 'diff_mov_centered')
#plotting_utils.animate_soln_arrows(wa,wb,sh,r,simpars,zers,undersamp = 1,fname = 'diffusive_arrows',spatial_undersamp = 10)
