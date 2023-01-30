import sys
sys.path.insert(0,'../')
import sim_utils
import spherical_integrate
import plotting_utils
import shtns
import numpy as np

lmax = 20
mmax = 20
nmax = 30
nr = 100
rmax = 1
dt = 0.005
nt = 300
phi0 = 0.5
eps = -1
lam = 0.001
ls = 0.001
ld = 0.1

simpars = sim_utils.SimPars(lmax,mmax,nmax,nr,rmax,dt,nt,np.linspace(0,1,nr)*rmax) #store parameters for the run in one object

physpars = sim_utils.PhysPars(eps,lam,ls,ld) #store physical parameters

PH,CO,AR,sh = sim_utils.make_coords(simpars)

wr,wth,wphi,mr,mth,mphi = sim_utils.initialize_arrays(simpars,sh)
#Initial conditions here

rho_init = np.exp(-(AR-0.5)**2/(0.1)**2)*np.exp(-CO**2/0.1**2)*np.exp(-(PH-np.pi)**2/0.5**2)
besselzers = np.loadtxt('../zerovals.txt')
rholm = sim_utils.my_analys(rho_init,sh,simpars,besselzers)

wr,wth,wphi = sim_utils.my_sh_to_spat(0*rholm,ld/(phi0*(1-phi0))*rholm,sh,simpars,besselzers)

initarrs = (wr,wth,wphi,mr,mth,mphi,sh)
wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh)
spherical_integrate.save_out(wa,wb,ma,mb,sh,r,simpars,'corner_blob')
plotting_utils.density_movie(wb,sh,r,simpars,rho_init,phi0,besselzers,undersamp = 1,name = 'density_mov',minmax = [np.min(rho_init),np.max(rho_init)])
