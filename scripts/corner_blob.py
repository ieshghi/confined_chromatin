import sys
sys.path.insert(0,'../')
import sim_utils
import spherical_integrate
import shtns
import numpy as np

lmax = 20
mmax = 15
nmax = 30
nr = 100
rmax = 1
dt = 0.03
nt = 3

simpars = sim_utils.SimPars(lmax,mmax,nmax,nr,rmax,dt,nt,np.linspace(0,1,nr)*rmax) #store parameters for the run in one object

physpars = sim_utils.PhysPars(0.1,0.001,0.01,0.1) #store physical parameters

PH,CO,AR,sh = sim_utils.make_coords(simpars)

wr,wth,wphi,mr,mth,mphi = sim_utils.initialize_arrays(simpars,sh)
#Initial conditions here

rho_init = np.exp(-(AR-0.5)**2/(0.1)**2)*np.exp(-CO**2/0.1**2)*np.exp(-PH**2/0.2**2)
rholm = sim_utils.my_analys(rho_init,sh,simpars)

wr,wth,wphi = my_spat_to_sh(0*rholm,rholm,sh,simpars)

initarrs = (wr,wth,wphi,mr,mth,mphi,sh)

wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh)
spherical_integrate.save_out(wa,wb,ma,mb,sh,r,'corner_blob')
