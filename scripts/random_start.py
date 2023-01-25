import sys
sys.path.insert(0,'../')

import sim_utils
import shtns
import numpy as np

simpars = sim_utils.SimPars(20,15,30,100,1) #store parameters for the run in one object
physpars = sim_utils.PhysPars(0.1,0.001,0.01,0.1) #store physical parameters

PH,CO,AR,sh = sim_utils.make_coords(simpars)

wr,wth,wphi,mr,mth,mphi = sim_utils.initialize_arrays(simpars)
#Initial conditions here

#rho_init = np.exp(-(AR-0.5)**2/(0.1)**2)*np.exp(-CO**2/0.1**2)*np.exp(-PH**2/0.2**2)
#rholm = sim_utils.my_analys(rho_init,sh)

#wth,wphi = sh.synth_grad(

wr = np.random.rand(CO.shape[0],CO.shape[1],CO.shape[2])

initarrs = (wr,wth,wphi,mr,mth,mphi,sh)

wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh)
spherical_integrate.save_out(wa,wb,ma,mb,sh,r,'randomstart')
