import sys
sys.path.insert(0,'../')
import sim_utils
import spherical_integrate
import plotting_utils
import shtns
import numpy as np
import multiprocessing as mlp
import matplotlib.pyplot as plt
from scipy.special import erf

###debugging
#import matplotlib.pyplot as plt
###


lmax = 10
mmax = 10
nmax = 10
nr = 100
rmax = 1
dt = 0.01
nt = 500
phi0 = 0.5
eps = 2
lam = 0.0
ls = 0.
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
sigma = 0.1
rho_init = np.exp(-AR**2/(2*sigma**2))

sh.nmax = nmax
print('Arrays initialized')
besselzers = np.loadtxt('../zerovals.txt')
zers = sim_utils.shaperight(besselzers[:lmax,:nmax],sh)
print('Zeros loaded')
#
ncpu = 8 
p = mlp.Pool(ncpu)
print('Pool started')
sh.nmax = nmax

zers_keep = sim_utils.shaperight(zers[:lmax,:nmax],sh)

gradphi_to_w = -2*ld**2/(phi0*(1-phi0)*(1+ls**2*zers_keep**2/rmax**2))
bnlm = gradphi_to_w*sim_utils.my_analys(rho_init.copy(),sh,simpars,zers,p)
anlm = 0*bnlm
print('Initial conditions calculated')

wth,wphi,wr = sim_utils.my_sh_to_spat(anlm,bnlm,sh,simpars,zers,p)
print('Initial conditions converted to spatial rep.')

initarrs = (wr,wth,wphi,mr,mth,mphi,sh) 
expected_coeffs = gradphi_to_w[0,:]*[(-1)**m*np.exp(-1/2*m*np.pi*(m*np.pi*sigma**2-2j))*m**2*np.pi**(5/2)*sigma**3/np.sqrt(2)*(erf((1-m*np.pi*sigma**2*1j)/(np.sqrt(2)*sigma))+erf((1+m*np.pi*sigma**2*1j)/(np.sqrt(2)*sigma))) for m in range(nmax)]
wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh,zers,p)

t0 = sigma**2/(4*ld**2)
tarr = np.linspace(0,1,nt)*dt + t0
expected_density = np.exp(-AR[:,:,:,None]**2/(8*ld**2*tarr[None,:]))*(t0*np.ones(AR.shape)[:,:,:,None]/tarr[None,:])**(3/2)

#spherical_integrate.save_out(wa,wb,ma,mb,sh,r,simpars,'corner_blob')
#wa,wb,ma,mb,sh,r,simpars = plotting_utils.load_w_hist('corner_blob')
rho_hist = plotting_utils.density_movie(wb,sh,r,simpars,rho_init.copy(),phi0,zers,undersamp = 1,name = 'density_mov_centered',minmax = [np.min(rho_init),np.max(rho_init)],pool = p)
#plotting_utils.animate_soln(wa,wb,ma,mb,sh,r,simpars,zers,undersamp = 1,fname = 'diff_mov_centered')
#plotting_utils.animate_soln_arrows(wa,wb,sh,r,simpars,zers,undersamp = 1,fname = 'diffusive_arrows',spatial_undersamp = 10)
plt.close('all')

midslice = (CO==np.min(np.abs(CO)))*(PH==np.min(PH))
slices_plot = [0,10,20]
fig,(ax1,ax2) = plt.subplots(2,1)
for i in range(len(slices_plot)):
    fr = slices_plot[i]
    ax1.plot(r,rho_hist[0,:,fr])
    ax1.plot(r,expected_density[midslice,fr],'--')
    
    wth,wph,wr = sim_utils.my_sh_to_spat(wa[:,:,fr],wb[:,:,fr],sh,simpars,zers,p)
    wr = wr[midslice]
    ax2.plot(r,wr)
    ax2.plot(r,-2*ld**2/(phi0*(1-phi0))*np.gradient(rho_hist[0,:,fr],edge_order=2),'--')

plt.show()
