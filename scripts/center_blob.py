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


lmax = 15
mmax = 15
nmax = 10
nr = 100
rmax = 1
dt = 0.01
nt = 10
phi0 = 0.5
eps = -1 
lam = 0.0
ls = 0.
ld = 0.1
iflinear = 1

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
x0 = 0.51
y0 = 0
z0 = 0
name = 'density_mov_xshift01'
sigma = 0.1
rho_init = np.exp(-((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)/(2*sigma**2))

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
tarr = np.linspace(0,1,nt)*dt*nt/2 + t0
ARsqshift = (X-x0)**2 + (Y-y0)**2 + (Z-z0)**2
expected_density = np.exp(-ARsqshift[:,:,:,None]/(8*ld**2*tarr[None,:]))*(t0*np.ones(AR.shape)[:,:,:,None]/tarr[None,:])**(3/2)


#spherical_integrate.save_out(wa,wb,ma,mb,sh,r,simpars,'corner_blob')
#wa,wb,ma,mb,sh,r,simpars = plotting_utils.load_w_hist('corner_blob')
rho_hist = plotting_utils.density_movie(wb,sh,r,simpars,rho_init.copy(),phi0,zers,1,name,[np.min(rho_init),np.max(rho_init)],pool = p)
#plotting_utils.animate_soln(wa,wb,ma,mb,sh,r,simpars,zers,undersamp = 1,fname = 'diff_mov_centered')
#plotting_utils.animate_soln_arrows(wa,wb,sh,r,simpars,zers,undersamp = 1,fname = 'diffusive_arrows',spatial_undersamp = 10)
plt.close('all')
ARshift=AR-abs(x0)
rshift = r-abs(x0)

midslice_r = (CO==np.min(np.abs(CO)))*(PH==np.min(PH))
midslice_p = (CO==np.min(np.abs(CO)))*(ARshift == np.min(np.abs(ARshift)))
rslice = rshift==np.min(np.abs(rshift))
rslice_val = r[rslice]
slices_plot = [0,5,9]
fig,ax = plt.subplots(3,2)

phi = np.linspace(0,2*np.pi,sh.nphi,endpoint = False)
for i in range(len(slices_plot)):
    fr = slices_plot[i]
    ax[0,0].plot(phi,rho_hist[:,rslice,fr])
    ax[0,0].plot(phi,expected_density[midslice_p,fr],'--')
    ax[0,0].set_title(r'$\rho$')
    ax[0,0].set_xlabel(r'$\phi$')
    
    wth,wph,wr = sim_utils.my_sh_to_spat(wa[:,:,fr],wb[:,:,fr],sh,simpars,zers,p)
    wr_p = wr[midslice_p]
    ax[1,0].plot(phi,wr_p)
    ax[1,0].plot(phi,-2*ld**2/(phi0*(1-phi0))*np.gradient(rho_hist[:,:,fr],r,axis=1,edge_order=2)[:,rslice][:,0],'--')
    ax[1,0].set_title(r'$w_{r}$')
    ax[1,0].set_xlabel(r'$\phi$')

    wph_p = wph[midslice_p]
    ax[2,0].plot(phi,wph_p)
    ax[2,0].plot(phi,-2*ld**2/(phi0*(1-phi0))*np.gradient(rho_hist[:,rslice,fr][:,0],phi,edge_order=2)/rslice_val,'--')
    ax[2,0].set_title(r'$w_{\phi}$')
    ax[2,0].set_xlabel(r'$\phi$')

    ax[0,1].plot(r,rho_hist[0,:,fr])
    ax[0,1].plot(r,expected_density[midslice_r,fr],'--')
    ax[0,1].set_title(r'$\rho$')
    ax[0,1].set_xlabel('r')
    
    wth,wph,wr = sim_utils.my_sh_to_spat(wa[:,:,fr],wb[:,:,fr],sh,simpars,zers,p)
    wr_r = wr[midslice_r]
    ax[1,1].plot(r,wr_r)
    ax[1,1].plot(r,-2*ld**2/(phi0*(1-phi0))*np.gradient(rho_hist[0,:,fr],r,edge_order=2),'--')
    ax[1,1].set_title(r'$w_r$')
    ax[1,1].set_xlabel('r')

    wph_r = wph[midslice_r]
    ax[2,1].plot(r,wph_r)
    ax[2,1].plot(r,-2*ld**2/(phi0*(1-phi0))*np.gradient(rho_hist[:,:,fr],phi,axis=0,edge_order=2)[0,:]/rslice_val,'--')
    ax[2,1].set_title(r'$w_{\phi}$')
    ax[2,1].set_xlabel('r')

fig.tight_layout()
plt.show()
