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
import matplotlib.pyplot as plt

###debugging
#import matplotlib.pyplot as plt
###

lmax = 10
mmax = 10
nmax = 20
nr = 100
rmax = 1
dt = 0.003
nt = 20
phi0 = 0.5
eps = -0.2
lam = 0.1
ls = 0.0000001
ld = 0.1
iflinear = 1

l_sim = 0
m_sim = 0
n_sim = 7

a_init = 0
b_init = 1

simpars = sim_utils.SimPars(lmax,mmax,nmax,nr,rmax,dt,nt,np.linspace(0,1,nr)*rmax,iflinear) #store parameters for the run in one object

physpars = sim_utils.PhysPars(eps,lam,ls,ld) #store physical parameters

PH,CO,AR,sh = sim_utils.make_coords(simpars)
sh.nmax = nmax
cost = sh.cos_theta
print('Coords made.')

besselzers = np.loadtxt('../zerovals.txt')
zers = sim_utils.shaperight(besselzers[:lmax,:nmax],sh)
print('Zeros loaded')

ncpu = 6
p = mlp.Pool(ncpu)
print('Pool started')

lm_num = len(sh.l)
el = sh.l
em = sh.m

anlm = np.zeros((lm_num,nmax),dtype = complex)
bnlm = np.zeros((lm_num,nmax),dtype = complex)

m_anlm = np.zeros((lm_num,nmax),dtype = complex)
m_bnlm = np.zeros((lm_num,nmax),dtype = complex)

anlm[(el==l_sim)*(em==m_sim),n_sim] = a_init 
bnlm[(el==l_sim)*(em==m_sim),n_sim] = b_init

wth,wphi,wr = sim_utils.my_sh_to_spat(anlm,bnlm,sh,simpars,zers,p)
mth,mphi,mr = sim_utils.my_sh_to_spat(m_anlm,m_bnlm,sh,simpars,zers,p)

initarrs = (wr,wth,wphi,mr,mth,mphi,sh) 

wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh,zers,p)
zero_thismode = zers[(el==l_sim)*(em==m_sim),n_sim]

combo_a = zero_thismode**2*lam**2/rmax**2
combo_bd = zero_thismode**2*ld**2/rmax**2
combo_bs = zero_thismode**2*ls**2/rmax**2

gamma_a = 2*(eps-combo_a)/(1+combo_a)
gamma_b = -2*(eps-combo_bd-combo_bs)/(1+combo_bs)
k_b = 4*combo_bd/(1+combo_bs)

b_init_prime = (2*(eps+1) - 2*combo_bd)/(1+combo_bs)*b_init #This is implied by m_init = 0

tarr = dt*np.array(range(nt))
a_evol = a_init*np.exp(tarr*gamma_a)
b_evol = np.exp(-tarr*gamma_b/2)*(b_init*np.cosh(tarr/2*(gamma_b**2-4*k_b + 0j)**(1/2)) + (gamma_b*b_init + 2*b_init_prime)*np.sinh(tarr/2*(gamma_b**2-4*k_b + 0j)**(1/2))/((gamma_b**2-4*k_b + 0j)**(1/2)))

wa_hist = wa[(el==l_sim)*(em==m_sim),n_sim,:]
wb_hist = wb[(el==l_sim)*(em==m_sim),n_sim,:]

name = 'l'+str(l_sim)+'m'+str(m_sim)+'n'+str(n_sim)+'hist.pdf'

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(tarr,wa_hist[0],'rx',label='Numerical')
ax1.plot(tarr,a_evol,'b--',label='Analytical')
ax2.plot(tarr,wb_hist[0],'rx',label='Numerical')
ax2.plot(tarr,b_evol,'b--',label='Analytical')

ax1.set_xlabel('Time')
ax1.set_ylabel('Mode amplitude')
ax1.set_title('Transverse')

ax2.set_xlabel('Time')
ax2.set_ylabel('Mode amplitude')
ax2.set_title('Longitudinal')
fig.tight_layout()
plt.show()
