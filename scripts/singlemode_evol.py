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
dt = 0.1
nt = 200
eps = .5
lam = 0.1
ls = 0.0000001
ld = 0.1
iflinear = 1

l_sim = 2
m_sim = 1
n_sim = 0

a_init = 1
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

ind_keep = (el==l_sim)*(em==m_sim)
zero_thismode = zers[ind_keep,n_sim]

combo_a = zero_thismode**2*lam**2/rmax**2
combo_bd = zero_thismode**2*ld**2/rmax**2
combo_bs = zero_thismode**2*ls**2/rmax**2

gamma_a = 2*(eps-combo_a)/(1+combo_a)
gamma_b = -2*(eps-combo_bd-combo_bs)/(1+combo_bs)
k_b = 4*combo_bd/(1+combo_bs)

b_init_prime = (2*(eps+1) - 2*combo_bd)/(1+combo_bs)*b_init #This is implied by m_init = 0

anlm[ind_keep,n_sim] = a_init 
bnlm[ind_keep,n_sim] = b_init
m_anlm[ind_keep,n_sim] = a_init/(gamma_a/2 + 1)

wth,wphi,wr = sim_utils.my_sh_to_spat(anlm,bnlm,sh,simpars,zers,p)
mth,mphi,mr = sim_utils.my_sh_to_spat(m_anlm,m_bnlm,sh,simpars,zers,p)

initarrs = (wr,wth,wphi,mr,mth,mphi,sh) 

wa,wb,ma,mb,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh,zers,p)
wa_corr = wa[ind_keep,n_sim,:][0]
wb_corr = wb[ind_keep,n_sim,:][0]
wa_else = np.sqrt(np.mean(wa**2,axis=(0,1))) - np.abs(wa_corr)/(wa.shape[0]*wa.shape[1])
wb_else = np.sqrt(np.mean(wb**2,axis=(0,1))) - np.abs(wb_corr)/(wa.shape[0]*wa.shape[1])

tarr = dt*np.array(range(nt))
a_evol = a_init*np.exp(tarr*gamma_a)
ma_evol = a_init*np.exp(tarr*gamma_a)/(gamma_a/2 + 1)

b_evol = np.exp(-tarr*gamma_b/2)*(b_init*np.cosh(tarr/2*(gamma_b**2-4*k_b + 0j)**(1/2)) + (gamma_b*b_init + 2*b_init_prime)*np.sinh(tarr/2*(gamma_b**2-4*k_b + 0j)**(1/2))/((gamma_b**2-4*k_b + 0j)**(1/2)))

wa_hist = wa[ind_keep,n_sim,:]
wb_hist = wb[ind_keep,n_sim,:]
ma_hist = ma[ind_keep,n_sim,:]
mb_hist = mb[ind_keep,n_sim,:]

name = 'l'+str(l_sim)+'m'+str(m_sim)+'n'+str(n_sim)+'hist.pdf'

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(tarr,wa_hist[0],'rx',label='Numerical w')
ax1.plot(tarr,a_evol,'b--',label='Analytical w')
ax1.plot(tarr,ma_hist[0],'gx',label='Numerical m')
ax1.plot(tarr,ma_evol,'k--',label='Analytical m')
ax2.plot(tarr,wb_hist[0],'rx',label='Numerical w')
ax2.plot(tarr,b_evol,'b--',label='Analytical w')
ax1.legend()
ax2.legend()

ax1.set_xlabel('Time')
ax1.set_ylabel('Mode amplitude')
ax1.set_title('Transverse')

ax2.set_xlabel('Time')
ax2.set_ylabel('Mode amplitude')
ax2.set_title('Longitudinal')
fig.tight_layout()

fig2,(ax20,ax21) = plt.subplots(1,2)
ax20.plot(tarr,wa_corr,label='Correct mode')
ax20.plot(tarr,wa_else,'--',label='RMS all other modes')

ax21.plot(tarr,wb_corr,label='Correct mode')
ax21.plot(tarr,wb_else,'--',label='RMS all other modes')
ax21.legend()
ax20.legend()

ax21.set_xlabel('Time')
ax21.set_ylabel('Mode amplitude')
ax21.set_title('Longitudinal')

ax20.set_xlabel('Time')
ax20.set_ylabel('Mode amplitude')
ax20.set_title('Transverse')
fig.tight_layout()
fig2.tight_layout()



plt.show()
