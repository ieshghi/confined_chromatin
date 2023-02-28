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
nmax = 5
nr = 100
rmax = 1
dt = 0.5
nt = 100
lam = 0.001
ls = 0.01
ld = 0.1
iflinear = 0

simpars = sim_utils.SimPars(lmax,mmax,nmax,nr,rmax,dt,nt,np.linspace(0,1,nr)*rmax,iflinear) #store parameters for the run in one object

PH,CO,AR,sh = sim_utils.make_coords(simpars)
sh.nmax = nmax
cost = sh.cos_theta
print('Coords made.')

lm_num = len(sh.l)
el = sh.l
em = sh.m

besselzers = np.loadtxt('../zerovals.txt')
zers = sim_utils.shaperight(besselzers[:lmax,:nmax],sh)
print('Zeros loaded')

ncpu = 6
p = mlp.Pool(ncpu)
print('Pool started')

criticalval_trans = (lam*zers[1,0])**2 #Assuming R = 1, this is the critical threshold for the transverse mode l = 1, n = 0
criticalval_long = (ld + ls)**2*(zers[0,1])**2 #This is the critical threshold for the longitudinal mode l=0,n=1

noiselevel = 1e-10
delta_eps = criticalval_long-criticalval_trans

epsvals = np.linspace(0,criticalval_long + delta_eps,20)

anlm = np.zeros((lm_num,nmax),dtype = complex) + np.random.rand(lm_num,nmax)*noiselevel
bnlm = np.zeros((lm_num,nmax),dtype = complex) + np.random.rand(lm_num,nmax)*noiselevel

m_anlm = np.zeros((lm_num,nmax),dtype = complex)
m_bnlm = np.zeros((lm_num,nmax),dtype = complex)

# Define initial conditions

#eps = criticalval_trans + delta_eps*1.5
eps = criticalval_long + delta_eps*(1.5)
physpars = sim_utils.PhysPars(eps,lam,ls,ld) #store physical parameters

ind_keep_a = (el==1)*(em==0)
n_sim_a = 0
ind_keep_b = (el==0)*(em==0)
n_sim_b = 1

a_init = 0.5
b_init = 0.1

gamma_a = 2*(eps-criticalval_trans)/(1+criticalval_trans)
anlm[ind_keep_a,n_sim_a] = a_init 
m_anlm[ind_keep_a,n_sim_a] = a_init/(gamma_a/2 + 1)
bnlm[ind_keep_b,n_sim_b] = b_init
wth,wphi,wr = sim_utils.my_sh_to_spat(anlm,bnlm,sh,simpars,zers,p)
mth,mphi,mr = sim_utils.my_sh_to_spat(m_anlm,m_bnlm,sh,simpars,zers,p)

initarrs = (wr,wth,wphi,mr,mth,mphi,sh) 

#run the evolution

wa,wb,ma,mb,mmax_hist,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh,zers,p)

ma_corr = ma[ind_keep_a,n_sim_a,:][0]
mb_corr = mb[ind_keep_b,n_sim_b,:][0]
ma_else = np.sqrt(np.mean(ma**2,axis=(0,1))) - np.abs(ma_corr)/(ma.shape[0]*ma.shape[1])
mb_else = np.sqrt(np.mean(mb**2,axis=(0,1))) - np.abs(mb_corr)/(ma.shape[0]*ma.shape[1])

tarr = dt*np.array(range(nt))
ma_hist = ma[ind_keep_a,n_sim_a,:][0]
mb_hist = mb[ind_keep_b,n_sim_b,:][0]

plt.close('all')

fig,(ax1) = plt.subplots(1,1)
ax1.plot(tarr,ma_hist,'-',label=r'$m_{\perp,10}$')
ax1.plot(tarr,mb_hist,'-',label=r'$m_{\parallel,01}$')
ax1.plot(tarr,ma_else,'--',label='RMS all other transverse modes')
ax1.plot(tarr,mb_else,'--',label='RMS all other longitudinal modes')
ax1.plot(tarr,mmax_hist,'-',label=r'Maximum value of $|\vec{m}|$')
ax1.legend()

ax1.set_xlabel('Time')
ax1.set_ylabel('Mode amplitude')
ax1.set_title('Epsilon = '+str(eps))
fig.tight_layout()

ma_last = np.abs(ma[:,:,-1])**2
mb_last = np.abs(mb[:,:,-1])**2


plt.show()
