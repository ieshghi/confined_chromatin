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
nmax = 10
nr = 100
rmax = 1
dt = 0.1
nt = 200
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

# define the values of epsilon over which we scan
neps = 10
epsvals = np.linspace(criticalval_trans*1.1,criticalval_long + delta_eps,neps)

#make figure
color_grad = plotting_utils.get_color_gradient('#FFFD00','#FF0000',neps) #make colors go from yellow to red
plt.close('all')
fig,axs = plt.subplots(lmax,1)
for i in range(lmax):
    axs[i].set_title('l='+str(i))
    axs[i].set_xlabel('n')
    axs[i].set_ylabel('saturation value')

for i in range(neps):
    print('Epsilon value '+str(i)+' / '+str(neps-1))
    # Define initial conditions
    eps = epsvals[i]
    physpars = sim_utils.PhysPars(eps,lam,ls,ld) #store physical parameters
    
    anlm = np.random.rand(lm_num,nmax) + 0j
    bnlm = np.random.rand(lm_num,nmax) + 0j
    
    gamma_a = 2*(eps-criticalval_trans)/(1+criticalval_trans)
    m_anlm = anlm/(gamma_a/2+1)
    m_bnlm = np.zeros((lm_num,nmax),dtype = complex)
    
    wth,wphi,wr = sim_utils.my_sh_to_spat(anlm,bnlm,sh,simpars,zers,p)
    mth,mphi,mr = sim_utils.my_sh_to_spat(m_anlm,m_bnlm,sh,simpars,zers,p)
    
    initarrs = (wr,wth,wphi,mr,mth,mphi,sh) 
    
    #run the evolution
    
    wa,wb,ma,mb,mmax_hist,sh,r = spherical_integrate.main(simpars,physpars,initarrs,sh,zers,p)
    
    ma_last = ma[:,:,-1]
    mb_last = mb[:,:,-1]
    
    for j in range(lmax):
        ind = (el==j)*(em==0)
        axs[j].plot(np.arange(nmax),ma_last[ind,:],color=color_grad[i])

plt.show()
