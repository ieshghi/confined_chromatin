import shtns
import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn as jn
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing as mlp

def make_coords(simpars): #put initial conditions here

    sh = picklable_sht(simpars.lmax,simpars.mmax)  # create sht object with given lmax and mmax (orthonormalized)
    [nlat,nphi] = sh.set_grid()
    phi = np.linspace(0,2*np.pi,sh.nphi)
    cost = sh.cos_theta
    r = np.linspace(0,1,simpars.nr)*simpars.rmax
    PH,CO,AR = np.meshgrid(phi,cost,r)

    return PH,CO,AR,sh

def initialize_arrays(simpars,sh):
    [nlat,nphi] = sh.set_grid()
    nr = simpars.nr
    wr = np.zeros((nlat,nphi,nr))
    wth = np.zeros((nlat,nphi,nr))
    wphi = np.zeros((nlat,nphi,nr))
    mr = np.zeros((nlat,nphi,nr))
    mth = np.zeros((nlat,nphi,nr))
    mphi = np.zeros((nlat,nphi,nr))
    
    return wr,wth,wphi,mr,mth,mphi

class SimPars():
    def __init__(self,lmax,mmax,nmax,nr,rmax,dt,nt,r,iflinear):
        self.lmax = lmax
        self.mmax = mmax
        self.nmax = nmax
        self.nr = nr
        self.rmax = rmax
        self.dt = dt
        self.nt = nt
        self.r = r
        self.iflinear = iflinear


class PhysPars():
    def __init__(self,eps,lam,ls,ld):
        self.eps = eps
        self.lam = lam
        self.ls = ls
        self.ld = ld
        

def meq(wr,wth,wph,iflinear = 0):
    wsq = (wr**2 + wth**2 + wph**2)
    mr_eq = wr*(1-3*wsq/5*(1-iflinear))
    mth_eq = wth*(1-3*wsq/5*(1-iflinear))
    mph_eq = wph*(1-3*wsq/5*(1-iflinear))

    return mr_eq,mth_eq,mph_eq

def meq_1d(w):
    return w*(1-3*w**2/5)

class PickalableSWIG:
    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}

class picklable_sht(shtns.sht,PickalableSWIG):
    def __init__(self, *args):
        self.args = args
        shtns.sht.__init__(self,*args)

def shaperight(zers,sh): #converts the array of zeros to the correct shape, so it can be used to evolve the coefficients straightforwardly
    if zers.shape != (sh.lmax,sh.nmax):
        raise ValueError('Array of zeros does not match expected size.')

    zers_out = np.zeros((sh.nlm,sh.nmax))
    el = sh.l
    for i in range(len(el)):
        if el[i]==0:
            zers_out[i,:] = np.pi*np.array(range(sh.nmax))
        else:
            zers_out[i,:] = zers[el[i]-1,:]

    return zers_out
        
def my_div(bnlm,sh,simpars,besselzer=None,pool = None):
    # each b_nlm mode looks like w_{nlm} = grad (j_l(r\alpha_{ln}/R)Y_{lm}). These modes are all eigenvalues of the Laplace operator, such that
    # div(w_{nlm}) = -\alpha_ln^2/R^2 j_l(r\alpha_ln/R)Y_lm. All that remains is summing over them.

    nmax = simpars.nmax
    nr = simpars.nr
    r = simpars.r/simpars.rmax
    el = sh.l
    em = sh.m
    lm_num = len(el)
    sh.set_grid()

    ang = [sh.synth(bnlm[:,i]) for i in range(nmax)]

    args = ((r,ang[j],el[i],sh,besselzer[el[i],j]) for i in range(lm_num) for j in range(nmax))

    if pool is None:
        return sum(list(map(divcomponent_packed,args)))
    else:
        return sum(list(pool.map(divcomponent_packed,args)))

def divcomponent_packed(args):
    r,ang,el,sh,besselzer = args
    rad = jn(el,r*besselzer)

    return besselzer**2*ang[:,:,None]*rad[None,:]/(r[-1]**2)

def my_analys(rho,sh,simpars,besselzer,pool = None):
    r = simpars.r 
    nmax = simpars.nmax
    nr = len(r)
    el = sh.l
    
    lm_num = len(el)

    olm_r = np.array([sh.analys(rho[:,:,i]) for i in range(nr)])
    
    args = ((r,olm_r[:,i],nmax,el[i],besselzer,False) for i in range(lm_num))
    if pool is None:
        onlm = np.array(list(map(func2bessel_packed,args))).T
    else:
        onlm = np.array(pool.map(func2bessel_packed,args)).T
    
    return onlm.T
    
def my_spat_to_sh(v_th,v_ph,v_r,sh,simpars,besselzer,pool = None): #this routine converts from spatial to harmonic representation, including the radial decomposition into bessel functions. Only keeps the curly part.

    r = simpars.r
    nmax = simpars.nmax
    nr = len(r)
    el = sh.l
    lm_num = len(el)

    anlm = np.tile(sh.spec_array(),[nmax,1])
    bnlm = np.tile(sh.spec_array(),[nmax,1])

    alm_r = np.tile(sh.spec_array(),[nr,1])
    blm_r = np.tile(sh.spec_array(),[nr,1])
    clm_r = np.tile(sh.spec_array(),[nr,1])

    slm = sh.spec_array()
    tlm = sh.spec_array()
    qlm = sh.spec_array()

    one_spat = sh.spat_array()
    two_spat = sh.spat_array()
    three_spat = sh.spat_array()
    
    for i in range(nr): #transform from spatial to spherical coords
        one_spat[:,:] = v_th[:,:,i]
        two_spat[:,:] = v_ph[:,:,i]
        three_spat[:,:] = v_r[:,:,i]
        #sh.spat_to_SHsphtor(one_spat,two_spat,slm,tlm)
        sh.spat_to_SHqst(three_spat,one_spat,two_spat,qlm,slm,tlm)
        alm_r[i,:] = tlm
        blm_r[i,:] = slm
        clm_r[i,:] = qlm

    for i in range(lm_num):
        if el[i]==0:
            blm_r[:,i] = cumulative_trapezoid(clm_r[:,i],r,initial=0)
        else:
            blm_r[:,i] = blm_r[:,i]*r

    args_a = ((r,alm_r[:,i],nmax,el[i],besselzer,False) for i in range(lm_num))
    args_b = ((r,blm_r[:,i],nmax,el[i],besselzer,True) for i in range(lm_num))

    if pool is None:
        anlm = np.array(list(map(func2bessel_packed,args_a)))
        bnlm = np.array(list(map(func2bessel_packed,args_b)))
    else:
        anlm = np.array(pool.map(func2bessel_packed,args_a))
        bnlm = np.array(pool.map(func2bessel_packed,args_b))

    return anlm,bnlm

def my_sh_to_spat(anlm,bnlm,sh,simpars,besselzer = None,pool = None):

    nr = simpars.nr
    r = simpars.r
    nlat,nphi = sh.set_grid()
    el = sh.l
    lm_num = len(el)

    vth = np.empty((nlat,nphi,nr))
    vph = np.empty((nlat,nphi,nr))
    vr = np.empty((nlat,nphi,nr))

    one_spat = sh.spat_array()
    two_spat = sh.spat_array()

    args_a = ((r,anlm[i,:],el[i],besselzer) for i in range(lm_num))
    args_b = ((r,bnlm[i,:],el[i],besselzer) for i in range(lm_num))

    if pool is None:
        alm_r = np.array(list(map(bessel2func_packed,args_a))).T.copy()
        blm_r = np.array(list(map(bessel2func_packed,args_b))).T.copy()
    else:
        alm_r = np.array(pool.map(bessel2func_packed,args_a)).T.copy()
        blm_r = np.array(pool.map(bessel2func_packed,args_b)).T.copy()

    scal_blm_r = np.zeros(blm_r.shape,dtype = complex)
    scal_blm_r = np.gradient(blm_r,r,edge_order=2,axis=0)
    #scal_blm_r[:,1:] = np.gradient(blm_r[:,1:],r,edge_order=2,axis=0)
    #scal_blm_r[1:,0] = blm_r[1:,0]/r[1:]
    #scal_blm_r[0,0] = 0

    for i in range(nr):
        if i>0:
            sh.SHsphtor_to_spat(blm_r[i,:]/r[i],alm_r[i,:],one_spat,two_spat)
        else:
            sh.SHsphtor_to_spat(blm_r[i,:],alm_r[i,:],one_spat,two_spat) #first point is always 0 anyways

        vth[:,:,i] = one_spat
        vph[:,:,i] = two_spat
        vr[:,:,i] = sh.synth(scal_blm_r[i,:])

    return vth,vph,vr

def func2bessel_packed(args):
    x,y,nmax,l,zers,iflong = args
    return func2bessel(x,y,nmax,l,zers,iflong)

def func2bessel(x,y,nmax,l,zer,iflong=False):
    x_sc = x/np.max(x)
    lzer = zer[l,:nmax]
    if l > 0:
        return -2*np.array([simpson(x_sc**2*jn(l,lzer[i]*x_sc)*y,x_sc)/(jn(l-1,lzer[i])*jn(l+1,lzer[i])) for i in range(nmax)])
    else:
        if y[0] !=1 and iflong==True:
            y += 1-y[0]
            out = [2*simpson(x_sc**2*jn(0,lzer[i]*x_sc)*y,x_sc)*(i*np.pi)**2 for i in range(1,nmax)]
            return np.array([0]+out)
        else:
            out = [2*simpson(x_sc**2*jn(0,lzer[i]*x_sc)*y,x_sc)*(i*np.pi)**2 for i in range(1,nmax)]
            return np.array([0]+out)
        

def bessel2func_packed(args):
    x,y,l,zers = args
    return bessel2func(x,y,l,zers)

def bessel2func(x,ncoeffs,l,besselzer):
    nmax = len(ncoeffs)
    lbesselzer = besselzer[l,:nmax]
    return np.sum(jn(l,x[:,None]*lbesselzer[None,:])*ncoeffs,1)
#    if l>0:
#        return np.sum(jn(l,x[:,None]*lbesselzer[None,:])*ncoeffs,1)
#    else:
#        return -np.sum(lbesselzer*jn(1,x[:,None]*lbesselzer[None,:])*ncoeffs,1)/np.max(x)


def genzeros(lmax,mmax):
    zervals = np.zeros((lmax+1,mmax))
    for i in range(lmax+1):
        if i > 0:
            zervals[i,:] = np.array(spherical_jn_zeros(i,mmax,int(1e6)))

    np.savetxt('zerovals.txt',zervals)

def spherical_jn_sensible_grid(n, m, ngrid=100):
    """Returns a grid of x values that should contain the first m zeros, but not too many.
    """
    return np.linspace(n, n + 2*m*(np.pi * (np.log(n)+1)), ngrid)
    

def spherical_jn_zeros(n, m, ngrid=100):
    """Returns first m zeros of spherical bessel function of order n
    """
    # calculate on a sensible grid
    x = spherical_jn_sensible_grid(n, m, ngrid=ngrid)
    y = jn(n, x)
    
    # Find m good initial guesses from where y switches sign
    diffs = np.sign(y)[1:] - np.sign(y)[:-1]
    ind0s = np.where(diffs)[0][:m]  # first m times sign of y changes
    x0s = x[ind0s]
    
    def fn(x):
        return jn(n, x)
    
    return [root(fn, x0).x[0] for x0 in x0s]

def animate_soln(cnhist,dnhist,x,undersamp = 1,fname = 'bla',minmax=[-1,1]):
    nx = len(r) 
    nt = cnhist.shape[1]
    plt.style.use('seaborn-pastel')
    
    fig,(ax1,ax2) = plt.subplots(2,1)
    ax1.set_xlim(np.min(x), np.max(x))
    ax2.set_xlim(np.min(x), np.max(x))

    #if minmax is None: 
    #    ax1.set_ylim(np.min(h), np.max(h))
    #    ax2.set_ylim(np.min(m), np.max(m))
    #else:
    #    ax1.set_ylim(minmax[0], minmax[1])
    #    ax2.set_ylim(minmax[0], minmax[1])

    ax1.set_ylim(minmax[0], minmax[1])
    ax2.set_ylim(minmax[0], minmax[1])

    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$w\tau/3a$')
    ax2.set_xlabel('r')
    ax2.set_ylabel('m')
    
    line, = ax1.plot([], [], lw=3)
    line2, = ax2.plot([], [], lw=3)

    def init():
        line.set_data([],[])
        line2.set_data([],[])
        if eps != 0:
            line3.set_data([],[])
            line4.set_data([],[])
            line5.set_data([],[])
            line6.set_data([],[])
        return line,

    def animate(i):
        y = ex.bessel2func(r,cnhist[:,i*undersamp])
        y2 = ex.bessel2func(r,dnhist[:,i*undersamp])
        line.set_data(x, y)
        line2.set_data(x, y2)
        
        return line,
    
    anim = FuncAnimation(fig, animate,frames=int(nt/undersamp), blit=True)
    
    anim.save('movies/'+fname+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
