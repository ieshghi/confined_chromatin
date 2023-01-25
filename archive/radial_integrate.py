import shtns
import sim_utils as ex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def meq(w):
    return w*(1-3*w**2/5)

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

nmax = 80
rnvals = 100

r = np.linspace(0,1,rnvals) #rmax still seems to have an issue.
rmax = np.max(r)
w = 1/2*np.sin(r*np.pi/rmax)**2*np.cos(2*r*np.pi/rmax)
m = 0*w
eps = 0.5
ld = 0.1
ls = 0.

cn = ex.func2bessel(r,w,nmax) 
dn = ex.func2bessel(r,m,nmax) 
zers = np.loadtxt('zerovals.txt')
zers_l1 = zers[0,:nmax]

dt = 0.01
nt = 1000
cnhist = np.zeros((nmax,nt+1)) + 0j
dnhist = np.zeros((nmax,nt+1)) + 0j
cnhist[:,0] = cn

#evol_array = (1-(2*ld**2*zers_l1**2*dt/rmax**2)/(1+ls**2*zers_l1**2/rmax**2)) #Not convinced this is correct
#evol_array_cn = 1-(2*ld**2*zers_l1**2*dt/rmax**2) # explicit, ignore screening
evol_array_cn = (1-(ld**2*zers_l1**2*dt/rmax**2))/(1+(ld**2*zers_l1**2*dt/rmax**2)) # crank-nicolson, ignore screening
evol_array_pn = dt*(eps+1)/(1+(ld**2*zers_l1**2*dt/rmax**2)) # crank-nicolson, ignore screening
for i in range(nt):
    print(i)
    mdot = 2*(meq(w)-m)
    pn = np.real(ex.func2bessel(r,mdot,nmax))
    m += dt*np.real(mdot)
    cn_new = evol_array_pn*pn + evol_array_cn*cn
    dn_new = np.real(ex.func2bessel(r,m,nmax))
    w = np.real(ex.bessel2func(r,cn_new))
    cnhist[:,i+1] = cn_new
    dnhist[:,i+1] = dn_new
    cn = cn_new
    dn = dn_new

animate_soln(cnhist,dnhist,r,10)

