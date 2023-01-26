import numpy as np
import matplotlib.pyplot as plt
import shtns
import sim_utils as utils
import pickle
from matplotlib.animation import FuncAnimation

def load_w_hist(name):
    with open('output_files/'+name+'.pickle','rb') as f:
        wahistdat = pickle.load(f)
        wbhistdat = pickle.load(f)
        mahistdat = pickle.load(f)
        mbhistdat = pickle.load(f)
        sh = pickle.load(f)
        r = pickle.load(f)
        simpars = pickle.load(f)
    return wahistdat,wbhistdat,mahistdat,mbhistdat,sh,r,simpars

def density_movie(bhistdat,sh,r,simpars,phi_init,phi0,besselzers = None,name = 'density_mov'):
    if besselzers is None:
        besselzers = np.loadtxt('zerovals.txt')

    nt = bhistdat.size[-1]

    [nlat,nphi] = sh.set_grid()
    cost = sh.cos_theta
    phi = np.linspace(0,2*np.pi,nphi)
    r = simpars.r
    nn = ahistdat.shape[1]
    sh.nvals = range(1,nn)
    [PH,CO,AR]  = np.meshgrid(phi,cost,r)

    phi_init_plane = phi_init[cost==min(cost)]
    phi_mov_plane = np.tile(phi_init_plane,[nt,1])
    for i in range(nt-1):
        dphi = -phi_0*(1-phi_0)*my_div(bhistdat[:,:,i].T)
        phi_mov_plane[:,:,:,i+1] = phi_mov[:,:,:,i] + dt*dphi

   


def planar_disk_frame(nf,ahistdat,bhistdat,sh,r,simpars,besselzers = None,along = None):
    if besselzers is None:
        besselzers = np.loadtxt('zerovals.txt')

    [nlat,nphi] = sh.set_grid()
    cost = sh.cos_theta
    phi = np.linspace(0,2*np.pi,nphi)
    r = simpars.r
    nn = ahistdat.shape[1]
    sh.nvals = range(1,nn)
    [PH,CO,AR]  = np.meshgrid(phi,cost,r)

    wanlm_this = ahistdat[:,:,nf].T
    wbnlm_this = bhistdat[:,:,nf].T

    wth,wph,wr = utils.my_sh_to_spat(wanlm_this,wbnlm_this,sh,simpars,besselzers)

    ## take the plane where cos(theta) = 0, aka theta = pi/2
    if along is None:
        wph_plane = wph[cost==np.min(cost),:,:][0]
        wph_planelines = wph[cost==np.min(cost),0,:][0]
        wph_planelines = np.vstack((wph_planelines,wph[cost==np.min(cost),int(nphi/4),:][0]))
        wph_planelines = np.vstack((wph_planelines,wph[cost==np.min(cost),int(nphi/2),:][0]))
        wph_planelines = np.vstack((wph_planelines,wph[cost==np.min(cost),int(3*nphi/4),:][0]))
    elif along == 'r':
        wph_plane = wr[cost==np.min(cost),:,:][0]
        wph_planelines = wr[cost==np.min(cost),0,:][0]
        wph_planelines = np.vstack((wph_planelines,wr[cost==np.min(cost),int(nphi/4),:][0]))
        wph_planelines = np.vstack((wph_planelines,wr[cost==np.min(cost),int(nphi/2),:][0]))
        wph_planelines = np.vstack((wph_planelines,wr[cost==np.min(cost),int(3*nphi/4),:][0]))
    elif along == 'th':
        wph_plane = wth[cost==np.min(cost),:,:][0]
        wph_planelines = wth[cost==np.min(cost),0,:][0]
        wph_planelines = np.vstack((wph_planelines,wth[cost==np.min(cost),int(nphi/4),:][0]))
        wph_planelines = np.vstack((wph_planelines,wth[cost==np.min(cost),int(nphi/2),:][0]))
        wph_planelines = np.vstack((wph_planelines,wth[cost==np.min(cost),int(3*nphi/4),:][0]))
    elif along == 'ph':
        wph_plane = wph[cost==np.min(cost),:,:][0]
        wph_planelines = wph[cost==np.min(cost),0,:][0]
        wph_planelines = np.vstack((wph_planelines,wph[cost==np.min(cost),int(nphi/4),:][0]))
        wph_planelines = np.vstack((wph_planelines,wph[cost==np.min(cost),int(nphi/2),:][0]))
        wph_planelines = np.vstack((wph_planelines,wph[cost==np.min(cost),int(3*nphi/4),:][0]))

    else:
        raise ValueError('Please enter a valid plotting direction')

    ar_plane = AR[cost==np.min(cost),:,:][0]
    ph_plane = PH[cost==np.min(cost),:,:][0]

    xp = ar_plane*np.cos(ph_plane)
    yp = ar_plane*np.sin(ph_plane)

    return xp,yp,wph_plane,wph_planelines

def animate_soln(wahistdat,wbhistdat,mahistdat,mbhistdat,sh,r,pars,besselzers = None,undersamp = 1,fname = 'bla',minmax=[-1,1]):
    if besselzers is None:
        besselzers = np.loadtxt('zerovals.txt')
    nt = wahistdat.shape[2]
    
    fig,ax = plt.subplots(2,2)
    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]
    ax4 = ax[1,1]
    xp0,yp0,ph_plane0,ph_line0 = planar_disk_frame(0,wahistdat,wbhistdat,sh,r,pars,besselzers)
    ax2.set_ylim(minmax)
    ax2.set_xlabel('r')
    ax2.set_ylabel(r'$w_{\phi}\tau/3a$')
    img = ax1.pcolormesh(xp0,yp0,np.zeros(ph_plane0.shape),vmin = minmax[0],vmax = minmax[1],cmap = 'magma',shading='gouraud')
    ax1.axis('off')
    ax1.set_title(r'$w_{\phi}(\theta = \pi/2)$')
    line1, = ax2.plot(r,ph_line0[0],linewidth=2,label=r'$\phi = 0$')
    line2, = ax2.plot(r,ph_line0[1],linewidth=2,label=r'$\phi = \pi/2$')
    line3, = ax2.plot(r,ph_line0[2],linewidth=2,label=r'$\phi = \pi$')
    line4, = ax2.plot(r,ph_line0[3],linewidth=2,label=r'$\phi = 3\pi/2$')
    ax2.legend(loc=1, prop={'size': 6})  
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    ax4.set_ylim(minmax)
    ax4.set_xlabel('r')
    ax4.set_ylabel(r'$w_{r}\tau/3a$')
    img2 = ax3.pcolormesh(xp0,yp0,np.zeros(ph_plane0.shape),vmin = minmax[0],vmax = minmax[1],cmap = 'magma',shading='gouraud')
    ax3.axis('off')
    ax3.set_title(r'$w_{r}(\theta = \pi/2)$')
    line21, = ax4.plot(r,ph_line0[0],linewidth=2,label=r'$\phi = 0$')
    line22, = ax4.plot(r,ph_line0[1],linewidth=2,label=r'$\phi = \pi/2$')
    line23, = ax4.plot(r,ph_line0[2],linewidth=2,label=r'$\phi = \pi$')
    line24, = ax4.plot(r,ph_line0[3],linewidth=2,label=r'$\phi = 3\pi/2$')
    ax4.legend(loc=1, prop={'size': 6})  
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    fig.colorbar(img,fraction=0.046, pad=0.04)
    fig.colorbar(img2,fraction=0.046, pad=0.04)
    fig.tight_layout()

    def animate(i):
        print('Frame '+str(i)+' / '+str(int(nt/undersamp)))
        xp,yp,wph_plane,wph_line = planar_disk_frame(i*undersamp,wahistdat,wbhistdat,sh,r,pars,besselzers,'ph')
        xp,yp,wr_plane,wr_line = planar_disk_frame(i*undersamp,wahistdat,wbhistdat,sh,r,pars,besselzers,'r')
        img.set_array(wph_plane.flatten())
        img2.set_array(wr_plane.flatten())
        line1.set_data(r,wph_line[0])
        line2.set_data(r,wph_line[1])
        line3.set_data(r,wph_line[2])
        line4.set_data(r,wph_line[3])
        
        line21.set_data(r,wr_line[0])
        line22.set_data(r,wr_line[1])
        line23.set_data(r,wr_line[2])
        line24.set_data(r,wr_line[3])

        return img,img2,line1,line2,line3,line4,line21,line22,line23,line24

    anim = FuncAnimation(fig, animate,frames=int(nt/undersamp), blit=True)
    
    anim.save('movies/'+fname+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


ha,hb,ma,mb,s,r,pars = load_w_hist('randomstart')
#x,y,p = planar_disk_frame(1,h,s,r)
#plt.pcolormesh(x,y,p)
#plt.show()
animate_soln(ha,hb,ma,mb,s,r,pars,None,1,'test_full',[-0.5,0.5])
