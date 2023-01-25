import sim_utils as util
import numpy as np
import matplotlib.pyplot as plt
import time

x = np.linspace(0,1,10)
y = np.sin(x*np.pi*4)**2*np.cos(x)**3 + 1j*np.sin(x*np.pi)

n = 100
l = 1

nvals = np.array([2,5,10,30,50,80,100])*1.
lvals = np.array([1,2,3,4])
lvals = np.array([1])
errs = np.zeros((len(nvals),len(lvals)))

#for j in range(len(lvals)):
#    for i in range(len(nvals)):
#        nc = util.func2bessel(x,y,int(nvals[i]),l=lvals[j])
#        y2 = util.bessel2func(x,nc,l=lvals[j])
#        errs[i,j] = np.max(np.abs(y2-y))

#do some timing
t_forward = np.zeros(len(lvals))
t_backward = np.zeros(len(lvals))
for i in range(len(lvals)):
    tick = time.time()
    nc_t = util.func2bessel(x,y,3,l=lvals[i])
    tock = time.time()
    t_forward[i] = tock-tick
    tick = time.time()
#    y_t = util.bessel2func(x,nc_t,l=lvals[i])
    tock = time.time()
    t_backward[i] = tock-tick

print('mean forward time is: '+str(np.mean(t_forward))+', with SD of '+str(np.std(t_forward))+'.')
print('mean backward time is: '+str(np.mean(t_backward))+', with SD of '+str(np.std(t_backward))+'.')

#for i in range(len(lvals)):
#    plt.loglog(nvals,errs[:,i],label = 'Bessel expansion, l = '+str(lvals[i]))
#
#plt.loglog(nvals,8*nvals**(-2),'k--',label=r'$n^{-2}$')
#plt.loglog(nvals,8*nvals**(-1),'k--',label=r'$n^{-1}$')
#plt.legend()
#plt.title('Accuracy')
#plt.xlabel('n')
#plt.ylabel(r'$L_1$ error')
#plt.tight_layout()
#
#plt.figure()
#plt.subplot(2,1,1)
#plt.title('Function to expand, real part')
#plt.plot(x,np.real(y))
#plt.plot(x,np.real(y2),'--')
#plt.ylabel('y')
#plt.xlabel('x')
#plt.subplot(2,1,2)
#plt.title('Function to expand, imaginary part')
#plt.plot(x,np.imag(y))
#plt.plot(x,np.imag(y2),'--')
#plt.ylabel('y')
#plt.xlabel('x')
#plt.tight_layout()
#plt.show()
