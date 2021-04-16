import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
from tqdm import tqdm

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from skimage.transform.radon_transform import _get_fourier_filter

def heat(*args):
    heatmap(*args)
    plt.show()

def merge(a,b,res):
    if 0 in [len(a),len(b)]: return np.vstack((a,b))
    i = np.argmin([a[0,0], b[0,0]])
    res[0]  = [a[0], b[0]][i]
    res[1:] = merge(a[1-i:],b[i:],res[1:])
    return res

def y_from_x(x,rho,phi,N):
    y = -x*1/np.tan(phi) + rho/np.sin(phi)
    mask = (y>=0)*(y<=N)
    return x[mask], y[mask]
    
def x_from_y(y,rho,phi,N):
    x = -y*np.tan(phi) + rho/np.cos(phi)
    mask = (x>=0)*(x<=N)
    return x[mask], y[mask]

def line_int(F, N, rho_, phi):
    x0 = y0 = -N/2
    rho = rho_ - x0*np.cos(phi) - y0*np.sin(phi)
    if phi == 0:      return np.sum(F[int(rho)])
    if phi == np.pi:  return np.sum(F[:,int(rho)])
    
    rev = 1 - (-1/np.tan(phi)<0)*2
    steps = np.arange(N+1)
    
    points1 = np.vstack(y_from_x(steps,rho,phi,N)).T
    points2 = np.vstack(x_from_y(steps[::rev],rho,phi,N)).T
    
    res = np.zeros((len(points1) + len(points2),2))   
    points = merge(points1,points2,res)
    
    slope = -1/np.tan(phi)
    ratio = (1+slope**2)**0.5
    line_int_res = 0
    for i in range(len(points[1:])):
        p = points[i:i+2]
        dx = p[1,0] - p[0,0]
        [x,y] = (np.sum(p,axis=0)/2).astype(int)
        f = F[x,y]
        
        line_int_res += f*dx*ratio
    
    return line_int_res

def sinogram(F,N):
    S = np.zeros_like(F)
    rhos = np.linspace(-N/2, N/2, N, False) + 1/2
    phis = np.linspace(0, np.pi, N, False)
    
    for j, phi in enumerate(tqdm(phis)):
        for i,rho in enumerate(rhos):
            S[i,j] = line_int(F, N, rho, phi)
    
    return S, rhos, phis



def one_point(S,N,x,y,phis):
    fbi = 0
    for j,phi in enumerate(phis):
        rho_ = x*np.cos(phi) + y*np.sin(phi)
        i = max(0,min(N-1,int(rho_ + N/2 - 1/2)))
        fbi += S[i,j]
    
    return fbi

def backproject(S,phis,N):
    N = S.shape[0]
    Fb = np.zeros_like(S)
    
    for n in tqdm(range(N)):
        x = n - N/2 + 1/2
        for m in range(N):
            y = m - N/2 + 1/2
            Fb[n,m] = 1/np.pi * one_point(S,N,x,y,phis)
    return Fb


def filtered_backproject(S,phis,N):
    sino = np.rot90(S)
    fourier = np.zeros((N, N), dtype=np.cfloat)
    for i in range(N):
        for k in range(N):
            fourier[i, k] = np.exp(-2 * np.pi * 1j / N * i * k)/N

    reverse_fourier = np.zeros((N, N), dtype=np.cfloat)
    for i in range(N):
        for k in range(N):
            reverse_fourier[i, k] = np.exp(2 * np.pi * 1j / N * i * k)

    transformed = sino @ fourier
    ram_filter = _get_fourier_filter(N, "ramp")[:, 0]
    filtered_sino = transformed * ram_filter

    filtered_sino = filtered_sino @ reverse_fourier
    filtered_sino = np.real(filtered_sino)
    filtered_sino = np.rot90(filtered_sino,axes = (1,0))

    recon = backproject(filtered_sino, phis, N)
    return recon

names = ["original","sinogram","back","back_filtered","rhos","phis"]
def save(name,F,S,Fb,Fb_f,rhos,phis):
    for i,array in enumerate([F,S,Fb,Fb_f,rhos,phis]):
        np.save(f'{name}_{names[i]}',array)
        
def load(name):
    ret = []
    for n in names:
        ret.append(np.load(f'{name}_{n}.npy'))
    return ret

              
#%%
name = "shepp_logan_phantom"

#%% load new image and simulate ct scan
F = shepp_logan_phantom()
F = rescale(F, scale=0.2, multichannel=False)
N = min(F.shape)
F = F[:N,:N] 
S,rhos,phis = sinogram(F,N)
Fb = backproject(S,phis,N)
Fb_f = filtered_backproject(S,phis,N)


#%% load results
[F,S,Fb,Fb_f,rhos,phis] = load(name)


#%% save results
save(name,F,S,Fb,Fb_f,rhos,phis)


#%%
#print images
heat(F)
heat(S)
heat(Fb)
heat(Fb_f)

         
            
        
    
        
        
        
    
    
