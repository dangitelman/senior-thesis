from scipy.optimize import minimize
from IPython.display import clear_output
from time import sleep
import warnings
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def plot_progress(mu,alpha,LL,e,names,save_path,title):
    fig = plt.figure(figsize=(15,7))
    #if title != False:
    #    st = fig.suptitle(title,y=1.05,fontsize=25)
    gs = GridSpec(4,3,figure=fig)
    ax_mu = fig.add_subplot(gs[0,:2])
    ax_alpha = fig.add_subplot(gs[1:,:2])
    ax_LL1 = fig.add_subplot(gs[:2,2])
    ax_LL2 = fig.add_subplot(gs[2:,2])

    matrix = mu.copy()
    img = ax_mu.imshow([matrix])
    divider = make_axes_locatable(ax_mu)
    cax = divider.append_axes('bottom', size='10%', pad=0.3)
    fig.colorbar(img, cax=cax, orientation='horizontal')
    ax_mu.set_xticks(np.arange(0,matrix.shape[0], matrix.shape[0]*1.0/len(names)))
    ax_mu.set_xticklabels(names)
    ax_mu.set_title('$μ_i$')
    ax_mu.set_xlabel('$i$')

    matrix = alpha.copy()
    img = ax_alpha.imshow(matrix)
    divider = make_axes_locatable(ax_alpha)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(img, cax=cax, orientation='vertical')
    ax_alpha.set_xticks(np.arange(0,matrix.shape[0], matrix.shape[0]*1.0/len(names)))
    ax_alpha.set_yticks(np.arange(0,matrix.shape[1], matrix.shape[1]*1.0/len(names)))
    ax_alpha.set_xticklabels(names,rotation = 90)
    ax_alpha.set_yticklabels(names)
    ax_alpha.set_title('$α_{i,j}$',fontsize = 15)
    ax_alpha.set_xlabel('$j$')
    ax_alpha.set_ylabel('$i$')

    if e > 3:
        ax_LL1.plot(list(range(2,len(LL)+1)),np.array(LL[1:]))
        if e > 7:
            ax_LL2.plot(list(range(2,len(LL)+1))[-5:],np.array(LL[1:])[-5:])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path != False:
        plt.savefig(save_path, dpi=300)
    plt.show()

def training_prep(data,lag,time_dep,omega):
    U = len(data)
    t = np.concatenate([np.array([data[i],np.full(len(data[i]),i)]).T for i in range(U)])
    t = t[t[:,0].argsort()]
    u = t[:,1].astype(int)
    t = t[:,0]
    N = len(t)
    T = t[-1]
    
    mu = np.random.uniform(0,1,size=U)
    alpha = np.random.uniform(0,1,size=(U,U))
    
    mask = [[[[],[]] for j in range(U)] for i in range(U)]
    ids_matrix = np.array([np.arange(i-lag,i) for i in range(len(t))])
    rows_matrix = np.array([np.full(lag,i) for i in range(len(t))])
    val_masks = [(u[ids_matrix]==i) & (ids_matrix >=0) for i in range(U)]
    ids_matrix -= rows_matrix
    for i in tqdm(range(U)):
        for j in range(U):
            mask[i][j][0] = rows_matrix[u==i][val_masks[j][u==i]]
            mask[i][j][1] = ids_matrix[u==i][val_masks[j][u==i]]
            
    B = np.zeros((N,lag))
    C = [[]]
    for i in tqdm(range(N-1)):
        start = max(0,i+1-lag)
        D = np.exp(-omega*(t[i+1]-t[i]))
        B[i+1,:] = np.concatenate([B[i,1:],[1]])*D
        Ci = np.array(list(range(start,i+1)))
        if len(Ci) != lag:
            pad = lag - len(Ci)
            Ci = np.concatenate([np.zeros(pad),Ci])
        C.append(Ci.astype(int))
    C[0] = C[1]
    alpha_mask = [np.array([[u[i]] for i in range(N)]),
                  np.array([u[C[i]] for i in range(N)])]
    
    TIME_DEP = [y.real[int(np.floor(t[i]) % 1440)] if time_dep else 1 for i in range(N)]
    val_array = [np.concatenate(np.argwhere(u==i)) for i in range(U)]
    LL_mask = np.ix_(list(range(U)), u[list(range(N))])
    
    return U,N,T,u,t,mu,alpha,mask,B,alpha_mask,TIME_DEP,val_array,LL_mask

def EM(data,omega,names,lag,time_dep=False,PRIORS=[1,5],use_loglike=True,save_path=False,title=False):
    U,N,T,u,t,mu,alpha,mask,B,alpha_mask,TIME_DEP,val_array,LL_mask = training_prep(data,lag,time_dep,omega)
    
    mus = [mu.copy()]
    alphas = [alpha.copy()]
    LL = []
    
    e = 1
    while(True):
        clear_output(wait=True)
        print('EPOCH {}'.format(e))
        
        if e > 2:
            if use_loglike:
                if LL[-1] - LL[-2] < 0.01:
                    plot_progress(mu,alpha,LL,e,names,save_path,title)
                    return mu,alpha,omega
            else:
                if max(np.max(abs(mus[-1]-mus[-2])),np.max(abs(alphas[-1]-alphas[-2]))) < 0.0001:
                    plot_progress(mu,alpha,LL,e,names,save_path,title)
                    return mu,alpha,omega
        
        """EXPECTATION STEP"""
        G = alpha[alpha_mask]
        H = G*B
        A = np.sum(H,axis=1)*omega
        denom = mu[u]*TIME_DEP + A
        p = H*omega/denom[:,None]
        q = mu[u]/denom
           
        """LOG LIKELIHOOD AND PLOTTING"""
        if use_loglike:
            LLe = np.sum(np.log(mu[u] + A/omega))
            LLe -= T*np.sum(mu)
            LLe -= np.sum(alpha[LL_mask]*(1-np.tile(np.exp(-omega*(T-t)),(U,1))))

        """MAXIMIZATION STEP"""
        for i in range(U):
            mu[i] = np.sum(q[u==i])/T
            for j in range(U):
                numerator = np.sum(p[mask[i][j]])
                numerator += PRIORS[0] - 1
                denominator = len(val_array[j]) + PRIORS[1]
                alpha[i,j] = numerator/denominator

        if use_loglike:
            LL.append(LLe)
        mus.append(mu.copy())
        alphas.append(alpha.copy())
        
        e += 1