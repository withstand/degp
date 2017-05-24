# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:50:18 2017

@author: qchen
"""
from scipy.stats import norm
from scipy.optimize import differential_evolution as de
import numpy as np
import cec2015expensive
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern as M
from sobol.sobol_seq import i4_sobol_generate as sg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import rosen

import cma



#obj_func = rosen
#obj_func = lambda x: 0.26 * (x[0]**2+x[1]**2)-0.48*x[0]*x[1]
#obj_func = lambda x: cec2015expensive.func(x, 2) 
#obj_func = cma.fcts.branin
obj_func = cma.fcts.cigar

np.random.seed(1)

dim = 2
lb = -15
ub = 10

bounds = [(lb, ub)]*dim
# result0 = differential_evolution(rosen, bounds)


def de_optimizer(obj_func, initial_theta, bounds):
    # * 'obj_func' is the objective function to be maximized, which
    #   takes the hyperparameters theta as parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta
    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.
    
    result = de(lambda x:obj_func(x)[0], [(b[0], b[1]) for b in bounds]  )
    
    return result.x, result.fun


kernel =  C(1.0, (1e-3, 1e3)) * RBF([1]*dim, [(1e-2, 1e2)]*dim)
#kernel = C(1.0, (1e-3, 1e3)) * M([1]*dim, [(1e-2, 1e2)]*dim, nu=1.5)
gp = GPR(kernel=kernel, optimizer=de_optimizer)


n = dim * 10 + 1
N0 = n
Nmax = 50 * dim

pop = list((sg(dim, n, 1)*(ub-lb)+lb).transpose())

y= [obj_func(pi) for pi in pop]

bestx = [pop[i] for i in range(len(y)) if y[i]==min(y)]
besty = [y[i] for i in range(len(y)) if y[i]==min(y)]
besty = besty[0]
bestx = bestx[0]




def eval_gp_model(x):
    try:
        y,s = gp.predict([x], return_std=True)
    finally:
        return y,s

def func1(x):
    y,s = eval_gp_model(x)
    return y - 3*s

def func2(x):
    y,s = eval_gp_model(x)
    return y - 2*s  

def func3(x):
    y,s = eval_gp_model(x)
    return y - 1*s  

def func4(x):
    y,s = eval_gp_model(x)
    return y

def func5(x):
    y,s = eval_gp_model(x)
    return y + 1*s 
def func6(x):
    y,s = eval_gp_model(x)
    return y + 2*s  

def func7(x):
    y,s = eval_gp_model(x)
    return y + 3*s 

def ei(x):
    y,s = eval_gp_model(x)
    if s<=0: return np.inf
    t = (besty-y)/s
    return -1* ((besty-y)*norm.cdf(t)+s*norm.pdf(t))

def pi(x):
    y,s = eval_gp_model(x)
    if s<=0: return np.inf  
    return -norm.cdf((besty-y)/s)



def distance(x1, x2):
    dx = x1 - x2
    return np.sqrt((dx*dx).sum())

def distance2(x1, x2):
    dx = x1 - x2
    return np.abs(dx).max()

def mdist(x1, x2, di):
    return np.exp(-np.log(2) * distance2(x1,x2)/di)
    
skip = len(pop) + 1    
funcs = [ ei, pi,func1, func2, func3, func4, func5, func6, func7]
#d = len(funcs) * [(ub-lb)*0.1]

#dmax = distance(np.array([lb]*dim), np.array([ub]*dim))

#print(bestx, besty)
#print('----------------------------------------\n')

while len(pop) < Nmax:
    # from 10**-1 to 10**-3
    d = len(funcs) * [(ub-lb)*10**(-1.1-3*(len(pop)-N0)/(Nmax-N0))]
    
    gp.fit(pop, y)
    
#    new_sample = differential_evolution(func7, bounds).x
    
    attrac_center = [de(f, bounds).x for f in funcs]
#    print(attrac_center)
#    print('Allocate attraction center', end='\t')
#
    while True:
        new_sample = sg(dim,1, skip).transpose()[0]*(ub-lb)+lb
        skip = skip + 1
        mdist_new = [mdist(new_sample, aci, di)  for aci,di in zip(attrac_center,d)]
        
        filter_on = [0.5 <= mdist(new_sample, aci, di)  for aci,di in zip(attrac_center,d)]
        if any(filter_on):
            choice = filter_on.index(True)
#            print(filter_on)
#            print(mdist_new)
            break
#    choice = np.random.choice(range(len(attrac_center)))
#    new_sample = attrac_center[choice]
#    print( attrac_center)
    
    new_y = obj_func(new_sample)   
    pop.append(new_sample)
    y.append(new_y)
    if (new_y <= besty):
        bestx = new_sample
        besty = new_y
    print(len(pop), max(d), funcs[choice].__name__.upper(), skip, new_sample, new_y, bestx, besty,sep='\t',end='\n')


#import matplotlib.pyplot as plt
#from scipy.spatial.distance import cdist
#    
#def test_sobol():
#    ni = np.arange(1, 20001, 10)
#    maxmin = []
#    sample = sg(dim,ni[-1],1).T
#    cd = cdist(sample, sample)
#    np.fill_diagonal(cd, np.Inf)
#    maxmin = [cd[:nii,:nii].min(axis=0).max() for nii in ni]
#    plt.plot(ni, maxmin)
        
def plot_results():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X = np.arange(lb, ub, 0.1)
    Y = np.arange(lb, ub, 0.1)
    X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)
    
    Z = np.array([obj_func([xi,yi]) for xi,yi in zip(X,Y)])
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z)
    ax.scatter(xs=[xy[0] for xy in pop], ys=[xy[1] for xy in pop], zs=y,color='red')
    # Customize the z axis.
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    
