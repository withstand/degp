# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:50:18 2017

@author: qchen
"""
from scipy.stats import norm
from scipy.optimize import differential_evolution as de
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern as M
from sobol.sobol_seq import i4_sobol_generate as sg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def test_functions():
    import benchmark
    functions = dir(benchmark)
    functions.remove('Axes3D')
    functions.remove('Benchmark')
    functions = [getattr(benchmark,fi)(2) for fi in functions  if fi[0].isupper() and \
                 not fi.startswith('Problem') and \
                                  (getattr(benchmark,fi)().dimensions==2 or getattr(benchmark,fi)().change_dimensionality)]
    return functions


#obj_func = rosen
#obj_func = lambda x: 0.26 * (x[0]**2+x[1]**2)-0.48*x[0]*x[1]
#obj_func = lambda x: cec2015expensive.func(x, 2) 
#obj_func = cma.fcts.branin
#obj_func = cma.fcts.cigar
#obj_func = Ackley(2)
def plot_results(obj_func, bounds, pop, y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X, Y = np.meshgrid(np.linspace(bounds[0][0], bounds[0][1], 201), np.linspace(bounds[1][0], bounds[1][1], 201))
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = obj_func(np.asarray([X[i,j], Y[i,j]]))

    Z_gp = np.array([[gp.predict([[xij,yij]])[0] for xij, yij in zip(xi,yi)] for xi,yi in zip(X,Y)])
    # Plot the surface.
#    ax.plot_surface(X, Y, Z-Z_gp, rstride=1, cstride=1, linewidth=0.0, shade=True )
#    ax.plot_surface(X, Y, Z_gp, rstride=1, cstride=1, linewidth=0.0, shade=True )
    
    plt.contour(X,Y,Z-Z_gp, zdir='z',offset=ax.get_zlim()[0], shade=True)
#    ax.scatter(xs=[xy[0] for xy in pop], ys=[xy[1] for xy in pop], zs=y,color='red')
    # Customize the z axis.
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)    
    plt.show()
    
def optimize_gp(obj_func,disp=True):
   
    dim = obj_func.dimensions 
    bounds = obj_func.bounds
    
    def apply_bounds(x, bounds):
        return np.array([xi * (lub[1]-lub[0]) + lub[0] for xi, lub in zip(x,bounds)])
    
    def de_optimizer(obj_func, initial_theta, bounds): 
        result = de(lambda x:obj_func(x)[0], [(b[0], b[1]) for b in bounds]  )        
        return result.x, result.fun
    
    
    kernel =  C(1.0, (1e-3, 1e3)) * RBF([1]*dim, [(1e-2, 1e2)]*dim)
    #kernel = C(1.0, (1e-3, 1e3)) * M([1]*dim, [(1e-2, 1e2)]*dim, nu=1.5)
    gp = GPR(kernel=kernel, optimizer=de_optimizer)
    
    
    n = dim * 10 + 1
    N0 = n
    Nmax = 20 * dim
    
    pop = [apply_bounds(sgi, bounds) for sgi in sg(dim, n, 1).transpose()]
    
    y= [obj_func(pi) for pi in pop]
    
    bestx = [pop[i] for i in range(len(y)) if y[i]==min(y)]
    besty = [y[i] for i in range(len(y)) if y[i]==min(y)]
   
    
    def eval_gp_model(x):
        y,s = gp.predict([x], return_std=True)
        return y[0],s[0]
    
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
        t = (min(besty)-y)/s
        return -1* ((min(besty)-y)*norm.cdf(t)+s*norm.pdf(t))
    
    def pi(x):
        y,s = eval_gp_model(x)
        if s<=0: return np.inf  
        return -norm.cdf((min(besty)-y)/s)   
    
    
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
    
    while len(pop) < Nmax:
        d = [(b[1]-b[0])*10**(-1.1-2.1*(len(pop)-N0)/(Nmax-N0)) for b in bounds]
        
        gp.fit(pop, y)
      
#        print(bounds)
#        print([f(obj_func.generator()) for f in funcs])
        attrac_center = [de(f, bounds).x for f in funcs]
        
        
        while True:
            new_sample = apply_bounds(sg(dim,1, skip).transpose()[0], bounds)
            skip = skip + 1
          
            filter_on = [0.5 <= mdist(new_sample, aci, di)  for aci,di in zip(attrac_center,d)]
            if any(filter_on):
                choice = filter_on.index(True)
                break
       
        new_y = obj_func(new_sample)  
        if np.isnan(new_y) or np.isinf(new_y):
            continue
        pop.append(new_sample)
        y.append(new_y)
        if (new_y <= min(besty)):
            bestx.append(new_sample)
            besty.append(new_y)
            bestx = [bestx[i] for i in range(len(bestx)) if besty[i]==min(besty)]
            besty = [besty[i] for i in range(len(besty)) if besty[i]==min(besty)]
        if disp:
            print(len(pop), max(d), funcs[choice].__name__.upper(), skip, new_sample, new_y, bestx[0], besty[0],sep='\t',end='\n')
    
    return {'pop':pop, 'y':y, 'bestx':bestx, 'besty':besty, 'model':gp,'func':obj_func}



if __name__ == "__main__":
    import warnings
    warnings.filters.append(('ignore', None, UserWarning,None,0))
    fcs = test_functions()

    np.random.seed(1)   
    run = 10
    run_seed = []
    for i in range(len(fcs)):
        run_seed.append([])
        for j in range(run):
            run_seed[i].append(np.random.randint(65535))
    
    
    results = {}
    
    for f in fcs:
        results[str(f)]=[]
        for i in range(run):
            try:
                np.random.seed(run_seed[fcs.index(f)][i])
                print('Optimize ', str(f), ' Run ', i, ' Seed:', run_seed[fcs.index(f)][i])
                ret = optimize_gp(f, disp=False)
                results[str(f)].append(ret)
                print(ret['bestx'], ret['besty'])
            except Exception as e:
                print('Error optimize ', str(f), ':', e)
        np.save(str(f), results[str(f)])
                
            
