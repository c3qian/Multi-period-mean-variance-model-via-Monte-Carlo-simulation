# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from   scipy.sparse import spdiags, identity
from   scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
#from   bokeh.plotting import figure, output_notebook, show, gridplot, save
import pandas as pd
import math
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator
import itertools

# %%
def simulateReturnPathsGBM(r, sigma, xi, dt, numOfPath, timesteps,seed):
    '''
    :timeSteps = 81, with the first columnn as 0
    :increZ = 5000 * 81
    :return log returns : 5000 * 80, increZ truncated 5000*80
    '''
    S0 = 1
    np.random.seed(seed)
    sigmaxi = sigma*xi
    timeSteps = len(timesteps) + 1
    increZ = np.random.normal(0, 1, numOfPath * timeSteps).reshape(numOfPath,timeSteps)
    logReturns = np.zeros_like(increZ)
    for i in range(timeSteps):
        if i == 0:
            logReturns[:,i] = 0
        else:
            logReturns[:,i] = (r + sigmaxi - sigma**2/2) * dt + sigma * np.sqrt(dt) * increZ[:,i]

    return logReturns[:,1:],increZ[:,1:]


# %%
# Cong and Oos (2016)
# multi-period
S0 = 1
r  = 0.03 #since r is log return
xi = 0.33 
sigma = 0.15
C = 0.1
# should be 50,000
numOfPath = 50000
#gamma = 14.47
#gamma = 40
M = 80 # rebalancing opportunities
T = 20 
dt = T / M
W0 = 1.0
Pmax = 1.5
Pmin = 0
alpha = 10** -8
timesteps = np.linspace( 0 , T, M + 1 )[:-1]
sid = 10

# %%
def multiStageStrategy(gamma,returns,increments,S0,r,xi,sigma,C,numOfPath,M,T,dt,W0,Pmax,Pmin,alpha,timesteps): 
    """
    : returns 500*80
    : timesteps: 80, [0,...,T-dt]
    :RETURN Wealth, Wealth_bdd, Wealth_nobkcy of shape (500,1) -- the last column
    """   
    returns = np.exp(returns) - np.exp(r)
    template = np.arange(numOfPath*(len(timesteps)+1)).reshape(numOfPath,len(timesteps)+1)
    wealth = np.zeros_like(template)
    wealth_nobkcys = np.zeros_like(template)
    wealth_bdds = np.zeros_like(template)
    wealth[:,0] = W0
    wealth_nobkcys[:,0] = W0
    wealth_bdds[:,0] = W0
    optPs = np.zeros_like(returns)
    optP_bdds = np.zeros_like(returns)
    optP_nobkcys = np.zeros_like(returns)

    def simulatePath_periodic(gamma, numOfPath,increments):
        for j in range(numOfPath):
            for i,t in enumerate(timesteps):
                returns_i = returns[:,i]
                alphaQuan = np.quantile(np.array(returns_i),float(alpha))
                oneMinusAlphaQuan = np.quantile(np.array(returns_i),float(1-alpha))
                denom = wealth[j][i] *np.mean(returns_i**2)
                fraction = (1 - r**((T - t + dt)/dt))/(1 - r)
                #print(denom)
                optP = (gamma/2 - C * dt*(fraction + 1) - wealth[j][i] *r )* np.mean(returns_i)/ denom
                optP_nobkcy = optP
                optP_bdd = optP 
                #bdd control
                if (optP_bdd < Pmax) and (optP_bdd > Pmin):
                    optP_bdd = optP_bdd 
                elif optP_bdd > Pmax:
                    optP_bdd = Pmax
                else:
                    optP_bdd = Pmin
                # no bankruptcy
                upper = (-C*dt -wealth_nobkcys[j][i]*r )/(wealth_nobkcys[j][i] * alphaQuan)
                lower = (-C*dt -wealth_nobkcys[j][i]*r )/(wealth_nobkcys[j][i] * oneMinusAlphaQuan)
                if( optP_nobkcy < upper) and ( optP_nobkcy >lower):
                    optP_nobkcy = optP_nobkcy
                elif optP_nobkcy > upper:
                    optP_nobkcy = upper
                else:
                    optP_nobkcy = lower

                optP_bdds[j][i] = optP_bdd
                optP_nobkcys[j][i] = optP_nobkcy
                optPs[j][i] = optP
                wealth[j][i+1] = wealth[j][i] * (optP*returns_i[j] + r) + C * dt
                wealth_nobkcys[j][i+1] = wealth_nobkcys[j][i] * (optP_nobkcy*returns_i[j] +r) + C * dt
                wealth_bdds[j][i+1] = wealth_bdds[j][i] * (optP_bdd*returns_i[j] +r) + C * dt
        return wealth[:,-1],wealth_nobkcys[:,-1],wealth_bdds[:,-1]

    termWealth, termWealth_nobkcy,termWealth_bdd = simulatePath_periodic(gamma,numOfPath,increments)
    return termWealth, termWealth_nobkcy,termWealth_bdd,optPs,optP_bdds,optP_nobkcys
# %%
def multiStageStrategyVectorize(gamma,returns,increments,S0,r,xi,sigma,C,numOfPath,M,T,dt,W0,Pmax,Pmin,alpha,timesteps): 

    """
    : returns 500*80
    : timesteps: 80, [0,...,T-dt]
    :RETURN Wealth, Wealth_bdd, Wealth_nobkcy of shape (500,1) -- the last column
            ctrls for three cases, they are of dimension 500*80
    """   
    returns = np.exp(returns) - np.exp(r)
    R = np.exp(r)
    template = np.arange(numOfPath*(len(timesteps)+1)).reshape(numOfPath,len(timesteps)+1)
    #template = np.array([[0 for i in range(len(timesteps)+1)] for j in range(numOfPath)])
    wealth = np.zeros_like(template,dtype=float)
    wealth_nobkcys = np.zeros_like(template,dtype=float)
    wealth_bdds = np.zeros_like(template,dtype=float)
    wealth[:,0] = W0
    wealth_nobkcys[:,0] = W0
    wealth_bdds[:,0] = W0
    optPs = np.zeros_like(returns,dtype=float)
    optP_bdds = np.zeros_like(returns,dtype=float)
    optP_nobkcys = np.zeros_like(returns,dtype=float)
    PmaxList = np.array([Pmax for i in range(numOfPath)])
    PminList = np.array([Pmin for i in range(numOfPath)])
    def simulatePath_periodic(gamma, numOfPath,increments):
        for i,t in enumerate(timesteps):
            returns_i = returns[:,i]
            alphaQuan = np.quantile(np.array(returns_i),float(alpha))
            oneMinusAlphaQuan = np.quantile(np.array(returns_i),float(1-alpha))
            denom = wealth[:,i] *np.mean(returns_i**2)
            fraction = (1 - R**((T - t + dt)/dt))/(1 - R)
            #print(denom)
            optP = np.divide((gamma/2 - C * dt*(fraction + 1) - wealth[:,i] *R )* np.mean(returns_i), denom)
            optP_nobkcy = optP.copy()
            optP_bdd = optP.copy()
            #bdd control
            indG = np.greater(optP_bdd, PmaxList)
            optP_bdd[indG] = PmaxList[indG]
            indL = np.less(optP_bdd, PminList)
            optP_bdd[indL] = PminList[indL]

            # no bankruptcy
            res1 = -C*dt -wealth_nobkcys[:,i]*R 
            res2 = wealth_nobkcys[:,i] * alphaQuan
            upper = np.divide(res1 ,res2)
            lower = np.divide(-C*dt -wealth_nobkcys[:,i]*R ,wealth_nobkcys[:,i] * oneMinusAlphaQuan)
            indG_nobkcy = np.greater(optP_nobkcy, upper)
            optP_nobkcy[indG_nobkcy] = upper[indG_nobkcy]
            indL_nobkcy = np.less(optP_nobkcy, lower)
            optP_nobkcy[indL_nobkcy] = lower[indL_nobkcy]

            optP_bdds[:,i] = optP_bdd
            optP_nobkcys[:,i] = optP_nobkcy
            optPs[:,i] = optP
            colTemp = np.multiply(wealth[:,i] ,(np.multiply(optP,returns_i) + R)) + C * dt
            wealth[:,i+1] = colTemp
            wealth_nobkcys[:,i+1] = np.multiply(wealth_nobkcys[:,i] ,(np.multiply(optP_nobkcy,returns_i) +R)) + C * dt
            wealth_bdds[:,i+1] = np.multiply(wealth_bdds[:,i] ,(np.multiply(optP_bdd,returns_i) +R)) + C * dt
        return wealth[:,-1],wealth_nobkcys[:,-1],wealth_bdds[:,-1]

    termWealth, termWealth_nobkcy,termWealth_bdd = simulatePath_periodic(gamma,numOfPath,increments)
    return termWealth, termWealth_nobkcy,termWealth_bdd,optPs,optP_bdds,optP_nobkcys
# %%

def multiStageStrategyVectorizeWhole(gamma,Type,returns,increments,S0,r,xi,sigma,C,numOfPath,M,T,dt,W0,Pmax,Pmin,alpha,timesteps): 
    
    """
    : returns 500*80
    : timesteps: 80, [0,...,T-dt]
    :RETURN Wealth, Wealth_bdd, Wealth_nobkcy of shape (500,1) -- the last column
            ctrls for three cases, they are of dimension 500*80 
            type: np.array
    """   
    returns = np.exp(returns) - np.exp(r)
    R = np.exp(r)
    template = np.arange(numOfPath*(len(timesteps)+1)).reshape(numOfPath,len(timesteps)+1)
    #template = np.array([[0 for i in range(len(timesteps)+1)] for j in range(numOfPath)])
    wealth = np.zeros_like(template,dtype=float)
    wealth_nobkcys = np.zeros_like(template,dtype=float)
    wealth_bdds = np.zeros_like(template,dtype=float)
    wealth[:,0] = W0
    wealth_nobkcys[:,0] = W0
    wealth_bdds[:,0] = W0
    optPs = np.zeros_like(returns,dtype=float)
    optP_bdds = np.zeros_like(returns,dtype=float)
    optP_nobkcys = np.zeros_like(returns,dtype=float)
    PmaxList = np.array([Pmax for i in range(numOfPath)])
    PminList = np.array([Pmin for i in range(numOfPath)])
    def simulatePath_periodic(gamma, numOfPath,increments):
        for i,t in enumerate(timesteps):
            returns_i = returns[:,i]
            alphaQuan = np.quantile(np.array(returns_i),float(alpha))
            oneMinusAlphaQuan = np.quantile(np.array(returns_i),float(1-alpha))
            denom = wealth[:,i] *np.mean(returns_i**2)
            fraction = (1 - R**((T - t + dt)/dt))/(1 - R)
            #print(denom)
            optP = np.divide((gamma/2 - C * dt*(fraction + 1) - wealth[:,i] *R )* np.mean(returns_i), denom)
            optP_nobkcy = optP.copy()
            optP_bdd = optP.copy()
            #bdd control
            indG = np.greater(optP_bdd, PmaxList)
            optP_bdd[indG] = PmaxList[indG]
            indL = np.less(optP_bdd, PminList)
            optP_bdd[indL] = PminList[indL]

            # no bankruptcy
            res1 = -C*dt -wealth_nobkcys[:,i]*R 
            res2 = wealth_nobkcys[:,i] * alphaQuan
            upper = np.divide(res1 ,res2)
            lower = np.divide(-C*dt -wealth_nobkcys[:,i]*R ,wealth_nobkcys[:,i] * oneMinusAlphaQuan)
            indG_nobkcy = np.greater(optP_nobkcy, upper)
            optP_nobkcy[indG_nobkcy] = upper[indG_nobkcy]
            indL_nobkcy = np.less(optP_nobkcy, lower)
            optP_nobkcy[indL_nobkcy] = lower[indL_nobkcy]

            optP_bdds[:,i] = optP_bdd
            optP_nobkcys[:,i] = optP_nobkcy
            optPs[:,i] = optP
            colTemp = np.multiply(wealth[:,i] ,(np.multiply(optP,returns_i) + R)) + C * dt
            wealth[:,i+1] = colTemp
            wealth_nobkcys[:,i+1] = np.multiply(wealth_nobkcys[:,i] ,(np.multiply(optP_nobkcy,returns_i) +R)) + C * dt
            wealth_bdds[:,i+1] = np.multiply(wealth_bdds[:,i] ,(np.multiply(optP_bdd,returns_i) +R)) + C * dt
        return wealth,wealth_nobkcys,wealth_bdds

    Wealth, Wealth_nobkcy,Wealth_bdd = simulatePath_periodic(gamma,numOfPath,increments)
    if Type == 'ms':
        return Wealth,optPs
    elif Type == 'nobkcy':
        return Wealth_nobkcy,optP_nobkcys
    else:
        return Wealth_bdd,optP_bdds

#%%
# Monte Carlo simulation with controls
# This should be executed after the backward recursive programming
def simulatePathsUseCtrls(r, W0, C,increZ, returns,ctrls,dt):
    # increZ = np.random.normal(0, 1, numOfPath*len(ctrls)).reshape(numOfPath,len(ctrls))
    Wealth = np.zeros_like(increZ)
    returns = np.exp(returns) - r
    for i in range(80):
        if i == 0:
            Wealth[:,i] = W0
        else:
            Wealth[:,i] = np.multiply(Wealth[:,i-1],np.multiply(ctrls[:,i],returns[:,i]) + r) + C*dt
    return Wealth[:,-1]
# %%
# utility functions
# simulate paths backward
def simulatePathsBackwardStrategy(wealth, intiCtrls:list, returns:list, C, r, dt, numOfPath,timesteps):
    W0 = 1
    for j in range(numOfPath):
        for i,t in enumerate(timesteps):
            if i == 0:
                wealth[j][i] = W0 * ( intiCtrls[j][i] * returns[j][i] + r ) + C * dt
            else:
                wealth[j][i] = wealth[j][i-1] * ( intiCtrls[j][i] * returns[j][i] + r ) + C * dt
    return wealth

# analytical solution for allow bankruptcy case
# params to be initialize later
def analyticSolution():
    lastPart = np.exp(-r*(T-t) + xi**2 * T)/(2* gamma)
    optP = -xi/(sigma * W) * (W - (W0*np.exp(r*t) + pi/r * (np.exp(r*t) - 1)) - lastPart)
    termWealth = simulatePaths(r, sigmaxi, W0, pi, sigma,optP,100)
    exptW = W0 * np.exp(r*T) + pi* (np.exp(r*T) - 1)/r + np.sqrt(np.exp(xi**2 * T) - 1)  * np.std(termWealth)
    lamb_da = 0.5* math.pow(gamma/2 - exptW, -1)
    varW = np.exp(xi**2 * T - 1)/(4* lamb_da**2)
    
# partition paths
def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

# partition based on the value similarity
def wealthPartition(pathsValues :list, n):
    """
    pathsValues: the wealth values at time t, dim (500,1)
    n: num of bundles
    :return [[wealth list index] ]
    """
    numInBundle = int(len(pathsValues)/n)
    order = sorted(range(len(pathsValues)), key=lambda k: pathsValues[k])
    return [order[x:x + numInBundle] for x in range(0, len(order), numInBundle)]


# regression using basis function
# https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
def quadraticFit(x,y,degree):
    polynomial_features= PolynomialFeatures(degree)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    return model

# costomized interpolation method
def interpolation(myArrayOne:np.array, myArrayTwo:np.array, value):
    leftIdx = myArrayOne.tolist().index(min(myArrayOne, key=lambda x:abs(x-value)))
    rightIdx = leftIdx + 1
    proportion = (value - myArrayOne[leftIdx])/(myArrayOne[rightIdx] - myArrayOne[leftIdx])
    return myArrayTwo[leftIdx] + proportion* (myArrayTwo[rightIdx] - myArrayTwo[leftIdx])

# %%
# backward recursion method
def backwardStrategyTest(gamma,Type,fromMultiPeriod,RETURNS,INCRE,seed):
    """
    PARAMETERS: 
        gamma: 2 * target wealth
        fromMultiPeriod: bool, which determines whether we use the multiperiod results
        termiWealth: from multi-period method dim 500*1
        RETURNS: 50,000 by 80
        INCRE: 50,000 by 80
    """
    r = 0.03
    sigma = 0.15
    xi = 0.33
    C = 0.1
    T = 2
    W0 = 1
    M = 80
    numOfPath = 50000
    numOfBundles = 20
    dt = T/M
    # Xmax = 1 + C*dt / 
    # every path the same? no
    timesteps = np.linspace( 0.0, T, M + 1 )[:-1] # 80, starts at 0
    # for iteration in range(num):
    if fromMultiPeriod:
        returns,increments  =  RETURNS,INCRE
        returns = np.exp(returns) - 1
        wealthProcess, optPs = multiStageStrategyVectorizeWhole(gamma,Type,returns,increments,S0,r,xi,sigma,C,numOfPath,M,T,dt,W0,Pmax,Pmin,alpha,timesteps)
        intiCtrls = optPs
        # at terminal time T
        initContiValue = (wealthProcess[:,-1] - gamma/2)**2
    else:
        intiCtrls = np.random.uniform(0,1.5,M * numOfPath ).reshape( numOfPath, M)
        returns = simulateReturnPathsGBM(r, sigma, xi,dt, numOfPath,timesteps,seed)
        returns = np.exp(returns) - r
        wealth = np.zeros_like(returns)
        wealthProcess = simulatePathsBackwardStrategy(wealth,intiCtrls,returns, C, r, dt, numOfPath,timesteps)
        initContiValue = list(map(lambda x : x**2, wealthProcess[:,-1] - gamma/2 ))
    
    # bundlePaths = dict([(str(idx) + '_' + str(i) ,wealthProcess[i,:])for idx,bundle in enumerate(bundles) for i in bundle]) 
    
    # backward recursion
    # starts form t = 79, index 78
    for i in range(len(timesteps)-2,-1,-1):
        optPsList = [0 for i in range(numOfBundles)]
        if i == (len(timesteps)-2):
            contiValuetList = initContiValue.copy()
        else:
            contiValuetList = nextContiValuetList
        #[[3,2,4],[0,1,5]]
        bundles = wealthPartition(wealthProcess[:,i].tolist(), numOfBundles)
        for idx , bundle in enumerate(bundles):
            optPsList[idx] = []
            wealthValueInBundle = []
            contiValueInBundle = []
            # W_t+dt
            # projected wealth level
            wealthValueInBundle = wealthProcess[bundle,i+1]
            contiValueInBundle = contiValuetList[bundle]
            # this is the same for every paths in a bundle
            # t+dt
            localApprox_1 = quadraticFit(np.array(wealthValueInBundle),np.array(contiValueInBundle),2)

            
            # t+dt
            # fist order condition - this might be wrong
            xhat = localApprox_1.coef_[0][1] + 2* localApprox_1.coef_[0][2] * wealthValueInBundle
            optPsList[idx] = xhat #len 78
            # step 2.3
            #temp = xhat * returns[path][i+1] + r 
            tempWealth = np.multiply(wealthProcess[bundle,i],(np.multiply(xhat,(returns[bundle,i] - r)) + r)) + C*dt
            ## regression again
            # should I use previous coefs? -no
            localApprox_2 = quadraticFit(np.array(tempWealth),np.array(contiValueInBundle),2)
            Jhat = localApprox_2.coef_[0][0] + localApprox_2.coef_[0][1]* tempWealth + \
                                    localApprox_2.coef_[0][2]*tempWealth**2
            # step 3
            # t
            temptWealthOld = np.multiply(wealthProcess[bundle,i],\
                            (np.multiply(intiCtrls[bundle,i], (returns[bundle,i] - r) + r))) + C*dt 
            ## regression
            localApprox_3 = quadraticFit(np.array(temptWealthOld),np.array(contiValueInBundle),2)
            Jtilda = localApprox_3.coef_[0][0] + localApprox_3.coef_[0][1]* temptWealthOld + \
                                    localApprox_3.coef_[0][2]*temptWealthOld**2
  
            #contiValuetList.append(contiValuet)
            indL = np.greater(Jtilda, Jhat)
            intiCtrls[bundle,i][indL] = xhat[indL]
            updatedCtrl = intiCtrls[bundle,i].copy()
            
            tempt = np.multiply(wealthProcess[bundle,i],\
                        (np.multiply(updatedCtrl,(returns[bundle,i] - r)) + r)) + C*dt
            ## regression

            localApprox_4 = quadraticFit(np.array(tempt),np.array(contiValueInBundle),2)
            J = localApprox_4.coef_[0][0] + localApprox_4.coef_[0][1]* tempt + \
                                    localApprox_4.coef_[0][2]*tempt**2

            contiValuetList[bundle] = J
            # print(max(J)) - J is indeed dreasing
        nextContiValuetList = contiValuetList
        optPs[:,i] = list(itertools.chain.from_iterable(optPsList))
    return nextContiValuetList, wealthProcess, optPs

# %%
stdFwdListAve =[]
muFwdListAve = []
stdFwdListAve_bdd =[]
muFwdListAve_bdd = []
stdFwdListAve_nobkcy =[]
muFwdListAve_nobkcy = []
for i in range(20):
    stdFwdList =[]
    muFwdList = []
    stdFwdList_bdd =[]
    muFwdList_bdd = []
    stdFwdList_nobkcy =[]
    muFwdList_nobkcy = []
    returns,increments = simulateReturnPathsGBM(r, sigma, xi,dt, numOfPath,timesteps,sid)
    gammaList = np.linspace(9.125,85.125, 80)
    for gamma in gammaList:
        termWealth, termWealth_nobkcy,termWealth_bdd,optPs,optP_bdds,optP_nobkcys = multiStageStrategyVectorize(gamma,returns,increments,\
                                                                    S0,r,xi,sigma,C,numOfPath,M,T,dt,W0,Pmax,Pmin,alpha,timesteps)
        stdFwdList.append(np.std(termWealth))
        muFwdList.append(np.mean(termWealth))
        stdFwdList_nobkcy.append(np.std(termWealth_nobkcy))
        muFwdList_nobkcy.append(np.mean(termWealth_nobkcy))
        stdFwdList_bdd.append(np.std(termWealth_bdd))
        muFwdList_bdd.append(np.mean(termWealth_bdd))
    stdFwdListAve.append(stdFwdList)
    muFwdListAve.append(muFwdList)
    stdFwdListAve_bdd.append(stdFwdList_bdd)
    muFwdListAve_bdd.append(muFwdList_bdd)
    stdFwdListAve_nobkcy.append(stdFwdList_nobkcy)
    muFwdListAve_nobkcy.append(muFwdList_nobkcy)

# %%
# multi-stage alone
# in order to get a smoother effient frontier, I use 20 random seeds and take
# average of std and expectation
# stdFwdListFin = np.asarray(stdFwdListAve).mean(axis=1)
# muFwdListFin = np.asarray(muFwdListAve).mean(axis=1)
# stdFwdListFin_bdd =np.asarray(stdFwdListAve_bdd).mean(axis=1)
# muFwdListFin_bdd = np.asarray(muFwdListAve_bdd).mean(axis=1)
# stdFwdListFin_nobkcy =np.asarray(stdFwdListAve_nobkcy).mean(axis=1)
# muFwdListFin_nobkcy = np.asarray(muFwdListAve_nobkcy).mean(axis=1)

stdFwdListFin = stdFwdListAve[2]
muFwdListFin = muFwdListAve[2]
stdFwdListFin_bdd =stdFwdListAve_bdd[2]
muFwdListFin_bdd = muFwdListAve_bdd[2]
stdFwdListFin_nobkcy = stdFwdListAve_nobkcy[2]
muFwdListFin_nobkcy = muFwdListAve_nobkcy[2]

fig4,ax = plt.subplots()
line1, = ax.plot(stdFwdListFin, muFwdListFin, label='Allow Bankruptcy')
line2, = ax.plot(stdFwdListFin_nobkcy, muFwdListFin_nobkcy, label='No Bankruptcy')
line3, = ax.plot(stdFwdListFin_bdd, muFwdListFin_bdd, label='Bounded Control')
ax.set_xlim([0, 10])
#ax.legend(loc='upper left')
ax.set_title('Efficient Frontier for Multi-Stage Strategy')
ax.set_xlabel('std[W(t=T)] at t = 0')
ax.set_ylabel('E[W(t=T)] at t = 0')
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

plt.show()


# %%
plt.savefig('nov20.png')



# %%
# efficient frontier for backward recursion approach.
# use simulation based approach rather than PDE
gammaList = np.linspace(9.125,85.125, 80)
#gammaList = [0.1]
stdList = []
meanList = []
stdListMulti = []
meanListMulti = []
x = []
r = 0.03
sigma = 0.15
xi = 0.33
C = 0.1
T = 2
W0 = 1
M = 80
numOfPath = 50000
numOfBundles = 20
dt = T/M
# one key point is that we should use the same brownian motion, this guarantee the smoothness of the frontier.
ret, incre = simulateReturnPathsGBM(r, sigma, xi, dt, numOfPath, timesteps,sid)
for gamma in gammaList:
    contiValuetList, wealthProcess, optimalCtrls= backwardStrategyTest(gamma,'nobkcy',True,ret,incre, sid)
    terminalW = simulatePathsUseCtrls(r, W0, C,incre, ret,optimalCtrls,dt)
    stdList.append(np.std(terminalW))
    meanList.append(np.mean(terminalW))
    stdListMulti.append(np.std(wealthProcess[:,-1]))
    meanListMulti.append(np.mean(wealthProcess[:,-1]))
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
plt.plot(stdList,meanList)
plt.title('Efficient frontier for 1 backward iteration')
plt.xlabel(r'$std_0[W^*_T]$')
plt.ylabel(r'$E_0[W^*_T]$')
#%%

fig9,ax = plt.subplots()
line1, = ax.plot(stdListMulti, meanListMulti, label='No Bankruptcy',color ='b')
line2, = ax.plot(stdList, meanList, label='1 Backward Recursion',color='r')

ax.set_xlim([0, 2])
ax.legend(loc='upper left')
ax.set_title('Efficient frontier comparison for No Bankruptcy Strategy \n and 1 Backward Recursion Strategy')
plt.xlabel(r'$std_0[W^*_T]$')
plt.ylabel(r'$E_0[W^*_T]$')
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)

plt.show()



# %%
# plot multi and backward frontier
gammaList = np.linspace(9.125,85.125, 20)
#gammaList = [0.1]
stdList1Back = []
meanList1Back = []
x = []
r = 0.03
sigma = 0.15
xi = 0.33
C = 0.1
T = 2
W0 = 1
M = 80
numOfPath = 50000
numOfBundles = 20
dt = T/M
# one key point is that we should use the same brownian motion, this guarantee the smoothness of the frontier.
ret, incre = simulateReturnPathsGBM(r, sigma, xi, dt, numOfPath, timesteps)
for gamma in gammaList:
    contiValuetList, wealthProcess, optimalCtrls= backwardStrategyTest(gamma,'ms',True,ret,incre)
    terminalW = simulatePathsUseCtrls(r, W0, C,incre, ret,optimalCtrls,dt)
    stdList1Back.append(np.std(terminalW))
    stdList1Back.append(np.mean(terminalW))
    termWealth, termWealth_nobkcy,termWealth_bdd,optPs,optP_bdds,optP_nobkcys = multiStageStrategyVectorize(gamma,ret,\
                                            incre,S0,r,xi,sigma,C,numOfPath,M,T,dt,W0,Pmax,Pmin,alpha,timesteps)
    stdFwdList.append(np.std(termWealth))
    muFwdList.append(np.mean(termWealth))
    stdFwdList_nobkcy.append(np.std(termWealth_nobkcy))
    muFwdList_nobkcy.append(np.mean(termWealth_nobkcy))
    stdFwdList_bdd.append(np.std(termWealth_bdd))
    muFwdList_bdd.append(np.mean(termWealth_bdd))

fig7,ax = plt.subplots()
line1, = ax.plot(stdFwdList, muFwdList, label='Allow Bankruptcy')
line2, = ax.plot(stdFwdList_nobkcy, muFwdList_nobkcy, label='No Bankruptcy')
line3, = ax.plot(stdFwdList_bdd, muFwdList_bdd, label='Bounded Control')
line4, = ax.plot(stdFwdList_bdd, muFwdList_bdd, label='1 Backward Iteration')
ax.set_xlim([0.1, 0.5])
#ax.legend(loc='upper left')
ax.set_title('Efficient Frontier comparision between Multi-Stage and 1 Backward Iteration')
ax.set_xlabel(r'$std_0[W^*_T]$')
ax.set_ylabel(r'$E_0[W^*_T]$')
#plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0.)
ax.legend(loc='upper left')
plt.show()
# %%



# %%






# %%


