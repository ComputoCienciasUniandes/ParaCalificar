# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 07:08:38 2018

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
import sklearn.model_selection

data = pd.read_csv('Cars93.csv')
X = data[['Horsepower', 'Turn.circle','MPG.highway']]
Y = data['Price'].values.reshape(-1,1)

X_train, X_validation, Y_train, Y_validation = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)

n_boot = 5000
beta_0 = np.ones(n_boot)
beta_1 = np.ones(n_boot)
beta_2 = np.ones(n_boot)
beta_3 = np.ones(n_boot)

r2_train = np.ones(n_boot)
r2_validation = np.ones(n_boot)

linear = sklearn.linear_model.LinearRegression()
for i in range(n_boot):
    X_train, X_validation, Y_train, Y_validation = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)
    linear.fit(X_train, Y_train)
    beta_0[i] = linear.intercept_[0]
    beta_1[i] = linear.coef_[0][0]
    beta_2[i] = linear.coef_[0][1]
    beta_3[i] = linear.coef_[0][2]

    r2_train[i] = linear.score(X_train, Y_train)
    r2_validation[i] = linear.score(X_validation, Y_validation) 
    
    
plt.figure(figsize=(20,4))

plt.subplot(141)
_ = plt.hist(beta_0, bins=40)
_ = plt.xlabel(r'$\beta_0$')

plt.subplot(142)
_ = plt.hist(beta_1, bins=40)
_ = plt.xlabel(r'$\beta_1$')

plt.subplot(143)
_ = plt.hist(beta_2, bins=40)
_ = plt.xlabel(r'$\beta_2$')


plt.subplot(144)
_ = plt.hist(beta_3, bins=40)
_ = plt.xlabel(r'$\beta_3$')


print('beta 0 {} +/- {}'.format(beta_0.mean(), beta_0.std() ))
print('beta 1 {} +/- {}'.format(beta_1.mean(), beta_1.std() ))
print('beta 2 {} +/- {}'.format(beta_2.mean(), beta_2.std() ))
print('beta 3 {} +/- {}'.format(beta_3.mean(), beta_3.std() ))














y_obs=data['Price']
sigma_y_obs=1

def model(beta0,beta1,beta2,beta3):
    return beta0+data['Horsepower']*beta1+data['Turn.circle']*beta2+data['MPG.highway']*beta3

def loglikelihood(y_obs, sigma_y_obs, beta0,beta1,beta2,beta3):
    d = y_obs -  model(beta0, beta1, beta2,beta3)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d

def logprior(beta0, beta1,beta2,beta3):
    p = -np.inf
    if beta0 < 10 and beta0 >-10 and beta1 >-10 and beta1<10 and beta2 >-10 and beta2<10 and beta3 >-10 and beta3<10:
        p = 0.0
    return p

N = 50000
lista_beta0 = [np.random.random()]
lista_beta1 = [np.random.random()]
lista_beta2 = [np.random.random()]
lista_beta3 = [np.random.random()]

logposterior = [loglikelihood( y_obs, sigma_y_obs, lista_beta0[0], lista_beta1[0],lista_beta2[0],lista_beta3[0]) + logprior(lista_beta0[0], lista_beta1[0],lista_beta2[0],lista_beta3[0])]

sigma_delta_beta0 = 0.2
sigma_delta_beta1 = 1.0
sigma_delta_beta2 = 1.0
sigma_delta_beta3 = 1.0

for i in range(1,N):
    propuesta_beta0  = lista_beta0[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_beta0)
    propuesta_beta1  = lista_beta1[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_beta1)
    propuesta_beta2  = lista_beta2[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_beta2)
    propuesta_beta3  = lista_beta3[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_beta3)
    
    logposterior_viejo = loglikelihood( y_obs, sigma_y_obs, lista_beta0[i-1], lista_beta1[i-1],lista_beta2[i-1],lista_beta3[i-1]) + logprior(lista_beta0[i-1], lista_beta1[i-1],lista_beta2[i-1],lista_beta3[i-1])
    logposterior_nuevo = loglikelihood( y_obs, sigma_y_obs, propuesta_beta0, propuesta_beta1,propuesta_beta2,propuesta_beta3) + logprior(propuesta_beta0, propuesta_beta1,propuesta_beta2,propuesta_beta3)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_beta0.append(propuesta_beta0)
        lista_beta1.append(propuesta_beta1)
        lista_beta2.append(propuesta_beta2)
        lista_beta3.append(propuesta_beta3)
        logposterior.append(logposterior_nuevo)
    else:
        lista_beta0.append(lista_beta0[i-1])
        lista_beta1.append(lista_beta1[i-1])
        lista_beta2.append(lista_beta2[i-1])
        lista_beta3.append(lista_beta3[i-1])
        
        logposterior.append(logposterior_viejo)
lista_beta0 = np.array(lista_beta0)
lista_beta1 = np.array(lista_beta1)
lista_beta2 = np.array(lista_beta2)
lista_beta3 = np.array(lista_beta3)
logposterior = np.array(logposterior)

plt.figure()
_=plt.hist(lista_beta1, bins=60)



k=['MPG.city', 'MPG.highway', 'EngineSize', 
   'Horsepower', 'RPM', 'Rev.per.mile',
   'Fuel.tank.capacity', 'Passengers', 'Length',
   'Wheelbase', 'Width', 'Turn.circle', 'Weight']

X2 = data[k]

n_boot = 5000
beta_0 = np.ones(n_boot)
beta_1 = np.ones(n_boot)
beta_2 = np.ones(n_boot)
beta_3 = np.ones(n_boot)
beta_4 = np.ones(n_boot)
beta_5 = np.ones(n_boot)
beta_6 = np.ones(n_boot)
beta_7 = np.ones(n_boot)
beta_8 = np.ones(n_boot)
beta_9 = np.ones(n_boot)
beta_10 = np.ones(n_boot)
beta_11 = np.ones(n_boot)
beta_12= np.ones(n_boot)
beta_13 = np.ones(n_boot)

r2_train = np.ones(n_boot)
r2_validation = np.ones(n_boot)

linear = sklearn.linear_model.LinearRegression()
for i in range(n_boot):
    X_train, X_validation, Y_train, Y_validation = sklearn.model_selection.train_test_split(X2, Y, test_size=0.5)
    linear.fit(X_train, Y_train)
    beta_0[i] = linear.intercept_[0]
    beta_1[i] = linear.coef_[0][0]
    beta_2[i] = linear.coef_[0][1]
    beta_3[i] = linear.coef_[0][2]
    beta_4[i] = linear.coef_[0][3]
    beta_5[i] = linear.coef_[0][4]
    beta_6[i] = linear.coef_[0][5]
    beta_7[i] = linear.coef_[0][6]
    beta_8[i] = linear.coef_[0][7]
    beta_9[i] = linear.coef_[0][8]
    beta_10[i] = linear.coef_[0][9]
    beta_11[i] = linear.coef_[0][10]
    beta_12[i] = linear.coef_[0][11]
    beta_13[i] = linear.coef_[0][12]


    r2_train[i] = linear.score(X_train, Y_train)
    r2_validation[i] = linear.score(X_validation, Y_validation) 


print('beta 0 {} +/- {}'.format(beta_0.mean(), beta_0.std() ))
print('beta 1 {} +/- {}'.format(beta_1.mean(), beta_1.std() ))
print('beta 2 {} +/- {}'.format(beta_2.mean(), beta_2.std() ))
print('beta 3 {} +/- {}'.format(beta_3.mean(), beta_3.std() ))


print('beta 4 {} +/- {}'.format(beta_4.mean(), beta_4.std() ))
print('beta 5 {} +/- {}'.format(beta_5.mean(), beta_5.std() ))
print('beta 6 {} +/- {}'.format(beta_6.mean(), beta_6.std() ))
print('beta 7 {} +/- {}'.format(beta_7.mean(), beta_7.std() ))


print('beta 8 {} +/- {}'.format(beta_8.mean(), beta_8.std() ))
print('beta 9 {} +/- {}'.format(beta_9.mean(), beta_9.std() ))
print('beta 10 {} +/- {}'.format(beta_10.mean(), beta_10.std() ))
print('beta 11 {} +/- {}'.format(beta_11.mean(), beta_11.std() ))


print('beta 12 {} +/- {}'.format(beta_12.mean(), beta_12.std() ))
print('beta 13 {} +/- {}'.format(beta_13.mean(), beta_13.std() ))


plt.figure(figsize=(20,4))

plt.subplot(141)
_ = plt.hist(beta_0, bins=40)
_ = plt.xlabel(r'$\beta_0$')

plt.subplot(142)
_ = plt.hist(beta_1, bins=40)
_ = plt.xlabel(r'$\beta_1$')

plt.subplot(143)
_ = plt.hist(beta_2, bins=40)
_ = plt.xlabel(r'$\beta_2$')


plt.subplot(144)
_ = plt.hist(beta_3, bins=40)
_ = plt.xlabel(r'$\beta_3$')


plt.figure(figsize=(20,4))

plt.subplot(141)
_ = plt.hist(beta_4, bins=40)
_ = plt.xlabel(r'$\beta_4$')

plt.subplot(142)
_ = plt.hist(beta_5, bins=40)
_ = plt.xlabel(r'$\beta_5$')

plt.subplot(143)
_ = plt.hist(beta_6, bins=40)
_ = plt.xlabel(r'$\beta_6$')


plt.subplot(144)
_ = plt.hist(beta_7, bins=40)
_ = plt.xlabel(r'$\beta_7$')


plt.figure(figsize=(20,4))

plt.subplot(141)
_ = plt.hist(beta_0, bins=40)
_ = plt.xlabel(r'$\beta_8$')

plt.subplot(142)
_ = plt.hist(beta_1, bins=40)
_ = plt.xlabel(r'$\beta_9$')

plt.subplot(143)
_ = plt.hist(beta_2, bins=40)
_ = plt.xlabel(r'$\beta_10$')


plt.subplot(144)
_ = plt.hist(beta_3, bins=40)
_ = plt.xlabel(r'$\beta_11$')


