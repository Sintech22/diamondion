# Quantum Generative Adversarial Neural Networks for universal information transmission
# Based on the paper "Quantum Relativity" and its source materials
# Note that the datasets referred to in the code are private. This file consists of a series of 
# tests preceding the simulation of a QGAN in classical computing. 
# In[1]:
from IPython.display import display, Math
import numpy as np  
# loading the training dataset
import pandas as pd      

def LoadData(filename):
    df = pd.read_csv(filename)
    X = df.values
    X, m, n = LoadData("Data.csv")
    df_test_12313414cadscf = pd.read_csv("Data.csv").values
    print(data.shape)
    print(df.values)   
    print(" X = \n",X)
    print('sample size m=' ,X.shape)
    
    print('feature length n=' ,Y.shape)
    
    return X, m, n

# The data is very likely nonlinear, so one may ask, why would one even bother testing for 
# linearity? For purely statistical reasons. 
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(x,y)

hypothesis_space = []
for i in range(0, 10):
    reg.coef_ = np.array([[i*0.05]])
    n = reg.predict(x)
    hypothesis_space.append(n)

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    

axes.scatter(x, y, color='blue',label="data points")
axes.plot(x, hypothesis_space[0], color='green',label='w = {:.2f}'.format(0*0.05))

for i in range(len(hypothesis_space)-1):
    y_n = hypothesis_space[i+1]
    l = 'w = {:.2f}'.format((i+1)*0.05)
    axes.plot(x, y_n, label=l)

plt.rc('legend', fontsize=10) 
plt.rc('axes', labelsize=20) 
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rc('font', size=20)

axes.set_title('linear predictors')
axes.set_xlabel('X')
axes.set_ylabel('y')
axes.legend(loc='upper left')
plt.show()
# 
# This is a quest for predictor functions.
# ## Probabilistic Models: Gaussian distribution  
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

n_samples = 300

np.random.seed(0)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
# fit a Gaussian Mixture Model with two components
clf = GaussianMixture(n_components=2, covariance_type='full', random_state=1)
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
# The code snippet below fits a two-dimensional Gaussian distribution to a set of data points which is read in from wikidata. 
# In[9]:
from scipy.stats import multivariate_normal
import pandas as pd  
from matplotlib import pyplot as plt 
from IPython.display import display, HTML
import numpy as np   
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math

reg = LinearRegression(fit_intercept=False) 
reg = reg.fit(X, y)
training_error = mean_squared_error(y, reg.predict(X))

display(Math(r'$\mathbf{w}_{\rm opt} ='))
optimal_weight = reg.coef_
optimal_weight = optimal_weight.reshape(-1,1)
print(optimal_weight)
print("\nThe resuling training error is ",training_error)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math
import matplotlib.pyplot as plt
import seaborn as sns
# import the local private dataset before feature extraction
m = 10
max_r = 10
def GetFeaturesLabels(m,n):
   
    dataset = load_csv()
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names) 
    
    error = np.zeros(10)
reg = LinearRegression(fit_intercept=False) 
    
for r in range(max_r):
    reg = reg.fit(X[:,:(r+1)], y)
    pred = reg.predict(X[:,:(r+1)])
    error[r] = mean_squared_error(y, pred)
    
    
       
    
    
print("Mean squared error: %.2f" % mean_squared_error(y_true, y_pred))
print('R²: %.2f' % r2_score(y_true, y_pred))

    
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_true, y_pred))
print('MSE:', metrics.mean_squared_error(y_true, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

plot_x = np.linspace(1, X, y, endpoint=True)
fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(12, 6))
axes[0].plot(plot_x, linreg_error, label='MSE', color='red')
axes[1].plot(plot_x, linreg_time, label='time', color='green')
axes[0].set_xlabel('features')
axes[0].set_ylabel('empirical error')
axes[1].set_xlabel('features')
axes[1].set_ylabel('time (ms)')
axes[0].set_title('training error vs number of features')
axes[1].set_title('computation time vs number of features')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()



import time

max_m = 10                            # maximum number of data points 
      # read in max_m data points using n=2 features 
train_error = np.zeros(max_m)         # vector for storing the training error of LinearRegresion.fit() for each r
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math
import matplotlib.pyplot as plt
import seaborn as sns

max_m = 10                            # maximum number of data points 
X, y = GetFeaturesLabels(max_m, 2)      # read in max_m data points using n=2 features 

train_error = np.zeros(max_m)
    
for r in range(max_m):
    reg = LinearRegression(fit_intercept=False) 
    reg = reg.fit(X[:(r+1),:], y[:(r+1)])
    y_pred = t_reg.predict(X[:(r+1),:])
    train_error[r] = mean_squared_error(y[:(r+1)], y_pred)
        
    
print(train_error[2])
plot_x = np.linspace(1, max_m, max_m, endpoint=True)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
axes.plot(plot_x, train_error, label='MSE', color='red')
axes.set_xlabel('number of data points (sample size)')
axes.set_ylabel('training error')
axes.set_title('training error vs. number of data points')
axes.legend()
plt.tight_layout()
plt.show()


# In[76]:

X,y = GetFeaturesLabels(10,1)   # read in 10 data points with single feature x_1 and label y 
### fit a linear model to the clean data 
reg = linear_model.LinearRegression(fit_intercept=False)
reg = reg.fit(X, y)
y_pred = reg.predict(X)

# now we intentionally perturb the label of the first data point 

y_perturbed = np.copy(y)  
y_perturbed[0] = 1000; 

### fit a linear model to the perturbed data 

reg1 = linear_model.LinearRegression(fit_intercept=False)
reg1 = reg1.fit(X, y_perturbed)
y_pred_perturbed = reg1.predict(X)


fig, axes = plt.subplots(1, 2, figsize=(15, 4))
axes[0].scatter(X, y, label='data')
axes[0].plot(X, y_pred, color='green', label='Fitted model')


# now add individual line for each error point
axes[0].plot((X[0], X[0]), (y[0], y_pred[0]), color='red', label='errors') # add label to legend
for i in range(len(X)-1):
    lineXdata = (X[i+1], X[i+1]) # same X
    lineYdata = (y[i+1], y_pred[i+1]) # different Y
    axes[0].plot(lineXdata, lineYdata, color='red')


axes[0].set_title('fitted model using clean data')
axes[0].set_xlabel('feature x_1')
axes[0].set_ylabel('y')
axes[0].legend()

axes[1].scatter(X, y_perturbed, label='data')
axes[1].plot(X, y_pred_perturbed, color='green', label='Fitted model')


# now add individual line for each error point
axes[1].plot((X[0], X[0]), (y_perturbed[0], y_pred_perturbed[0]), color='red', label='errors') # add label to legend
for i in range(len(X)-1):
    lineXdata = (X[i+1], X[i+1]) # same X
    lineYdata = (y_perturbed[i+1], y_pred_perturbed[i+1]) # different Y
    axes[1].plot(lineXdata, lineYdata, color='red')


axes[1].set_title('fitted model using perturbed training data')
axes[1].set_xlabel('feature x_1')
axes[1].set_ylabel('price y')
axes[1].legend()

plt.show()
plt.close('all') 

print("clean training data : ", reg.coef_)
print("perturbed training data : ", reg1.coef_)


import numpy as np
from matplotlib import pyplot as plt
def Phi(t, c):
    t = abs(t)
    flag = (t > c)
    return (~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t)
fig = plt.figure(figsize=(10, 3.75))
ax = fig.add_subplot(111)

x = np.linspace(-5, 5, 100)

for c in (1,2,10):
    y = Phi(x, c)
    ax.plot(x, y, '-k')

    if c > 10:
        s = r'\infty'
    else:
        s = str(c)

    ax.text(x[6], y[6], '$c=%s$' % s,
            ha='center', va='center',
            bbox=dict(boxstyle='round', ec='k', fc='w'))

ax.plot(x,np.square(x),label="squared loss")

ax.set_xlabel(r'$y - \hat{y}$')
ax.set_ylabel(r'loss $\mathcal{L}(y,\hat{y})$')
ax.legend()
plt.show()


from sklearn import linear_model
from sklearn.linear_model import HuberRegressor

reg = HuberRegressor().fit(X, y)
y_pred = reg.predict(X)

# now we intentionaly perturb the label of the first data point 

y_perturbed = np.copy(y)  
y_perturbed[0] = 1000; 

### fit a linear model (using Huber loss) to the perturbed data 

#reg1 = linear_model.LinearRegression(fit_intercept=False)
reg1 = HuberRegressor().fit(X, y_perturbed)
y_pred_perturbed = reg1.predict(X)


fig, axes = plt.subplots(1, 2, figsize=(15, 4))
axes[0].scatter(X, y, label='data')
axes[0].plot(X, y_pred, color='green', label='Fitted model')


# now add individual line for each error point
axes[0].plot((X[0], X[0]), (y[0], y_pred[0]), color='red', label='errors') # add label to legend
for i in range(len(X)-1):
    lineXdata = (X[i+1], X[i+1]) # same X
    lineYdata = (y[i+1], y_pred[i+1]) # different Y
    axes[0].plot(lineXdata, lineYdata, color='red')


axes[0].set_title('fitted model using clean data')
axes[0].set_xlabel('feature x_1')
axes[0].set_ylabel('y')
axes[0].legend()

axes[1].scatter(X, y_perturbed, label='data')
axes[1].plot(X, y_pred_perturbed, color='green', label='Fitted model')


# now add individual line for each error point
axes[1].plot((X[0], X[0]), (y_perturbed[0], y_pred_perturbed[0]), color='red', label='errors') # add label to legend
for i in range(len(X)-1):
    lineXdata = (X[i+1], X[i+1]) # same X
    lineYdata = (y_perturbed[i+1], y_pred_perturbed[i+1]) # different Y
    axes[1].plot(lineXdata, lineYdata, color='red')


axes[1].set_title('fitted model using perturbed data')
axes[1].set_xlabel('feature x_1')
axes[1].set_ylabel('y')
axes[1].legend()

plt.show()
plt.close('all') # clean up after using pyplot

print("clean data : ", reg.coef_)
print("perturbed data : ", reg1.coef_)

import time

m = 10                             
max_r = 10                         

X,y = GetFeaturesLabels(m,max_r)   

linreg_time = np.zeros(max_r)     # vector for storing the exec. times of LinearRegresion.fit() for each r
linreg_error = np.zeros(max_r)    # vector for storing the training error of LinearRegresion.fit() for each r


for r in range(max_r):
    reg_hub = HuberRegressor(fit_intercept=False) 
    start_time = time.time()
    reg_hub = reg_hub.fit(X[:,:(r+1)], y)
    end_time = (time.time() - start_time)*1000
    linreg_time[r] = end_time
    pred = reg_hub.predict(X[:,:(r+1)])
    linreg_error[r] = mean_squared_error(y, pred)

plot_x = np.linspace(1, max_r, max_r, endpoint=True)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
axes[0].plot(plot_x, linreg_error, label='MSE', color='red')
axes[1].plot(plot_x, linreg_time, label='time', color='green')
axes[0].set_xlabel('features')
axes[0].set_ylabel('empirical error')
axes[1].set_xlabel('features')
axes[1].set_ylabel('Time (ms)')
axes[0].set_title('training error vs number of features')
axes[1].set_title('computation time vs number of features')
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from unittest.mock  import patch

def sigmoid_func(x):
    f_x = 1/(1+np.exp(-x))
    return f_x

fig, axes = plt.subplots(1, 1, figsize=(15, 5)) #used only for testing purpose

range_x = np.arange(-5 , 5 , 0.01).reshape(-1,1)
print(range_x.shape)
logloss_y1 = np.empty(len(range_x))
logloss_y0 = np.empty(len(range_x))
#squaredloss_y1 = np.empty(len(range_x))
#squaredloss_y0 = np.empty(len(range_x))
plt.rc('legend', fontsize=20) 
plt.rc('axes', labelsize=20) 
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

for i in range(len(range_x)):
    logloss_y1[i] = -np.log(sigmoid_func(range_x[i]))     # logistic loss when true label y=1
    logloss_y0[i] = -np.log(1-sigmoid_func(range_x[i]))   # logistic loss when true label y=0
     


axes.plot(range_x,logloss_y1, linestyle=':', label=r'$y=1$',linewidth=5.0)
axes.plot(range_x,logloss_y0, label=r'$y=0$',linewidth=5.0)

axes.set_xlabel(r'$\mathbf{w}^{T}\mathbf{x}$')
axes.set_ylabel(r'$\mathcal{L}((y,\mathbf{x});\mathbf{w})$')
axes.set_title("logistic loss",fontsize=20)
axes.legend()

def sigmoid_func(x):
    f_x = 1/(1+np.exp(-x))
    return f_x

fig, axes = plt.subplots(1, 1, figsize=(15, 5))

range_x = np.arange(-2 , 2 , 0.01).reshape(-1,1)
print(range_x.shape)
logloss_y1 = np.empty(len(range_x))
logloss_y0 = np.empty(len(range_x))
squaredloss_y1 = np.empty(len(range_x))
squaredloss_y0 = np.empty(len(range_x))

plt.rc('legend', fontsize=20) 
plt.rc('axes', labelsize=40) 
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30) 

for i in range(len(range_x)):
    logloss_y1[i] = -np.log(sigmoid_func(range_x[i]))     # logistic loss when true label y=1
    logloss_y0[i] = -np.log(1-sigmoid_func(range_x[i]))   # logistic loss when true label y=0
    squaredloss_y1[i] = ((1-range_x[i]) ** 2)
    squaredloss_y0[i] = ((0-range_x[i]) ** 2)


axes.plot(range_x,logloss_y1, linestyle=':', label=r'logistic loss $y=1$',linewidth=5.0)
axes.plot(range_x,logloss_y0, label=r'logistic loss $y=0$',linewidth=5.0)
axes.plot(range_x,squaredloss_y0/2, label=r'squared error for $y=0$',linewidth=5.0)
axes.plot(range_x,squaredloss_y1/2, label=r'squared error for $y=1$',linewidth=5.0)

axes.set_xlabel(r'$\mathbf{w}^{T}\mathbf{x}$')
axes.set_ylabel(r'$\mathcal{L}((y,\mathbf{x});\mathbf{w})$')
axes.legend()


print('First entry of squaredloss_y0:', squaredloss_y0[0])
print('First entry of squaredloss_y1:', squaredloss_y1[0])
print('Last entry of squaredloss_y0:', squaredloss_y0[-1])
print('Last entry of squaredloss_y1:', squaredloss_y1[-1])

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = LoadData()         
X = data['data']                    
cat = data['target'].reshape(-1, 1) 

m = cat.shape[0]         # set m equal to the number of rows in features  
y = np.zeros((m, 1));    # initialize label vector with zero entries
    
for i in range(m):
        if (cat[i] == 0):
            y[i,:] = 1 # Class 0
        else:
            y[i,:] = 0 #Not class 0


print(X.shape, y.shape)
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(random_state=0,C=1e6)
logReg.fit(X, y)
y_pred = logReg.predict(X).reshape(-1, 1)
print(y_pred.shape)
 

from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm

data = LoadData()         
X = data['data'][:, :2]            
c = data['target']                  

m = cat.shape[0]         # set m equal to the number of rows in features  
y = np.zeros((m,1));    # initialize label vector with zero entries
    
for i in range(m):
        if (c[i] == 0):
            y[i] = 1 # Class 0
        else:
            y[i] = 0 #Not class 0


# Create an instance of Logistic Regression Classifier and fit the data.
logreg = LogisticRegression(C=1e6,random_state=0)
logreg.fit(X, y)
weight = logreg.coef_ 
weight = weight.reshape(-1,1)
y_pred = logreg.predict(X)
accuracy = metrics.accuracy_score(y, y_pred)

print("Accuracy:", round(100*accuracy, 2), '%')

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5


h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:,0]

Z = Z.reshape(xx.shape)
x1 = xx[:,np.argmax(Z==1)][0]
x2 = xx[:,np.argmax(Z[-1]==1)][0]

xgrid = np.linspace(x_min, x_max, 100)
y_boundary = (-1/weight[1])*(xgrid*weight[0] + logreg.intercept_)



def calculate_accuracy(y, y_hat):
  
    
    from sklearn.metrics import accuracy_score
    data = LoadData()#load the private dataset
    X = data['data']
    cat = wine['target'].reshape(-1,1)
    m = cat.shape[0]
    y = np.zeros((m, 1))

    for i in range(m):
        if (cat[i] == 0):
            y[i,:] = 1
        else:
            y[i,:] = 0

    y_hat = logReg.predict(X).reshape(-1, 1)        
    
    accuracy = metrics.accuracy_score(y, y_pred) * 100
    return accuracy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

data = LoadData()         
X = data['data']                   
cat = data['target'].reshape(-1, 1) 

m = cat.shape[0]           
y = np.zeros((m, 1));    
    
for i in range(m):
        if (cat[i] == 0):
            y[i,:] = 1 # Class 0
        else:
            y[i,:] = 0 #Not class 0

logReg = LogisticRegression(random_state=0)

logReg = logReg.fit(X, y)

y_pred = logReg.predict(X).reshape(-1, 1)
            
# Tests
test_acc = calculate_accuracy(y, y_pred)
print ('Accuracy of the result is: %f%%' % test_acc)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logReg = LogisticRegression(random_state=0,multi_class="ovr") 
# set multi_class to one versus rest ('ovr')

logReg = logReg.fit(X, y)

y_pred = logReg.predict(X).reshape(-1, 1)

# This function is used to plot the confusion matrix and normalized confusion matrix
import itertools
from sklearn.metrics import confusion_matrix
def visualize_cm(cm):
    """
    Function visualizes a confusion matrix with and without normalization
    """
    plt.rc('legend', fontsize=10) 
    plt.rc('axes', labelsize=10) 
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 


    fig, axes = plt.subplots(1, 2,figsize=(10,5))

    im1 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im1, ax=axes[0])
    classes = ['Class 0','Class 1','Class 2']
    tick_marks = np.arange(len(classes))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(classes,rotation=45)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    axes[0].set_xlabel('predicted label $\hat{y}$')
    axes[0].set_ylabel('true label $y$')
    axes[0].set_title(r'$\bf{Figure\ 6.}$Without normalization')
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im2, ax=axes[1])
    
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(classes,rotation=45)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[1].text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    axes[1].set_xlabel('predicted label $\hat{y}$')
    axes[1].set_ylabel('true label $y$')
    axes[1].set_title(r'$\bf{Figure\ 7.}$Normalized')
    
    axes[0].set_ylim(-0.5,2.5) 
    axes[1].set_ylim(-0.5,2.5)
    
    plt.tight_layout()
    plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred)
visualize_cm(cm)


y_probs = logReg.predict_proba(X)
# show the inputs and predicted probabilities
print('first five samples and their probabilities of belonging to classes 0, 1 and 2:')
for i in range(5):
    print("Probabilities of Sample", i+1,':', 'Class 0:',"{:.2f}".format(100*y_probs[i][0],2),'%', 'Class 1:', "{:.2f}".format(100*y_probs[i][1]), '%', 'Class 2:', "{:.2f}".format(100*y_probs[i][2]),'%' )

n_of_discarded_samples = 0

from sklearn.metrics import log_loss
yhat = [x*0.01 for x in range(178, 3)] 
n_of_discarded_samples_0 = [log_loss([0], [i], labels=[0,1]) for i in yhat]
n_of_discarded_samples_1 = [log_loss([1], [i], labels=[0,1]) for i in yhat]
n_of_discarded_samples_2 = [log_loss([2], [i], labels=[0,1]) for i in yhat]
if (y_probs[i] > n+1):
        y[i] = 1 
else:
        y[i] = 0 
          
print('Number of discarded samples:', n_of_discarded_samples)



import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics 

def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    indx_1 = np.where(Y == 1)[0] # index of each class 
    indx_2 = np.where(Y == 0)[0] # index of each not class 
    

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[indx_1, 0], X[indx_1, 1],marker='x',label='class 0', edgecolors='k')
    plt.scatter(X[indx_2, 0], X[indx_2, 1],marker='o',label='class 1', edgecolors='k')
    plt.xlabel(r'Feature 1')
    plt.ylabel(r'Feature 2')

    
#load dataset
X = ['data'][:, :2]             # matrix containing the feature vectors 
c = ['target']                  # vector contaiing the true categories 

m = cat.shape[0]         # set m equal to the number of rows in features  
y = np.zeros((m,1));    # initialize label vector with zero entries
    
for i in range(m):
        if (c[i] == 0):
            y[i] = 1 # Class 0
        else:
            y[i] = 0 #Not class 0


tree = DecisionTreeClassifier()   # define object 
tree.fit(X, y)                    # learn a decision tree   
y_pred = tree.predict(X)          # compute the predicted labels 
accuracy = metrics.accuracy_score(y, y_pred)  
print("Accuracy:", round(100*accuracy, 2), '%')

plot_decision_boundary(tree, X, y)
plt.show()


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import confusion_matrix 
from IPython.core.getipython import get_ipython
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import metrics

m = 20 
n = 10 

X = np.random.randn(m,n)   # create feature vectors using random numbers
y = np.random.randn(m,1)   # create labels using random numbers 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2) # 80% training and 20% test

plt.rc('legend', fontsize=20) 
plt.rc('axes', labelsize=20) 
fig1, axes1 = plt.subplots(figsize=(15, 5))
axes1.scatter(X[:, 0], X[:, 1], c='g', s=200,marker ='x', label='original dataset')
axes1.legend(loc='best')
axes1.set_xlabel('feature x1')
axes1.set_ylabel('feature x2')

fig2, axes2 = plt.subplots(figsize=(15, 5))
axes2.scatter(X_train[:, 0], X_train[:, 1], c='g', s=200,marker ='o', label='training set')
axes2.scatter(X_val[:, 0], X_val[:, 1], c='brown', s=200,marker ='s', label='validation set')
axes2.legend(loc='best')
axes2.set_xlabel('feature x1')
axes2.set_ylabel('feature x2')
fig2.show()
# Model validation and selection 
scaler = StandardScaler().fit(X)
X = scaler.transform(X)          # normalize feature values to standard value range 
scaler = StandardScaler().fit(y)
y = scaler.transform(y)
    
        
return X, y



m = 20                         
n = 10                        

X,y = GetFeaturesLabels(m,n)  # read in m data points using n features 
linreg_error = np.zeros(n)    # vector for storing the training error of LinearRegresion.fit() for each r

for r_minus_1 in range(n):  # loop over number of features r (minus 1)
    reg = LinearRegression(fit_intercept=False)   # create an object for linear predictors
    reg = reg.fit(X[:,:(r_minus_1 + 1)], y)                 # find best linear predictor (minimize training error)
    pred = reg.predict(X[:,:(r_minus_1 + 1)])               # compute predictions of best predictors 
    linreg_error[r_minus_1] = mean_squared_error(y, pred) # compute training error 

plot_x = np.linspace(1, n, n, endpoint=True)      # plot_x contains grid points for x-axis

# Plot training error E(r) as a function of feature number r
plt.rc('legend', fontsize=12)
plt.plot(plot_x, linreg_error, label='$E(r)$', color='red')
plt.xlabel('# of features $r$')
plt.ylabel('training error $E(r)$')
plt.title('training error vs number of features')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split 

m = 20                         
n = 10                         

X,y = GetFeaturesLabels(m,n)   

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

X,y = GetFeaturesLabels(m,n)  # read in m data points using n features 
linreg_error = np.zeros(n)    # vector for storing the training error of LinearRegresion.fit() for each r

for r_minus_1 in range(n):  # loop over number of features r (minus 1)
    reg = LinearRegression(fit_intercept=False)   # create an object for linear predictors
    reg = reg.fit(X[:,:(r_minus_1 + 1)], y)                 # find best linear predictor (minimize training error)
    pred = reg.predict(X[:,:(r_minus_1 + 1)])               # compute predictions of best predictors 
    linreg_error[r_minus_1] = mean_squared_error(y, pred) # compute training error 


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)
err_train = np.zeros([n,1])
err_val = np.zeros([n,1])

if(err_train[0] < 0.33):
    intercept = True
else:
    intercept = False

for r in range(n):
    lin_reg = LinearRegression(fit_intercept=intercept)
    lin_reg = lin_reg.fit(X_train[:,:(r+1)], y_train)
    y_pred_train = lin_reg.predict(X_train[:,:(r+1)])
    err_train[r] = mean_squared_error(y_train, y_pred_train)
    y_pred_val = lin_reg.predict(X_val[:,:(r+1)])
    err_val[r] = mean_squared_error(y_val, y_pred_val)

best_model = np.argmin(err_val)+1

# Plot the training and validation errors for the different number of features r

plt.plot(range(1, n + 1), err_train, color='black', label=r'$E_{\rm train}(r)$', marker='o')
plt.plot(range(1, n + 1), err_val, color='red', label=r'$E_{\rm val}(r)$', marker='x')

plt.title('Training and validation error for different number of features')
plt.ylabel('Empirical error')
plt.xlabel('r features')
plt.xticks(range(1, n + 1))
plt.legend()
plt.show()

print(err_val[:4])
# ## K-fold Cross-Validation

from sklearn.model_selection import KFold

K=5    # number of folds/rounds/splits
kf = KFold(n_splits=K, shuffle=False)
kf = kf.split(X)
kf = list(kf)                 # kf is a list representing the rounds of k-fold CV

m = 20                        # we use the first m=20 data points from the house sales database 
n = 10                        # maximum number of features used 

X,y = GetFeaturesLabels(m,n)  # read in m data points with n features 
r=2    # we use only first two features for linear predictors h(x) = w^{T}x



train_errors_per_cv_iteration = []
test_errors_per_cv_iteration = []  

# for loop over K rounds 
        
for train_indices, test_indices in kf:
        
    reg = LinearRegression(fit_intercept=False)
    reg = reg.fit(X[train_indices,:(r+1)], y[train_indices])
    y_pred_train = reg.predict(X[train_indices,:(r+1)])
    train_errors_per_cv_iteration.append(mean_squared_error(y[train_indices], y_pred_train))
    y_pred_val = reg.predict(X[test_indices,:(r+1)])
    test_errors_per_cv_iteration.append(mean_squared_error(y[test_indices], y_pred_val))
            

err_train= np.mean(train_errors_per_cv_iteration) # compute the mean of round-wise training errors
err_val = np.mean(test_errors_per_cv_iteration)   # compute the mean of round-wise validation errors
        
print("Training error (averaged over 5 folds): ",err_train)
print("Validation error (averaged over 5 folds):", err_val)


# ##  Regularization
# 
# In what follows, we stick to the hypothesis space $\mathcal{H}^{(n)}$ of linear predictors $h(\mathbf{x}) = \mathbf{w}^{T} \mathbf{x}$ using the maximum number $n$ of features. To find a good predictor function we could try to mininize the training error $E_{\rm train}$ over some training set $\mathbb{X}^{(t)}$. However, as discussed above, for large $n$ (i.e., if we have collected a large amount of features for a house) we will easily find a predictor function $h(\mathbf{x})=\mathbf{w}^{T} \mathbf{x}$ such that $E_{\rm train}$ is small but the predictor function will do poorly on data points different from $\mathbb{X}^{(t)}$. 
# 
# The idea of regularization is to somehow estimate (or approximate) the increase of the prediction error in new data which is different from the training set. This estimated (or anticipated) increase is represented by adding a **regularization term** to the training error: 
# \begin{equation}
# h^{(\lambda)}_{\rm opt}  = {\rm argmin}_{h \in \mathcal{H}} \underbrace{(1/m_{t}) \sum_{\big(\mathbf{x}^{(i)},y^{(i)}\big) \in \mathbb{X}^{(t)}} \big(y^{(i)} - h(\mathbf{x}^{(i)}) \big)^{2}}_{\mbox{ training error}} + \underbrace{\alpha \mathcal{R}(h)}_{\mbox{anticipated increase of error (loss) on new data}}.  
# \end{equation}
# 
# 
# The regularization term $\mathcal{R}(h)$ quantifies the anticipated increase in the validation error (compared to the training error) due to the "complexity" (e.g. the number of features used in a linear predictor) of a particular predictor. In a nutshell, the regularization term penalizes the use of more complex predictors and therefore favors "simpler" predictor functions. The precise meaning of "complexity" or "simpler" is determined by the (design) choice for the regularization term $\mathcal{R}(h)$. 
# 
# Two widely used choices for measuring the complexity of linear predictors $h(\mathbf{x}) = \mathbf{w}^{T}\mathbf{x}$ is the squared Euclidean norm $\mathcal{R}(h) = \|\mathbf{w}\|^{2}_{2}=\sum_{r=1}^{n} w_{r}^{2}$ or the $\ell_{1}$ norm $\mathcal{R}(h) = \|\mathbf{w}\|_{1}=\sum_{r=1}^{n} |w_{r}|$. 
# 
# The regularization parameter $\alpha$ offers a trade off between the prediction error (training error) incurred on the training data and the complexity of a predictor. The larger we choose $\alpha$, the more emphasis is put on obtaining "simple" predictor functions. Using very small values for $\alpha$ prefers predictor functions which achieve a small training error (at the expense of being a more complicated function). 
# 
# Ridge regression is obtained by linear regression, using linear predictor functions $h^{(\mathbf{w})}(\mathbf{x}) =\mathbf{w}^{T} \mathbf{x}$, with the regularization term $\mathcal{R}(h)=\|\mathbf{w}\|_{2}^{2}$. The optimal weight vector $\mathbf{w}_{\rm opt}$ can be found using the Python function `Ridge.fit()`. 

from sklearn.linear_model import Ridge
alpha_scaled = 2*n 

ridge = Ridge(alpha=alpha_scaled, fit_intercept=True)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_train)
w_opt = ridge.coef_
err_train = mean_squared_error(y_pred, y_train)

# Print optimal weights and training error
print('Optimal weights: \n', w_opt)
print('Training error: \n', err_train)
from sklearn.linear_model import Lasso


X,y = GetFeaturesLabels(m,n)  # read in m data points using n features 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2) # 80% training and 20% test
alpha_val = 1


if(w_opt[0] < 0.09):
    intercept = True
else:
    intercept = False

def fitLasso(x_train, y_train, lambd=0):
    lasso = Lasso(alpha=lambd/2, fit_intercept=intercept)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_train)
    w_opt = lasso.coef_
    training_error = mean_squared_error(y_pred, y_train)
    return w_opt, training_error

w_opt, training_error = fitLasso(X_train, y_train, lambd=1)

# Print optimal weights and the corresponding training error
print('Optimal weights: \n', w_opt)
print('Training error: \n', training_error)

# Compute the Lasso estimator $h^{(\alpha)}$ for each value $\alpha$ from the list $\alpha^{(1)},\ldots,\alpha^{(9)}$ defined in the code snippet below (numpy array `alpha_values`). Store the resulting validation errors $E_{\rm val}(\alpha^{(i)})$ and training errors $E_{\rm train}(\alpha^{(i)})$ in the numpy array `err_val` of shape (9,1) and `err_train` of shape (9,1). The first entry `err_val[0]` should be $E_{\rm val}(\alpha^{(1)})$, and so on. For the optimal choice $\alpha^{(i)}$ (yielding smallest validation error), store the weight vector of the optimal predictor in the numpy array `w_opt` of shape (n,1). 
# 
# Specify a list of values for lambda to be considered
alpha_values = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 1, 3, 10, 10e3])
nr_values = len(alpha_values)

intercept = False
if w_opt[0] < 0.57:
    intercept = True

err_val = np.zeros([nr_values,1])
err_train = np.zeros([nr_values,1])
for l in range(nr_values):
    lasso = Lasso(alpha=alpha_values[l]/2, fit_intercept=intercept)
    lasso = lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    err_train[l] = mean_squared_error(y_train_pred, y_train)
    y_val_pred = lasso.predict(X_val)
    err_val[l]=mean_squared_error(y_val_pred, y_val)


best_alpha_idx=np.argmin(err_val)
lasso = Lasso(alpha=alpha_values[best_alpha_idx]/2, fit_intercept=intercept)
lasso = lasso.fit(X_train, y_train)
w_opt = lasso.coef_.reshape(-1,1)

# Plot the training and validation errors
plt.plot(alpha_values, err_train, marker='o', label='training error')
plt.plot(alpha_values, err_val, marker='o', label='validation error')
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$E(\alpha)$')
plt.legend()
plt.show()

# Clustering

#read in data from the csv file and store it in the numpy array data.
df = pd.read_csv("data.csv")
data = np.array(df)

#display first 5 rows
display(df.head(5))  
def plotting(data, centroids=None, clusters=None):
    
    plt.figure(figsize=(5.75,5.25))
    data_colors = ['orangered','dodgerblue','springgreen']
    plt.style.use('ggplot')
    plt.title("Data")
    plt.xlabel("feature $x_1$: customers' age")
    plt.ylabel("feature $x_2$: money spent during visit")

    alp = 0.5             # data points alpha
    dt_sz = 20            # marker size for data points 
    cent_sz = 130         # centroid sz 
    
    if centroids is None and clusters is None:
        plt.scatter(data[:,0], data[:,1],s=dt_sz,alpha=alp ,c=data_colors[0])
    if centroids is not None and clusters is None:
        plt.scatter(data[:,0], data[:,1],s=dt_sz,alpha=alp, c=data_colors[0])
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=cent_sz, c=centroid_colors[:len(centroids)])
    if centroids is not None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i-1] for i in clusters], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", c=centroid_colors[:len(centroids)], s=cent_sz)
    if centroids is None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i-1] for i in clusters], s=dt_sz, alpha=alp)
    
    plt.show()

#plot the data
plotting(data)   
#  The k-Means Algorithm
from sklearn.cluster import KMeans

X = np.zeros([400,2])               
cluster_means = np.zeros([2,2])     # store the resulting clustering means in the rows of this np array
cluster_indices = np.zeros([400,1]) # store here the resulting cluster indices (one for each data point)
data_colors = ['orangered','dodgerblue','springgreen'] # colors for data points
centroid_colors = ['red','darkblue','limegreen'] # colors for the centroids

X = data      
k_means = KMeans(n_clusters = 3, max_iter = 100).fit(X) # apply k-means with k=3 cluster and using 100 iterations
cluster_means = k_means.cluster_centers_         # read out cluster means (centers)
cluster_indices = k_means.labels_                # read out cluster indices for each data point
cluster_indices = cluster_indices.reshape(-1,1)  # enforce numpy array cluster_indices having shape (400,1)

# code below creates a colored scatterplot 

plt.figure(figsize=(5.75,5.25))
plt.style.use('ggplot')
plt.title("Data")
plt.xlabel("feature $x_1$: ")
plt.ylabel("feature $x_2$: ")

alp = 0.5             # data points alpha
dt_sz = 40            # marker size for data points 
cent_sz = 130         # centroid sz 
       
# iterate over all cluster indices (minus 1 since Python starts indexing with 0)
for cluster_index in range(3):
    # find indices of data points which are assigned to cluster with index (cluster_index+1)
    indx_1 = np.where(cluster_indices == cluster_index)[0] 

    # scatter plot of all data points in cluster with index (cluster_index+1)
    plt.scatter(X[indx_1,0], X[indx_1,1], c=data_colors[cluster_index], s=dt_sz, alpha=alp) 
    
# plot crosses at the locations of cluster means 

plt.scatter(cluster_means[:,0], cluster_means[:,1], marker="x", c='black', s=cent_sz)
    
plt.show()

from sklearn.cluster import KMeans

X = np.zeros([400,2])               
cluster_means = np.zeros([2,2])     # store the resulting clustering means in the rows of this np array
cluster_indices = np.zeros([400,1]) # store here the resulting cluster indices (one for each data point)

data_colors = ['orangered','dodgerblue','springgreen'] 
centroid_colors = ['red','darkblue','limegreen'] 
X = data
k_means = KMeans(n_clusters = 2, max_iter = 10).fit(X)
cluster_means = k_means.cluster_centers_
cluster_indices = k_means.labels_
cluster_indices = cluster_indices.reshape(-1,1)
print(type(cluster_indices))
plotting(X,cluster_means,k_means.labels_)


#print(type(cluster_indices))
#plotting(X,cluster_means,cluster_indices)
print("The final cluster mean values are:\n",cluster_means)

min_ind= 0   
max_ind= 0  


cluster_assignment = np.zeros((50, data.shape[0]),dtype=np.int32)

clustering_err = np.zeros([50,1]) # init numpy array for storing the clustering errors in each repetition

np.random.seed(42)  


init_means_cluster1 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means 
init_means_cluster2 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means 
init_means_cluster3 = np.random.randn(50,2)  # use the rows of this numpy array to init k-means 

best_assignment = np.zeros((400,1))     # store here the cluster assignment achieving smallest clustering error
worst_assignment = np.zeros((400,1))    # store here the cluster assignment achieving largest clustering error

cluster_assignment = np.zeros((50, data.shape[0]),dtype=np.int32)
clustering_err = np.zeros(50)

for i in range(50):
    init_means = np.vstack((init_means_cluster1[i,:],init_means_cluster2[i,:],init_means_cluster3[i,:]))             
    k_means= KMeans(n_clusters = 3, n_init = 1, max_iter = 10, init = init_means).fit(data)
    clustering_err[i] = k_means.inertia_
    cluster_assignment[i,:] = k_means.labels_

    

min_ind = np.argmin(clustering_err)
max_ind = np.argmax(clustering_err)
best_assignment = cluster_assignment[min_ind]
worst_assignment = cluster_assignment[max_ind] 

print("Cluster assignment with smallest clustering error")
plotting(data, clusters = cluster_assignment[min_ind, :])
print("Cluster assignment with largest clustering error")
plotting(data, clusters = cluster_assignment[max_ind,:])

data_num = data.shape[0]
err_clustering = np.zeros([8,1])

for i in range(8):
    k_means = KMeans(n_clusters = i+1).fit(data)
    err_clustering[i] = k_means.inertia_/data_num


fig=plt.figure(figsize=(8,6))
plt.plot(range(1,9),err_clustering)
plt.xlabel('Number of clusters')
plt.ylabel('Clustering error')
plt.title("The number of clusters vs clustering error")
plt.show()    

data_num = data.shape[0]
err_clustering = np.zeros([8,1])

for i in range(8):
    k_means = KMeans(n_clusters = i+1).fit(data)
    err_clustering[i] = k_means.inertia_/data_num


fig=plt.figure(figsize=(8,6))
plt.plot(range(1,9),err_clustering)
plt.xlabel('Number of clusters')
plt.ylabel('Clustering error')
plt.title("The number of clusters vs clustering error")
plt.show()    

data_num = data.shape[0]
err_clustering = np.zeros([8,1])

for i in range(8):
    k_means = KMeans(n_clusters = i+1).fit(data)
    err_clustering[i] = k_means.inertia_/data_num


fig=plt.figure(figsize=(8,6))
plt.plot(range(1,9),err_clustering)
plt.xlabel('Number of clusters')
plt.ylabel('Clustering error')
plt.title("The number of clusters vs clustering error")
plt.show()    

def plot_GMM(data,means,covariances,k,cluster_vectors=None):
    
    
    data_colors = ['orangered','dodgerblue','springgreen'] # colors for data points
    centroid_colors = ['red','darkblue','limegreen'] # colors for the centroids
    
    if cluster_vectors is None:
        plt.scatter(data[:,0], data[:,1], s=13,alpha=0.5)
    else:
        clusters = np.argmax(cluster_vectors,axis=0)
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i] for i in clusters], s=13,alpha=0.5)

    #Visualization of results
    x_plot = np.linspace(19,35, 100)
    y_plot = np.linspace(0,12, 100)

    for i in range(k):
        x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
        
        pos = np.empty(x_mesh.shape + (2,))
        pos[:,:,0] = x_mesh; pos[:,:,1] = y_mesh
        z = multivariate_normal(mean = [means[i,0],means[i,1]],                                cov = [[np.sqrt(covariances[i,0, 0]), covariances[i,0,1]],[covariances[i,0,1], np.sqrt(covariances[i,1, 1])]]).pdf(pos)

        
        plt.contour(x_mesh , y_mesh , z,4,colors=centroid_colors[i],alpha=0.5)
        plt.scatter( [means[i,0]], [means[i,1]], marker='x',c=centroid_colors[i])

    plt.title("Soft clustering with GMM")
    plt.xlabel("feature x_1: ")
    plt.ylabel("feature x_2: ")
    plt.show()



from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal # Multivariate normal random variable

def belonging_num(data_point, mean_point, covariance_matrix, coefficient_number):
    belonging_num = coefficient_number*multivariate_normal.pdf(data_point, mean_point, covariance_matrix)
    return belonging_num

def belonging_den(data_point, means, covariances, coefficients, k):
    belonging_all=np.sum([belonging_num(data_point, means[f], covariances[f], coefficients[f]) for f in range(k)])
    return belonging_all

def update_degrees_of_belonging(data, means, covariances,coefficients,k): 
    cluster_vectors=np.zeros((k,data.shape[0]))   
    for i in range(data.shape[0]):
        belonging_all = belonging_den(data[i], means, covariances, coefficients, k)
        for t in range(k):
            cluster_vectors[t,i]=belonging_num(data[i], means[t], covariances[t], coefficients[t])/belonging_all
    return cluster_vectors


k = 3
gmm = GaussianMixture(n_components = k, max_iter = 50).fit(data)


cluster_vectors = update_degrees_of_belonging(data, gmm.means_, gmm.covariances_, gmm.weights_, k)
plot_GMM(data, gmm.means_, gmm.covariances_, 3, cluster_vectors)
print("The means are:\n",gmm.means_)
print("The covariance matrices are:\n",gmm.covariances_)
# Density Based Clustering
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

np.random.seed(844)
clust1 = np.random.normal(5, 2, (1000,2))
clust2 = np.random.normal(15, 3, (1000,2))
clust3 = np.random.multivariate_normal([17,3], [[1,0],[0,1]], 1000)
clust4 = np.random.multivariate_normal([2,16], [[1,0],[0,1]], 1000)

dataset1 = np.concatenate((clust1, clust2, clust3, clust4))
dataset2 = datasets.make_circles(n_samples=1000, factor=.5, noise=.05)[0]

# plot clustering output on the two datasets
def cluster_plots(set1, set2, colours1, colours2, 
                  title1 = 'Dataset 1',  title2 = 'Dataset 2'):
    colours1 = colours1.reshape(-1,)
    colours2 = colours2.reshape(-1,)
    fig,(ax1,ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)
    ax1.set_title(title1,fontsize=14)
    ax1.set_xlim(min(set1[:,0]), max(set1[:,0]))
    ax1.set_ylim(min(set1[:,1]), max(set1[:,1]))
    ax1.scatter(set1[:, 0], set1[:, 1],s=8,lw=0,c=colours1)
    ax2.set_title(title2,fontsize=14)
    ax2.set_xlim(min(set2[:,0]), max(set2[:,0]))
    ax2.set_ylim(min(set2[:,1]), max(set2[:,1]))
    ax2.scatter(set2[:, 0], set2[:, 1],s=8,lw=0,c=colours2)
    fig.tight_layout()
    plt.show()

cluster_plots(dataset1, dataset2,np.ones([dataset1.shape[0],1]),np.ones([dataset2.shape[0],1]))
from sklearn import cluster

kmeans_dataset1 = cluster.KMeans(n_clusters=4, max_iter=300, init='k-means++',n_init=10).fit_predict(dataset1)
kmeans_dataset2 = cluster.KMeans(n_clusters=2, max_iter=300, init='k-means++',n_init=10).fit_predict(dataset2)
cluster_plots(dataset1, dataset2, kmeans_dataset1, kmeans_dataset2)
# DBSCAN 

from sklearn.cluster import DBSCAN

dataset1 = DBSCAN(eps=1, min_samples=5, metric='euclidean').fit(X) 
dataset2 = DBSCAN(eps=0.1, min_samples=5, metric='euclidean').fit(X)
dbscan_dataset1 = cluster.KMeans(n_clusters=4, max_iter=300, init='k-means++',n_init=10).fit_predict(dataset1)
dbscan_dataset2 = cluster.KMeans(n_clusters=2, max_iter=300, init='k-means++',n_init=10).fit_predict(dataset2)

dataset1_clusters = []
for i in range(4):
    dataset1_clusters.append(sum(dbscan_dataset1==i))
    

dataset2_clusters = []
for i in range(2):
    dataset2_clusters.append(sum(dbscan_dataset2==i))
        
print(dbscan_dataset1.shape)
print('Dataset1:')
print("Number of Noise Points: ",dataset1_noise_points," (",len(dbscan_dataset1),")",sep='')
print('Dataset2:')
print("Number of Noise Points: ",dataset2_noise_points," (",len(dbscan_dataset2),")",sep='')

cluster_plots(dataset1, dataset2, dbscan_dataset1, dbscan_dataset2)

from PIL import Image


m = 30 # number of images 
dataset = np.zeros((m,50,50),dtype=np.uint8)   # create numpy array for images and fill with zeros 
D = 50*50  # length of raw feature vectors 

for i in range(1,m+1):
    # with convert('L') we convert the images to grayscale
    try:
        img = Image.open('ImageNet.jpg'%(str(i))).convert('L') # read in image from jpg file
    except:
        img = Image.open('ImageNet.jpg'%(str(i))).convert('L') # read if you are doing exercise locally
    dataset[i-1] = np.array(img,dtype=np.uint8)             # convert image to numpy array with greyscale values
    Z = np.reshape(dataset,(m,-1))  
# reshape the 50 x 50 pixels into a long numpy array of shape (2500,1)
print("The shape of the datamatrix Z is", Z.shape) 


fig,ax = plt.subplots(3,2,figsize=(10,10),gridspec_kw = {'wspace':0, 'hspace':0})
for i in range(3):
    for j in range(2):
        bitmap = Z[i+(15*j),:] 
        bitmap = np.reshape(bitmap,(50,50))
      #  ax[i,j].imshow(dataset[i+(15*j)], cmap='gray')
        ax[i,j].imshow(bitmap, cmap='gray')
        ax[i,j].axis('off')
plt.show()
# Principal Component Analysis

from sklearn.decomposition import PCA

n = 10 #   nr of principal components

from numpy import linalg as LA
m = Z.shape[0]
Z_hat = np.zeros(Z.shape)
error = 0
pca = PCA(n_components=n).fit(Z)
W_pca = pca.components_
for i in range (m):
    X = np.dot(W_pca,Z[i].T)
    Z_hat[i] = np.dot(X, W_pca)
    error = error + LA.norm(Z[i] - Z_hat[i])**2


err_pca = error/m

print(Z_hat.shape)
print(W_pca.shape)
print(err_pca)


import numpy as np
from sklearn.decomposition import PCA



from numpy import linalg as LA

err_pca = np.zeros((m,1))

for n_minus_1 in range(m):

    
    pca = PCA(n_components=(n_minus_1+1)) 
    pca.fit(Z)
    W_pca = pca.components_
    X = np.dot(Z,W_pca.T)
    Z_hat = np.dot(X,W_pca)
    err_pca[n_minus_1] = np.sum(np.square(Z - Z_hat))/Z.shape[0]

# työkirja on korjattu 
   
    
# plot the number of PCs vs the reconstruction error
plt.plot([i + 1 for i in range(m)],err_pca)
plt.xlabel('Number of PCs ($n$)')
plt.ylabel(r'$\mathcal{E}(\mathbf{W}_{\rm PCA})$')
plt.title('Number of PCs vs the reconstruction error')
plt.show()   
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

## Input:
#  Z: Dataset
#  n: number of dimensions
#  m_pics: number of images per class
def plot_reconstruct(Z,n,m_pics=3):


    # x=w*z
    X_pca = np.matmul(W_pca[:n,:],Z[:,:,None])
    # x_reversed=r*x
    X_reversed = np.matmul(W_pca[:n,:].T, X_pca)[:,:,0]
    
    # Setup Figure size that scales with number of images
    fig = plt.figure(figsize = (10,10))
    
    # Setup a (n_pics,2) grid of plots (images)
    gs = gridspec.GridSpec(1, 2*m_pics)
    gs.update(wspace=0.0, hspace=0.0)
    for i in range(m_pics):
        for j in range(0,2):
            # Add a new subplot
            ax = plt.subplot(gs[0,i+j*m_pics])
            # Insert image data into the subplot
            ax.imshow(np.reshape(X_reversed[i+(15*j)],(50,50)),cmap='gray',interpolation='nearest')
            # Remove x- and y-axis from each plot
            ax.set_axis_off()
    
    plt.subplot(gs[0,0]).set_title("Reconstructed images using %d PCs:"%(n), size='large', y=1.08)
    # Render the plot
    plt.show()


    
pca = PCA(n_components = m) # create the object
pca.fit(Z)     # compute optimal transform W_PCA
W_pca = pca.components_
    

num_com=[1,5,10]
for n in num_com:
    print(n)
    plot_reconstruct(Z,n,m_pics=3)

def plot_princ_comp(W_pca):
    fig,ax = plt.subplots(1,5,figsize=(15,15))
    
    plot_pd = [0,4,9,14,19]

    for i in range(len(plot_pd)):
        ax[i].imshow(np.reshape(W_pca[plot_pd[i]]*255,(50,50)),cmap='gray')
        ax[i].set_title("Principal Direction %d"%(plot_pd[i]+1))
        
        ax[i].set_axis_off()
    plt.show()

plot_princ_comp(W_pca)

X_visual = np.zeros((m,2))


pca = PCA(n_components=2) 
pca.fit(Z)
W_pca = pca.components_
X_visual = np.dot(Z,W_pca.T)

print(Z_hat.shape)
print(W_pca.shape)
print(err_pca)

pca = PCA(n_components=m) 
pca.fit(Z)
W_pca = pca.components_
X = np.dot(Z,W_pca.T)


X_PC12 = X[:,[0,1]]
X_PC89 =X[:,[7,8]]

plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
 

plt.figure()
plt.title('using first two PCs $x_{1}$ and $x_{2}$ as features')
plt.scatter(X_PC12[:15,0],X_PC12[:15,1],c='r',marker='o',label='Apple')
plt.scatter(X_PC12[15:,0],X_PC12[15:,1],c='y',marker='^',label='Banana')
plt.legend()
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.show()
    
plt.figure()
plt.title('using 8th and 9th PC as features')
plt.scatter(X_PC89[:15,0],X_PC89[:15,1],c='r',marker='o',label='Apple')
plt.scatter(X_PC89[15:,0],X_PC89[15:,1],c='y',marker='^',label='Banana')
plt.legend()
plt.xlabel('$x_{8}$')
plt.ylabel('$x_{9}$')
plt.show()

# Dmensionality reduction with linear prediction
import pandas as pd  
from matplotlib import pyplot as plt 
from IPython.display import display, HTML
import numpy as np   
import random

def GetFeaturesLabels(m=10, D=10):
    dataset = LoadData()
    data = pd.DataFrame(dataset.data, columns=house_dataset.feature_names) 
    x1 = data['1'].values.reshape(-1,1)   # vector whose entries are the tax rates of the sold houses
    x2 = data['2'].values.reshape(-1,1)   # vector whose entries are the ages of the sold houses
    x1 = x1[0:m]
    x2 = x2[0:m]
    np.random.seed(43)
    Z = np.hstack((x1,x2,np.random.randn(m,n))) 
    
    Z = Z[:,0:D] 
    y = dataset.target.reshape(-1,1)  
    
    y = y[0:m]
    
    return Z, y



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from IPython.display import display, Math
import numpy as np
from sklearn.model_selection import train_test_split
n = 10
m = 500   # total number of data points 
D = 10    # length of raw feature vector 


Z,y = GetFeaturesLabels(m,D)   
        
## use this features for PCA 
Z_pca = Z[:480,:]    # read out feature vectors of first 480 data points 


## use this features and labels for linear regression (with transformed features)
Z     = Z[480:,:]    # read out feature vectors of last 20 data points 
y     = y[480:,:]    # read out labels of last 20 data points 

# Datasets which will be preprocessed and used with linear regression
Z_train, Z_val, y_train, y_val = train_test_split(Z, y, test_size=0.2, random_state=42)


err_val = np.zeros(D)     # this numpy array has to be used to store the validation 
                        #  errors of linear regression when combined with PCA with n=1,2,..,D
    
err_train = np.zeros(D)

for n in range(1,D+1,1):
    
 
    pca = PCA(n_components = n)
    pca.fit(Z_pca)
    X_train = pca.transform(Z_train)
    X_val   = pca.transform(Z_val)
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    err_train[n-1] = np.mean((y_train-y_pred_train)**2)  # compute training error 
    err_val[n - 1] = np.mean((y_val - y_pred_val)**2)   # compute validation error 
    


plt.plot([n for n in range(1,D+1,1)],err_val,label="validation")
plt.plot([n for n in range(1,D+1,1)],err_train,label="training")
plt.xlabel('number of PCs ($n$)')
plt.ylabel(r'error')
plt.legend()
plt.title('validation/training error vs. number of PCs')
plt.show()

# This is just the humble beginning of what is to be built into a QGAN based on GAN equations.
