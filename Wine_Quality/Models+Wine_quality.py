
# coding: utf-8

# In[156]:

import numpy as np
import pandas as pd

from bokeh.plotting import figure,show,output_notebook
from bokeh.models import Range1d

from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

output_notebook()
get_ipython().magic(u'matplotlib inline')
import statsmodels.api as sm


# In[157]:

xa = np.array([0,0.5,1,1.5,2,2.5])
ya = np.array([0,20.5,31.36,36.25,30.41,28.23])


# In[158]:

pa = figure()
pa.scatter(xa,ya)
show(pa) #visulaizing distribution of data to determine best fit


# In[159]:

Xa = np.c_[xa**2, xa, np.ones(len(xa))]
resa = sm.OLS(ya, Xa).fit()

pa = figure()
pa.circle(xa, ya, size=8,color='blue')

xxa = np.linspace(0,4,100)
pa.line(xxa, resa.predict(np.vander(xxa,3)), color='red')
show(pa) #hits 0 at 3.3 sec


# In[160]:

Xa


# In[191]:

xb = np.array([1976,1980,1987,1993,1998])
yb = np.array([618,860,1324,1865,2256])
pb = figure()
pb.scatter(xb,yb)
show(pb)


# In[194]:

Xb = np.c_[xb**2, xb, np.ones(len(xb))]
resb = sm.OLS(yb, Xb).fit()

pb = figure()
pb.circle(xb, yb, size=8,color='blue')

xxb = np.linspace(0,2500,100)
pb.line(xxb, resb.predict(np.vander(xxb,3)), color='red')
resb.params


# In[4]:

xc = np.array([-1,0,1,2,3,5,7,9])
yc = np.array([-1,3,2.5,5,4,2,5,4])
pc = figure()
pc.scatter(xc,yc)
show(pc)


# In[187]:

Xc = np.c_[xc**4,xc**3,xc**2,xc,np.ones(len(xc))]
resc = sm.OLS(yc, Xc).fit()
pc = figure()
pc.circle(xc,yc,size=8,color='Blue')

xxc = np.linspace(-5,10,20)
pc.line(xxc, resc.predict(np.vander(xxc,5)), color='red')
show(pc)


# In[188]:

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha = .02)
ridge.fit(np.vander(xc,6),yc)

lasso = Lasso(alpha = .01)
lasso.fit(np.vander(xc,6),yc)


# In[189]:

xxc2=np.linspace(-5,10,10)
pc2 = figure()
pc2.circle(xc,yc,size = 8, color='Blue')
pc2.line(xxc2, ridge.predict(np.vander(xxc2,6)), color='green') #overfitted?
pc2.line(xxc2, lasso.predict(np.vander(xxc2,6)), color='red') #better model 
show(pc2)


# In[179]:

np.vander(xxc2,3)


# In[178]:

xxc2


# In[70]:

winedf = pd.DataFrame.from_csv("winequality-red.csv",sep=';',index_col=None)


# In[72]:

winedf.head(10)


# In[162]:


from pandas.tools.plotting import scatter_matrix
scat = scatter_matrix(winedf, figsize = (20,20))
#relationshp between fixed acidity and density, citric acid and fixed acidity, anything w/ acid is correlated w/ pH


# In[ ]:

#quality score is somewhat negatively correlated with volatile acidity, fixed acidity and residual sugar follow almost a normalized distrubtion relative to quality score

#12 features, not normalized


# In[96]:

target = winedf['quality']
features = winedf.drop('quality',1)


# In[99]:

from sklearn.preprocessing import StandardScaler


# In[101]:


scalar = StandardScaler()
featuresNORM = scalar.fit_transform(features)


# In[103]:

featuresStandard = pd.DataFrame(featuresNORM, columns = features.columns)


# In[104]:

featuresStandard.head(3)


# In[146]:

x = featuresStandard.values
y = target.values
x[0]


# In[151]:

X = sm.add_constant(x, prepend=True)
wineGLM = sm.GLM(y,X).fit()
wineGLM.summary() #features with coefficients with low std err signify accuracy


# In[170]:

#important features are x2, x5, x7, x9, x10, x11

importantfeatures = featuresStandard.drop(['fixed acidity','citric acid','residual sugar','free sulfur dioxide','density'],1)


# In[ ]:

#could not figure out regularization for this problem...

