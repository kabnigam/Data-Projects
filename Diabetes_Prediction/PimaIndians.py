
# coding: utf-8

# In[183]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.neighbors import KNeighborsClassifier

import json
import urllib
import requests
from bokeh.plotting import figure,output_notebook,show,gridplot 
output_notebook()


# In[184]:

pimaURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"


# In[185]:

pima_response = requests.get(pimaURL)


# In[186]:

PIMAdf = pd.read_csv('pima-indians-diabetes.data')


# In[187]:

PIMAdf


# In[188]:

PIMAdf.columns = ['Pregnancies', 'Plasma Glucose','Diastolic','Tricep Fold','Insulin','BMI','DPF','Age','Class']


# In[189]:

PIMAdf


# In[205]:

PIMAdf.plot(kind="scatter",x=3,y=5,c='r',title="Base Visual")


# In[204]:

BMITri = PIMAdf['weight'].plot(kind='box')


# In[191]:

PIMAdf.plot(kind="scatter",x=1,y=4,c='r',title="Base Visual")


# In[192]:

PIMAdf['Tricep Fold'].plot(kind='box')


# In[181]:

PIMAdf['Tricep Fold'].plot(kind='KDE')


# In[180]:

PIMAdf['Insulin'].plot(kind='box')


# In[168]:

plots = []
for feat_x in PIMAdf.columns:
    for feat_y in PIMAdf.columns:
        
        temp_p = figure(plot_width=200, 
                        plot_height=200, 
                        x_axis_label=feat_x, 
                        y_axis_label=feat_y
                       )
        temp_p.circle(PIMAdf[feat_x], 
                      PIMAdf[feat_y], 
                      line_width=1, 
                      alpha=0.4,
                      size=5)
        
        temp_p.xaxis.axis_label_text_font_size = '9pt'
        temp_p.yaxis.axis_label_text_font_size = '9pt'

        plots.append(temp_p)

# gridplot takes nested lists of bokeh figures and arranges them on the grid in the positions given. 
# Passing None inserts a blank.

sqrt = len(plots)**0.5
gplots = np.array(plots).reshape(sqrt,sqrt)

# To convert to a square, we reshape the array into a grid with the # of rows equal to the # of columns. 

#REMEMBER: gridplot takes a list of lists, so we convert gplots with the .tolists() method
a = gridplot(gplots.tolist())
show(a)


# In[214]:

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly


# In[256]:

coefs = poly.polyfit(PIMAdf['BMI'], PIMAdf['Tricep Fold'],2)
reversed_co = coefs[::-1]
np.polyval(reversed_co, 26)





#plt.figure(figsize=(15,6.6))
#plt.subplot(1,2,1) 
#plt.plot(PIMAdf['BMI'],PIMAdf['Tricep Fold'], 'kx')
#plt.xlabel('x')
#plt.ylabel('y')
#y=PIMAdf['Tricep Fold']
#x=PIMAdf['BMI']
#fit=plt.polyfit(x,y,1)
##slope, fit_fn=pl.poly1d(fit)
#fit_fn=pl.poly1d(fit)
#scat=pl.plot(x,y, 'kx', x,fit_fn(x), '-b' )


# In[257]:

for i in xrange(len(PIMAdf['Tricep Fold'])):
        if PIMAdf['Tricep Fold'][i] == 0:
            PIMAdf['Tricep Fold'][i] = np.polyval(reversed_co, PIMAdf['BMI'][i])


# In[258]:

PIMAdf


# In[262]:

coefs2 = poly.polyfit(PIMAdf['Plasma Glucose'], PIMAdf['Insulin'],2)
reversed_co2 = coefs2[::-1]


# In[263]:

reversed_co2


# In[266]:

np.polyval(reversed_co2, 200)


# In[265]:

PIMAdf.plot(kind="scatter",x=1,y=4,c='r',title="Base Visual")


# In[270]:

for i in xrange(len(PIMAdf['Insulin'])):
        if PIMAdf['Insulin'][i] == 0:
            PIMAdf['Insulin'][i] = np.polyval(reversed_co2, PIMAdf['Plasma Glucose'][i])


# In[272]:

PIMAdf


# In[ ]:




# In[274]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

x = PIMAdf.ix[:,:-1]
y = PIMAdf.Class

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.20)

PIMAknn = KNeighborsClassifier(3).fit(x_train,y_train)

features=['Pregnancies','Plasma Glucose', 'Diastolic', 'Tricep Fold', 'Insulin', 'BMI','DPF','Age']

to_predict = "Class"

data = PIMAdf[features]
label = PIMAdf[to_predict]

n_neighbors = range(1,50,2)
scores = []


for n in n_neighbors:
   PIMAknn = KNeighborsClassifier(n).fit(x_train,y_train)
   score = PIMAknn.score(x_test, y_test)
   scores.append(score)




CVdf = pd.DataFrame(n_neighbors, columns=['n'])
CVdf['scores'] = scores



fig = plt.figure(figsize=(6,5))
plt.title('KNN Score as a function of number of Neighbors \n (For a single train test split)')
plt.ylim(0.2,1.1)
plt.plot(CVdf.n, CVdf.scores)




# In[275]:

cross_val_score(PIMAknn, data, label, cv=5)


# In[277]:

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nb = MultinomialNB()

nb.fit(x_train, y_train)


# In[ ]:



