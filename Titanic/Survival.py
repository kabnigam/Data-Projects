
# coding: utf-8

# In[31]:

import numpy as np
import pandas as pd

from bokeh.plotting import figure,show,output_notebook
from bokeh.models import Range1d

from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split 

output_notebook()
get_ipython().magic(u'matplotlib inline')
import statsmodels.api as sm


# In[32]:

df = pd.read_csv('titanic-train.csv')


# In[33]:

df.head()


# In[34]:

import collections
print 'Embarked', collections.Counter(df.Embarked.values)
print 'Pclass', collections.Counter(df.Pclass.values)
print 'Survived',collections.Counter(df.Survived.values)
print 'Sex',collections.Counter(df.Sex.values)
print 'Age',collections.Counter(df.Age.values)
print 'SibSp',collections.Counter(df.SibSp.values)
print 'Parch',collections.Counter(df.Parch.values)
print 'Ticket',collections.Counter(df.Ticket.values)
print 'Fare',collections.Counter(df.Fare.values)
print 'Embarked',collections.Counter(df.Embarked.values)


# In[35]:

df['Embarked']=df['Embarked'].fillna('S')


# In[36]:

df['Fare'] = df.Fare.fillna(df.Fare.median())


# In[37]:

df['Age'].plot(kind='kde') #normal distibution, thus can fill with median


# In[38]:


df['Age'] = df['Age'].fillna(df.Age.median())


# In[39]:

df['Fare'].plot(kind = 'kde')


# In[40]:

df['Age'].plot(kind='kde')


# In[41]:

for i in range(len(df.Sex)):
    if df.Sex[i] == 'male':
        df.Sex[i] = 0
    else:
        df.Sex[i] = 1


# In[42]:

for i in range(len(df.Embarked)):
    if df.Embarked[i] == 'S':
        df.Embarked[i] = 0
    elif df.Embarked[i] == 'C':
        df.Embarked[i] = 1
    else:
        df.Embarked[i] = 2


# In[43]:

df.Embarked.values


# In[44]:

df.Fare.values


# In[45]:

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
target = ['Survived']


# In[46]:

data = df[features]
outcome = df[target].values.ravel()
X_train, X_test, Y_train, Y_test = train_test_split(data, outcome, test_size=0.2, random_state=1)


# In[47]:

from __future__ import division
c = [.01,.05,.07,.09,.1,.11,.12,.15,.17,.2,.3,.5,1,5,10,200]
score = []
for i in range(1,100):
    model_lr = LogisticRegression(C=(i/100)).fit(X_train, Y_train)
    score.append(cross_val_score(model_lr, data, outcome, cv=3).mean())
p = figure(title = 'CV Score for Various Regularization Parameters')
p.scatter(range(1,100),score)
show(p) #c = .08 yields the higest result


# In[48]:

score = []
for i in range(1,100):
    model_lr = LogisticRegression(C=(i)).fit(X_train, Y_train)
    score.append(cross_val_score(model_lr, data, outcome, cv=3).mean())
p3 = figure(title = 'CV Score for Various Regularization Parameters')
p3.scatter(range(1,100),score)
show(p3) #c = .08 yields the higest result


# In[49]:

model_lr = LogisticRegression(C=.08).fit(X_train, Y_train)
cross_val_score(model_lr, data, outcome, cv=3).mean()


# In[50]:

scores = []
for i in range(2,100):
    scores.append(cross_val_score(model_lr, data, outcome, cv=i).mean())
p2=figure(title='CV Score vs. Folds')
p2.scatter(range(2,100),scores)
show(p2) #c=28 yields the best result


# In[ ]:




# In[59]:

cross_val_score(model_lr, data, outcome, cv=28)


# In[52]:

model_lr.coef_


# In[53]:

predictions = model_lr.predict(X_train)
predictions


# In[54]:

model_lr.densify()


# 

# In[55]:

p = figure(title='Logistic Regression Coefficients')
for val in range(len(features)):
    p.quad(top=model_lr.coef_.ravel()[val], 
           bottom=0, left=val+0.2,right=val+0.8, 
           color=['red','orange','green','purple','blue','yellow','pink'][val],
           legend=features[val]
          )
    
p.y_range = Range1d(min(model_lr.coef_.ravel())-0.1, max(model_lr.coef_.ravel())+1.5)
show(p)


# In[56]:

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def plot_roc_curve(target_test, target_predicted_proba):
    fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:, 1])
    
    roc_auc = auc(fpr, tpr)
    
    p = figure(title='Receiver Operating Characteristic')
    # Plot ROC curve
    p.line(x=fpr,y=tpr,legend='ROC curve (area = %0.3f)' % roc_auc)
    p.x_range=Range1d(0,1)
    p.y_range=Range1d(0,1)
    p.xaxis.axis_label='False Positive Rate or (1 - Specifity)'
    p.yaxis.axis_label='True Positive Rate or (Sensitivity)'
    p.legend.orientation = "bottom_right"
    show(p)
    


# In[57]:

target_predicted_proba = model_lr.predict_proba(X_test)
plot_roc_curve(Y_test, target_predicted_proba)
roc_curve


# In[62]:

roc_curve.func_defaults


# In[58]:


rel = []

for i in range(len(df.Name)):

    if df.Parch[i] !=0 or df.SibSp[i] != 0:
        lastName = df.Name[i].split(',')
        if lastName[0] in df.Name[i]:
            rel.append(i)

cabin = []
for i in range(len(rel)):
    cabin.append(df.Cabin[rel[i]])   
    
#attempt to fill null cabin values by sibling/spouse/parent


# In[ ]:



