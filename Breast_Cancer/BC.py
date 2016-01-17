
# coding: utf-8

# In[242]:

import pandas as pd
import numpy as np
from unbalanced_dataset import UnderSampler, OverSampler, SMOTE

from bokeh.plotting import figure,show,output_notebook
from bokeh.models import Range1d
output_notebook()

import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[449]:

df = pd.read_csv('breast-cancer-wisconsin.data', names = ['Code Number','Clump Thickness','Cell Size','Cell Shape','Marginal Adhesion','Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'])


# In[450]:

df[df.values == '?']


# In[451]:

df['Bare Nuclei'] = df['Bare Nuclei'].replace('?','1')


# In[452]:

df['Bare Nuclei'] = df['Bare Nuclei'].astype(int)


# In[453]:

class_map = {2:0,4:1}
df['Class'] = df['Class'].apply(lambda x: class_map[x])



# In[454]:

from collections import Counter
Counter(df['Class'])


# In[457]:

from sklearn.cross_validation import train_test_split 
from sklearn.preprocessing import StandardScaler

X = df.ix[:,1:10].values
y = df['Class']



# In[467]:


xn = StandardScaler().fit_transform(X)
Xn = pd.DataFrame(xn,columns=df.ix[:,1:10].columns)


OS = SMOTE(ratio=.85, verbose=True)

osx, osy = OS.fit_transform(Xn.values, y)


# In[468]:

X_train, X_test, y_train, y_test = train_test_split(osx, osy, test_size=0.2, random_state=1)


# In[469]:

from sklearn.svm import SVC


# In[470]:

modelOS = SVC(kernel = 'linear',C=1).fit(osx,osy)


# In[471]:

from sklearn.metrics import classification_report


# In[472]:

print classification_report(y_test,modelOS.predict(X_test))


# In[473]:

print confusion_matrix(y_test,modelOS.predict(X_test))


# In[474]:

from sklearn.cross_validation import cross_val_score


# In[475]:

scoresOS = cross_val_score(model, osx, osy, cv = 5)
scoresOS.mean()


# In[338]:

preds = modelOS.predict(X_test)


# In[500]:

modelOS.probability = True


fpr, tpr, thresholds = roc_curve(y_test, modelOS.predict_proba(X_test)[:, 0])
    
roc_auc = auc(fpr, tpr)

print "AUC =",roc_auc


# In[510]:

modelOS.predict_proba(X_test)[:, 0]


# In[506]:

train_sizes,train_scores,test_scores = learning_curve(SVC(kernel='linear',C=1,probability=True),
                                                      osx,
                                                      osy,
                                                      train_sizes=np.linspace(0.05, 1.0, 20))

p = figure(title='Learning Curve',y_range=(0,1))

p.line(x=train_sizes,y=train_scores.mean(axis=1),color='red',legend="Training Scores")

p.line(x=train_sizes,y=test_scores.mean(axis=1),color='blue',legend = "Test Scores")

p.legend.orientation = "bottom_left"

show(p)


# In[482]:

from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(max_depth = 5, random_state=1)
treereg.fit(osx,osy)


# In[484]:

cross_val_score(treereg, X, y, cv=3)


# In[489]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(osx,osy)


# In[490]:

cross_val_score(rf, X, y, cv=3)


# In[498]:

print confusion_matrix(y_test, rf.predict(X_test))


# In[502]:

fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr, tpr)
print "Random Forest =",roc_auc_rf


# In[513]:

from sklearn.ensemble import VotingClassifier
clf1 = SVC(kernel = 'linear',C=1, probability = True)
clf2 = RandomForestClassifier()
eclf = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2)], voting = 'hard')
eclf.fit(X,y)


# In[ ]:



