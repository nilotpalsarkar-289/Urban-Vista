#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('accidents_india.csv')
df.head()


# In[3]:


df.isna()


# In[4]:


pd.unique(df['Accident_Severity'])


# In[5]:


df.dropna(inplace = True)


# In[6]:


df.columns[df.isna().any()]


# In[7]:


df.Sex_Of_Driver = df.Sex_Of_Driver.fillna(df.Sex_Of_Driver.mean())
df.Vehicle_Type = df.Vehicle_Type.fillna(df.Vehicle_Type.mean())
df.Speed_limit = df.Speed_limit.fillna(df.Speed_limit.mean())
df.Road_Type = df.Road_Type.fillna(df.Road_Type.mean())
df.Number_of_Pasengers = df.Number_of_Pasengers.fillna(df.Speed_limit.mean())


# In[8]:


corr = df.corr()
import seaborn as sns
sns.set()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="RdYlBu", annot=True, fmt=".1f")


# In[60]:


# df.replace(np.nan, '', regex=True)


# In[61]:


# df.fillna('', inplace=True)
# df.columns[df.isna().any()]


# In[9]:


#LabelEncoding
c = LabelEncoder()
df['Day'] = c.fit_transform(df['Day_of_Week'])
df.drop('Day_of_Week', axis=1, inplace=True)
l = LabelEncoder()
df['Light'] = l.fit_transform(df['Light_Conditions'])
df.drop('Light_Conditions', axis=1, inplace=True)
s = LabelEncoder()
df['Severity'] = s.fit_transform(df['Accident_Severity'])
df.drop('Accident_Severity', axis=1, inplace=True)
df.head() 


# In[10]:


from sklearn.model_selection import train_test_split
x = df.drop(['Pedestrian_Crossing', 'Special_Conditions_at_Site', 'Severity'], axis=1)
y = df['Severity']


# In[11]:


pd.unique(y)


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.86)
from sklearn.tree import DecisionTreeClassifier
reg = DecisionTreeClassifier(criterion='gini')
reg.fit(x_train, y_train)
reg.score(x_test, y_test)


# In[49]:


yp = reg.predict(x_test)
import seaborn as sn
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, yp)
cm


# In[52]:


import matplotlib.pyplot as plt
from pylab import savefig
#labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
#labels = np.asarray(labels).reshape(2,2)
#plt.figure(figsize =(11,8))
#sn.heatmap(cm, annot=labels, cmap="Greens")

group_names = ['True Neg','False Pos', 'False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')


# In[46]:


import numpy as np
import pickle
inputt=[int(x) for x in "2 10 201 10 10 8 3".split(' ')]
final=[np.array(inputt)]
b = reg.predict(final)
pickle.dump(reg,open('test1.pkl','wb'))
test=pickle.load(open('test1.pkl','rb'))


# ## Random Forest

# In[31]:


from sklearn.ensemble import RandomForestClassifier
r_forest = RandomForestClassifier(criterion='entropy')
r_forest.fit(x_train, y_train)


# In[32]:


r_forest.score(x_test, y_test)


# In[33]:


df.hist(bins=50, figsize=(20, 15))
plt.show()


# In[34]:


df["Severity"].value_counts().plot.bar(color='b', edgecolor='red',linewidth=1)


# # SVM

# In[77]:


from sklearn import svm
model = svm.SVC(C=50, kernel = 'linear')


# In[78]:


model.fit(x_train, y_train)


# In[79]:


model.score(x_test, y_test)


# In[42]:


from sklearn.linear_model import LogisticRegression
tru = LogisticRegression()
tru.fit(x_train, y_train)
tru.score(x_test, y_test)


# # Logistic Regression

# To Do
# 
# 1. Classificaiton Report
# 2. Confusion Matrix

# In[80]:



import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_adult_census(model):
#     n_features = X.shape[1]
    plt.barh(range(7),model.feature_importances_,align='center')
    plt.yticks(np.arange(7),x.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("feature")
    ##plt.show()
    ##fig=plt.figure()
#     plt.savefig("feature_imporatnace_diabetes.png")
    plt.show()
#     plt.close()
plot_feature_importances_adult_census(reg)


# In[81]:


from sklearn import tree
import matplotlib.pyplot as plt
tree.plot_tree(reg)
plt.show()


# In[85]:


import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


# In[88]:


import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
dot_data = StringIO()
export_graphviz(reg, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=x.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())
graph.write_png('graph.png')


# In[ ]:




