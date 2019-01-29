#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import seaborn as sns
from sklearn.datasets import load_boston


# In[2]:


df = load_boston()
print(df.data.shape)


# In[3]:


boston = pd.DataFrame(df.data, columns=df.feature_names)
boston.head(10)


# In[4]:


boston['MEDV'] = df.target
boston.describe()


# In[5]:


boston.info()


# In[6]:


boston.hist(bins=100, figsize=(70,60))
plt.show()


# In[7]:


correlation_matrix = boston.corr().round(2)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data= correlation_matrix, annot=True)


# In[8]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(boston, test_size=0.2, random_state=42)
print(len(train_set), "train ,", len(test_set), "test")


# In[9]:


boston1 = train_set.copy()
from sklearn.linear_model import LinearRegression
X = boston1[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 
            'B', 'LSTAT']]
Y = boston1['MEDV']
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


# In[10]:


from sklearn.metrics import mean_squared_error
prediction = lin_reg.predict(X)
lin_mse = mean_squared_error(Y, prediction)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[11]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X,Y)


# In[12]:


prediction1 = tree_reg.predict(X)
tree_mse = mean_squared_error(Y, prediction1)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[13]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, X, Y, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
display_scores(tree_rmse_scores)


# In[14]:


lin_scores = cross_val_score(lin_reg, X, Y, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[15]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X,Y)
prediction2 = forest_reg.predict(X)
forest_mse = mean_squared_error(Y, prediction2)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
scores = cross_val_score(forest_reg, X, Y, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)


# In[16]:


from sklearn.model_selection import GridSearchCV
grid_param = {  
    'n_estimators': [400, 450, 500, 550, 600],
    'max_features': [2,3,4,5,6,7]
}
gd_sr = GridSearchCV(estimator=forest_reg,  
                     param_grid=grid_param,
                     scoring='neg_mean_squared_error',
                     cv=5,refit=True)
gd_sr.fit(X,Y)


# In[17]:


gd_sr.best_params_


# In[18]:


final_model = gd_sr.best_estimator_


# In[19]:


feature_importance = gd_sr.best_estimator_.feature_importances_
feature_importance


# In[20]:


X_test = test_set.drop("MEDV", axis=1)
Y_test = test_set["MEDV"].copy()
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

