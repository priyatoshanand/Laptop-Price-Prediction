#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade scikit-learn --user


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[3]:


df = pd.read_csv('laptop_data.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.duplicated().sum()


# In[8]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# # Data Preprocessing

# In[9]:


df.drop(columns=['Unnamed: 0'],inplace= True)


# In[10]:


df.head()


# In[11]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[12]:


df.head()


# In[13]:


df['Ram'] = df['Ram'].astype('int')
df['Weight'] = df['Weight'].astype('float')


# In[14]:


df.info()


# In[15]:


sns.distplot(df['Price'])


# In[16]:


df['Company'].value_counts().plot(kind='bar')


# In[17]:


sns.barplot(x= df['Company'],y=df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[18]:


df['TypeName'].value_counts().plot(kind = 'bar')


# In[19]:


sns.barplot(x= df['TypeName'],y=df['Price'])
plt.xticks(rotation = 'vertical')
plt.show()


# In[20]:


sns.distplot(df['Inches'])


# In[21]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[22]:


df['ScreenResolution'].value_counts()


# In[23]:


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[24]:


df.sample(5)


# In[25]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[26]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[27]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[28]:


df.head()


# In[29]:


df['Ips'].value_counts().plot(kind='bar')


# In[30]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[31]:


new = df['ScreenResolution'].str.split('x',n=1,expand = True)


# In[32]:


df['X_resolution'] = new[0]
df['Y_resolution'] = new[1]


# In[33]:


df.head()


# In[34]:


df['X_resolution'] = df['X_resolution'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[35]:


df.head()


# In[36]:


df.info()


# In[37]:


df['X_resolution'] = df['X_resolution'].astype('int')
df['Y_resolution'] = df['Y_resolution'].astype('int')


# In[38]:


df.info()


# In[39]:


df.corr()['Price']


# In[40]:


df['Pixel_density'] = (((df['X_resolution']**2) + (df['Y_resolution']**2))**0.5/df['Inches']).astype('float')


# In[41]:


df.corr()['Price']


# In[42]:


df.drop(columns=['ScreenResolution'],inplace=True)
df.drop(columns=['Inches','X_resolution','Y_resolution'],inplace=True)


# In[43]:


df.head()


# In[44]:


df['Cpu'].value_counts()


# In[45]:


df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[46]:


df.head()


# In[47]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[48]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[49]:


df.head()


# In[50]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[51]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[52]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[53]:


df.head()


# In[54]:


df['Ram'].value_counts().plot(kind='bar')


# In[55]:


sns.barplot(x= df['Ram'],y=df['Price'])
plt.xticks(rotation= 'vertical')
plt.show()


# In[56]:


df['Memory'].value_counts()


# In[57]:


df1 = df.copy()


# In[58]:


df1['Memory'] = df1['Memory'].astype(str).replace('\.0', '', regex=True)
df1["Memory"] = df1["Memory"].str.replace('GB', '')
df1["Memory"] = df1["Memory"].str.replace('TB', '000')
new = df1["Memory"].str.split("+", n = 1, expand = True)

df1["first"]= new[0]
df1["first"]=df1["first"].str.strip()

df1["second"]= new[1]

df1["Layer1HDD"] = df1["first"].apply(lambda x: 1 if "HDD" in x else 0)
df1["Layer1SSD"] = df1["first"].apply(lambda x: 1 if "SSD" in x else 0)
df1["Layer1Hybrid"] = df1["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df1["Layer1Flash_Storage"] = df1["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df1['first'] = df1['first'].str.replace(r'\D', '',regex=True)

df1["second"].fillna("0", inplace = True)

df1["Layer2HDD"] = df1["second"].apply(lambda x: 1 if "HDD" in x else 0)
df1["Layer2SSD"] = df1["second"].apply(lambda x: 1 if "SSD" in x else 0)
df1["Layer2Hybrid"] = df1["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df1["Layer2Flash_Storage"] = df1["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df1['second'] = df1['second'].str.replace(r'\D', '',regex=True)

df1["first"] = df1["first"].astype(int)
df1["second"] = df1["second"].astype(int)

df1["HDD"]=(df1["first"]*df1["Layer1HDD"]+df1["second"]*df1["Layer2HDD"])
df1["SSD"]=(df1["first"]*df1["Layer1SSD"]+df1["second"]*df1["Layer2SSD"])
df1["Hybrid"]=(df1["first"]*df1["Layer1Hybrid"]+df1["second"]*df1["Layer2Hybrid"])
df1["Flash_Storage"]=(df1["first"]*df1["Layer1Flash_Storage"]+df1["second"]*df1["Layer2Flash_Storage"])

df1.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[59]:


df1.sample(5)


# In[60]:


df1.drop(columns = ['Memory'], inplace = True)


# In[61]:


df1.head()


# In[62]:


df1.corr()['Price']


# In[63]:


df1.drop(columns=['Hybrid','Flash_Storage'],inplace = True)


# In[64]:


df1.head()


# In[65]:


df1['Gpu'].value_counts()


# In[66]:


df2= df1.copy()


# In[67]:


df2['Gpu brand']=df2['Gpu'].apply(lambda x:x.split()[0])


# In[68]:


df2.head()


# In[69]:


df2['Gpu brand'].value_counts()


# In[70]:


df2=df2[df2['Gpu brand'] != 'ARM']


# In[71]:


sns.barplot(x=df2['Gpu brand'], y=df2['Price'],estimator = np.median)
plt.xticks(rotation ='vertical')
plt.show()


# In[72]:


df2.drop(columns=['Gpu'],inplace=True)


# In[73]:


df3 = df2.copy()


# In[74]:


df3['OpSys'].value_counts()


# In[75]:


sns.barplot(x=df3['OpSys'],y=df3['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[76]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[77]:


df3['os'] = df3['OpSys'].apply(cat_os)


# In[78]:


df3.drop(columns=['OpSys'],inplace=True)


# In[79]:


df3.head()


# In[80]:


sns.barplot(x=df3['os'],y=df3['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[81]:


sns.distplot(df3['Weight'])


# In[82]:


sns.scatterplot(x=df3['Weight'],y=df3['Price'])


# In[83]:


sns.heatmap(df3.corr())


# # Handling the Skewness of Target column

# In[84]:


sns.distplot(df3['Price'])


# #We can see that the Target column is Left skewed and apllying Log function to overcome this .

# In[85]:


sns.distplot(np.log(df3['Price']))


# In[86]:


df_final = df3.copy()


# In[87]:


X = df_final.drop(columns=['Price'])
y = np.log(df_final['Price'])


# In[88]:


X


# In[89]:


y


# # Train Test Split

# In[90]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[91]:


X_train


# # Model Training
# 1. LinearRegression
# 2. Ridge
# 3. Lasso
# 4. KNeighborsRegressor
# 5. DecisionTreeRegressor
# 6. RandomForestRegressor
# 7. GradientBoostingRegressor
# 8. AdaBoostRegressor
# 9. ExtraTreesRegressor
# 10. SVR
# 11. XGBRegressor
# 
#  Also haandling the Categorical Variable using One Hot Encoding. A one hot encoding is a representation of categorical variables as binary vectors.

# ## LinearRegression

# In[92]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## Ridge

# In[93]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## Lasso

# In[94]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## KNN

# In[95]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## DecisionTreeRegressor

# In[96]:


step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(sparse=False,drop= 'first'),[0,1,7,10,11])
                                       ],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)
 
y_pred = pipe.predict(X_test)

print('R2 score', r2_score(y_test,y_pred))
print('MAE', mean_absolute_error(y_test,y_pred))


# ## SVM

# In[97]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## RandomForestRegressor

# In[98]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## ExtraTreesRegressor

# In[100]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## AdaBoostRegressor

# In[101]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## XGBRegressor

# In[102]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# ## GradientBoostingRegressor

# In[103]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Hyperparameter Tunning using Randomized searchCV
# 
# From above Model trainning I can see that RandomForestRegressor is giving the best R2 score i.e 0.8873. I will further try to improve the score by doing Hyperparameter Tunning using Randomized searchCV.

# In[106]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[107]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

rf = RandomForestRegressor()

# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

step2 = rf_random

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# Here I can see that after doing hyperparameter tunning score is improved and new R2 score is 0.9038 and mean absolute error is also reduced to 0.1443.

# In[ ]:


np.exp(0.143)


# In[ ]:


rf_random.best_params_


# # Stacking

# In[105]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


estimators = [
    ('rf', RandomForestRegressor(n_estimators= 1000,min_samples_split= 2,min_samples_leaf= 1,
                                 max_features= 'sqrt',max_depth=25)),
    
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)),
    ('etr', ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_features=0.75,
                              max_depth=15))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Conclusion
# 
# From above Model training I can conclude that RamdomForestRegressor with Hyperparameter tunning is giving the best R2 score and the best parameter for the model is 
# {'n_estimators': 1000,
# 'min_samples_split': 2,
#  'min_samples_leaf': 1,
#  'max_features': 'sqrt',
#  'max_depth': 25}

# # Exporting the Model

# In[ ]:


import pickle

pickle.dump(df_final,open('df_final.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:





# In[ ]:




