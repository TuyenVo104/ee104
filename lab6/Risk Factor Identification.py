#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, chi2_contingency
from sklearn import preprocessing,linear_model
le = preprocessing.LabelEncoder()


# 

# In[3]:


df= pd.read_csv('./hmeq.csv')
df.BAD.mask(df.BAD == 0,"Cleared")
df.BAD.mask(df.BAD == 1,"Defaulted")
df.BAD = le.fit_transform(df.BAD)
# def hotEncode(df:pd.DataFrame,column:str):
#     new_columns=list(filter(lambda val: not pd.isnull(val),df[column].unique()))
#     for append in new_columns:
#         df[append]=df[column]==append
#         df[append]=df[append].astype(int)
#     df.drop(column,axis=1)
#     return df

# df = hotEncode(df,"REASON")
# df = hotEncode(df,"JOB")
# df = df.drop(["JOB","REASON"],axis=1)


# In[4]:


print("columns",list(df.columns.values))
data=df[df.columns.values]
sns.pairplot(df,kind="scatter")
plt.show()


# In[6]:


corr=df.corr()
corr = corr.iloc[0]
corr
pd.crosstab(df.REASON,df.BAD)
print(chi2_contingency(pd.crosstab(df.REASON,df.BAD))) ## really low p value; no linear relationship
print(chi2_contingency(pd.crosstab(df.JOB,df.BAD)))


# In[7]:


test = df.dropna()
pear = corr[np.abs(corr) > 0] ## All columns with a moderate Pearson Correlation Coefficient
# pear = pear.drop("BAD")
pear_columns = []
for dataset in pear.iteritems():
    # print(test["BAD"])
    # print(test[dataset[0]])
    print("testing BAD vs ",dataset[0])
    res = (linregress(test["BAD"],test[dataset[0]])) ## All pvalues are great than 0.05
    print(res)
    if(res.pvalue > 0.05):
        pear_columns.append(dataset[0])
# print(np.corrcoef(test["LOAN"].dropna(),test["MORTDUE"]))
# print(linregress(test["LOAN"],test["MORTDUE"]))


# In[6]:


sns.pairplot(df[pear_columns],kind="scatter")


# We can use the plot to determine multicolinearity. Loan has a directly linear proportionality between the other datasets. Mortdue has a linear relatinoship with Value. Therefore either mortdue or value can be removed for the final multivariable regression.

# In[7]:


# Count the column with more data points to determine which variable to use: Mortdue or Value
print(np.count_nonzero(~np.isnan(df.MORTDUE)))
print(np.count_nonzero(~np.isnan(df.VALUE))) # More Value datapoints so this is used instead
# pear_columns.remove("MORTDUE")
# pear_columns.remove("LOAN")


# 

# In[33]:


cleandf = df.fillna(df.median())
X = cleandf[pear_columns]# here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = cleandf['BAD']
# with sklearn
regr = linear_model.LinearRegression()
model = regr.fit(X,Y)
Predicted = model.predict(X)
# regr.fit(Y)


# In[34]:


pd.crosstab(Predicted,df.BAD,)
print(chi2_contingency(pd.crosstab(Predicted,df.BAD,)
))
plt.scatter(Predicted,df.BAD) ## Two groups, the higher the predicted BAD the higher the chance of defaulting
plt.show()


# In[35]:


df["Predicted_BAD"] = Predicted


# In[36]:


mean = df.Predicted_BAD[df.BAD == 1].mean()
std = df.Predicted_BAD[df.BAD == 1].std()
HighRiskBound = mean + std
LowRiskBound = mean - std


# In[37]:


highRiskMask = df.Predicted_BAD >= HighRiskBound
lowRiskMask = df.Predicted_BAD <= LowRiskBound
mediumRiskMask= ~(lowRiskMask | highRiskMask)
df["RiskLevel"]=np.zeros(len(df["BAD"]))
df["RiskLevel"] = df.RiskLevel.mask(highRiskMask,"High Risk") 
df["RiskLevel"] = df.RiskLevel.mask(lowRiskMask,"Low Risk") 
df["RiskLevel"] = df.RiskLevel.mask(mediumRiskMask,"Medium Risk") 
# mask = df["RiskLevel"] == True
# df["RiskLevel"]=df["RiskLevel"].where(~mask,other="High Risk")


# In[ ]:




