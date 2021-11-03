#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np
from collections.abc import MutableSequence


# In[45]:


def removeNanRows(df:pd.DataFrame,columns:MutableSequence[str]):
    masks = []
    for column in columns:
        masks.append(~df[column].isna())
    return df[np.bitwise_and(*masks)]


# In[54]:


dat = pd.read_csv("caltrans.csv").replace(' -   ',np.nan)
print('Data:(Rows, Columns)')
print(dat.shape)
dat.head(5)


# df = dat[['route','suffix','county','rank','delay','distance','incidents','incidentsperday','incidentspermiles']]
dat = dat[['delay','distance','incidents']]
dat["incidents"]=pd.to_numeric(dat.incidents)
dat["delay"]=pd.to_numeric(dat.delay)
sns.pairplot(dat, kind="scatter")
plt.show()


# In[56]:


#pearson correlation coefficient
print('\n Correlation - Pearson Correlation Coefficient')
print(dat.corr())
plt.title('Distance traveled vs Delay')
nonan=removeNanRows(dat,["distance", "incidents"])
plt.scatter(nonan.distance,(nonan.incidents))
plt.xlabel('Distance Traveled')
plt.ylabel('Delays')
plt.show()


# In[59]:


#CHI-square test confirmed vs suspect
pd.crosstab(dat.distance,dat.incidents )
print(chi2_contingency(pd.crosstab(dat.distance,dat.incidents)))
plt.scatter(dat.distance,dat.incidents)
plt.xlabel('Distance Traveled')
plt.ylabel('Incidents')
plt.show()


    


# In[60]:


#CHI-square test confirmed vs ICU
print(chi2_contingency(pd.crosstab(dat.distance,dat.delay)))
plt.scatter(dat.distance,dat.delay)
plt.xlabel('Distance Traveled')
plt.ylabel('delay')
plt.show()


# In[61]:


#CHI-square test confirmed vs ICU
print(chi2_contingency(pd.crosstab(dat.incidents,dat.delay)))
plt.scatter(dat.incidents,dat.delay)
plt.xlabel('incidents')
plt.ylabel('delay')
plt.show()


# In[ ]:




