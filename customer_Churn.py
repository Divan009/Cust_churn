import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

%matplotlib inline
pd.set_option('display.max_columns', None)

data = pd.read_csv('Churn.csv')

data.head()
data.columns
data.shape
data.describe(include = 'all') #Total charges is an object type??
data.info()

data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')

print ("\nUnique values :  \n",data.nunique())

data.isnull().sum()
#we notice 11 missing values, once we convert the TotalCharges to numerical

#I could fill it up with values, or delete the rows. 
data.dropna(inplace = True)

#We don't really need customerID
data.drop(['customerID'], axis = 1, inplace = True)

#Convert Senior Citizen into object, by replacing it with Y or N
data['SeniorCitizen'] = data['SeniorCitizen'].replace({1:"Yes", 0:"No"})

data.tenure.plot(kind='hist', color='c', bins = 6)
plt.xlabel('months')
plt.title('Tenure')
plt.show()

#We can bin the tenure
bins = [0,12,24,36,48,60,74]
labels = ['1yr','2yr','3yr','4yr','5yr','6yr']
data['tenure_bin'] = pd.cut(data['tenure'], bins=bins, labels=labels)

print(data['tenure_bin'], data['tenure'])
data['tenure_bin'] = data['tenure_bin'].astype(object)


g1=sns.catplot(x="tenure_bin", y="Churn", data=data,kind="bar")
g1.set_ylabels("Churn Probability")






























