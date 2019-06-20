import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

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

#We can bin the tenure
bins = [0,12,24,36,48,60,74]
labels = ['1yr','2yr','3yr','4yr','5yr','6yr']
data['tenure_bin'] = pd.cut(data['tenure'], bins=bins, labels=labels)

print(data['tenure_bin'], data['tenure'])
data['tenure_bin'] = data['tenure_bin'].astype(object)

#data.drop(['tenure'], axis = 1, inplace = True)
#Separating categorical and numerical cols
target_col = ["Churn"]
#we add all the categorical var w less than 6 categories to these list
cat_cols = data.nunique()[data.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in data.columns if x not in cat_cols + target_col]


# DATA E X P L O R A T I O N

#Index(['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
#      'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
#      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
#       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
#       'MonthlyCharges', 'TotalCharges', 'Churn', 'tenure_bin'],
#      dtype='object')

tenure_freq = data["tenure_bin"].value_counts()
churn_freq = data["Churn"].value_counts()

sns.countplot(x="tenure_bin", data=data)

sns.countplot(x="Churn", data=data)

data.tenure.plot(kind='hist', color='c', bins = 6)
plt.xlabel('months')
plt.title('Tenure')
plt.show()

#low tenure + high monthly charges = churn more
sns.pairplot(data,vars = ['tenure','MonthlyCharges','TotalCharges'], 
             hue="Churn")

contract = data['Contract'].value_counts()
print("Condition counting: ")
print(contract)
sns.countplot(x='Contract', data=data)

fig, ax = plt.subplots(ncols=2, figsize=(14,5))
sns.violinplot(x="Contract", y="MonthlyCharges", hue="Churn",
                    data=data, palette="muted", ax = ax[0])
sns.violinplot(x="InternetService", y="MonthlyCharges", hue="Churn",
                    data=data, palette="muted", ax = ax[1])

#### VIZ categorical variable





























