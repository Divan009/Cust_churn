import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
data["Churn"] = data["Churn"].replace({"Yes":1,"No":0})
#We can bin the tenure
bins = [0,12,24,36,48,60,74]
labels = ['1yr','2yr','3yr','4yr','5yr','6yr']
data['tenure_bin'] = pd.cut(data['tenure'], bins=bins, labels=labels)

data['tenure_bin'] = data['tenure_bin'].astype(object)
data['SeniorCitizen'] = data['SeniorCitizen'].astype(object)
data['Churn'] = data['Churn'].astype(object)

#data.drop(['tenure'], axis = 1, inplace = True)
#Separating categorical and numerical cols
target_col = ["Churn"]
#we add all the categorical var w less than 6 categories to these list
cat_cols = data.nunique()[data.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in data.columns if x not in cat_cols + target_col]


# DATA E X P L O R A T I O N



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
# 6 binary features
# 9 features w 3 levels
#1 w 4 levels

ax = (data['SeniorCitizen'].value_counts()*100.0 /len(data))\
.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 )                                                                           
ax.set_ylabel('Senior Citizens',fontsize = 12)
ax.set_title('% of Senior Citizens', fontsize = 12)

#senior and gender
sns.countplot(x="gender", data=data, hue="Churn")

sns.countplot(x="SeniorCitizen", data=data, hue="Churn")

g = sns.FacetGrid(data, row='SeniorCitizen', col="gender", hue="Churn", height=3.5)
g.map(plt.scatter, "tenure", "MonthlyCharges", alpha=0.6)
g.add_legend();

#partner and dependent
sns.countplot(x="Partner", data=data, hue="Churn")
sns.countplot(x="Dependents", data=data, hue="Churn")

# Phone and Internet Services
sns.countplot(x="PhoneService", data=data, hue="Churn")

sns.countplot(x="MultipleLines", data=data, hue="Churn")

sns.violinplot(x="MultipleLines", y="MonthlyCharges", hue="Churn",
                    data=data, palette="muted")

sns.countplot(x="InternetService", data=data, hue="Churn")

sns.violinplot(x="InternetService", y="MonthlyCharges", hue="Churn",
                    data=data, palette="muted")

# 6 other features

col6 = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
data1 = pd.melt(data[data["InternetService"] != "No"][col6]).rename({'value': 'Has service'}, axis=1)
plt.figure(figsize=(10, 4.5))
ax = sns.countplot(data=data1, x='variable', hue='Has service')
ax.set(xlabel='Additional service', ylabel='Num of customers')
plt.show()

plt.figure(figsize=(10, 4.5))
df1 = data[(data.InternetService != "No") & (data.Churn == "Yes")]
df1 = pd.melt(df1[col6]).rename({'value': 'Has service'}, axis=1)
ax = sns.countplot(data=df1, x='variable', hue='Has service', hue_order=['No', 'Yes'])
ax.set(xlabel='Additional service', ylabel='Num of churns')
plt.show()

# paperless billing and payment
sns.violinplot(x="PaperlessBilling", y="MonthlyCharges", hue="Churn",
                    data=data, palette="muted")
sns.countplot(x="PaperlessBilling", data=data, hue="Churn")


plt.figure(figsize=(10, 4.5))
sns.countplot(x="PaymentMethod", data=data, hue="Churn")

#tenure_bin
sns.countplot(x="tenure_bin", data=data, hue="Churn")

#########################################
# 'tenure','MonthlyCharges', 'TotalCharges',
#Index(['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
#      'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
#      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
#       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
#        'tenure_bin'],
#      dtype='object')
cat_val = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
      'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'tenure_bin']

for var in cat_val:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_val=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
      'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'tenure_bin']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_val]

data_final=data[to_keep]
data_final.columns.values

# Separate the target value
X = data_final.loc[:, data_final.columns != 'Churn']
y = data_final.loc[:, data_final.columns == 'Churn']

##### LOGISTIC REGRESSIONS
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Create Train & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
y_train=y_train.astype('int')
y_test=y_test.astype('int')

###  MODELSSS

# Running logistic regression model
model = LogisticRegression()
result = model.fit(X_train, y_train)

prediction_test = model.predict(X_test)
# Print the prediction accuracy
print(metrics.accuracy_score(y_test, prediction_test))
cm1 = confusion_matrix(y_test, prediction_test)
print('Confusion Matrix : \n', cm1)

def cm_ans(cm):
    total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
    accuracy1=(cm[0,0]+cm[1,1])/total1
    print ('Accuracy : ', accuracy1)
    sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    print('Specificity : ', specificity1)
    return total1

cm_ans(cm1)

# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print(weights.sort_values(ascending = False)[:10].plot(kind='bar'))

print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))


#### RANDOM FOREST


from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
y_train=y_train.astype('int')
y_test=y_test.astype('int')
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))

importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')



## SVM

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
y_train=y_train.astype('int')
y_test=y_test.astype('int')

from sklearn.svm import SVC

model.svm = SVC(kernel='linear') 
model.svm.fit(X_train,y_train)
preds = model.svm.predict(X_test)
metrics.accuracy_score(y_test, preds)

# Print the prediction accuracy
print(metrics.accuracy_score(y_test, prediction_test))
cm1 = confusion_matrix(y_test, prediction_test)
print('Confusion Matrix : \n', cm1)

def cm_ans(cm):
    total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
    accuracy1=(cm[0,0]+cm[1,1])/total1
    print ('Accuracy : ', accuracy1)
    sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    print('Specificity : ', specificity1)
    return total1

cm_ans(cm1)


# AdaBoost Algorithm
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
# n_estimators = 50 (default value) 
# base_estimator = DecisionTreeClassifier (default value)
model.fit(X_train,y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)



### XGBOOST
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
metrics.accuracy_score(y_test, preds)