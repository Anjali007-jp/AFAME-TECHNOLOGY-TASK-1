# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:22:01 2024

@author: 91620
"""
"""AFAME TECHNOLOGY
HR DATA ANALYSIS """ 

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv('HR DATA.csv')
pd.set_option('display.max_columns',None)
data

data.head()

data.info()

data.describe()

data.describe(include="O")

""" EDA """

sns.countplot(x=data.Attrition)
plt.show()


sns.countplot(hue=data.Attrition,x=data.BusinessTravel)
plt.show()

sns.countplot(hue=data.Attrition,x=data.Department)
plt.show()

sns.countplot(x=data.Attrition,hue=data.EducationField)
plt.show()

sns.countplot(x=data.Attrition,hue=data.Gender)
plt.show()

sns.countplot(hue=data.Attrition,x=data.OverTime)
plt.show()

plt.figure(figsize=(20,10),facecolor='white')
sns.countplot(x="JobRole",hue="Attrition",data=data)
plt.xlabel('JobRole',fontsize=10)

numerical_col=[]
for column in data.columns:
    if data[column].dtype == "int64" and len(data[column].unique())>= 10:
        numerical_col.append(column)

numerical_col 

sns.histplot(hue=data.Attrition,x=data.Age)
plt.show() 


sns.histplot(hue=data.Attrition,x=data.DistanceFromHome)
plt.show()

sns.histplot(x=data.MonthlyIncome,hue=data.Attrition)
plt.show()

sns.histplot(hue=data.Attrition,x=data.NumCompaniesWorked)
plt.show()

sns.histplot(hue=data.Attrition,x=data.PercentSalaryHike)
plt.show()

sns.histplot(x=data.YearsAtCompany,hue=data.Attrition)
plt.show()

discrete_col=[]
for column in data.columns:
    if data[column].dtype == "int64" and len(data[column].unique())>= 10:
        discrete_col.append(column) 

sns.countplot(hue=data.Attrition,x=data.EnvironmentSatisfaction)
plt.show() 

sns.countplot(x=data.JobLevel,hue=data.Attrition)
plt.show() 

sns.countplot(x=data.JobInvolvement,hue=data.Attrition)
plt.show()

sns.countplot(x=data.StockOptionLevel,hue=data.Attrition)
plt.show()

sns.countplot(x=data.PerformanceRating,hue=data.Attrition)
plt.show()

data.isnull().sum()

data.Attrition=data.Attrition.map({'Yes':1,'No':0})

data.BussinessTravel=data.Attrition.map({'Travel_Frequently':1,'Travel_Rarely':2,'Non-Travel':0})

data.Department=data.Department.map({'Research & Development':2,'Sales':1,'Human Resource':0})

data.EducationField=data.EducationField.map({'Life Science':5,'Medical':4,'Marketing':3,'Technical Degree':2,'Other':1,'Human Resources':0})

data.Gender=pd.get_dummies(data.Gender,drop_first=True)

data.JobRole=data.JobRole.map({'Laboratory Technician':8,'Sales Executive':7,'Research Scientist':6,'Sales Representative':5,'Human Resources':4,'Manufacturing Director':3,'Healthcare Representative':2,'Manager':1,'Reserch & Development':0})

data.MaritalStatus=data.MaritalStatus.map({'Single':2,'Married':1,'Divorced':0})

data.OverTime=data.OverTime.map({'Yes':1,'No':0})

data.info()

""" Feature Selection
Droping Unique Ones """

data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1,inplace=True)

""" Model Creation """

x=data.drop("Attrition",axis=1)
y=data["Attrition"]

print(x.dtypes)

from collections import Counter
from imblearn.over_sampling import SMOTE
sm=SMOTE()
print("Unbalanced :",Counter(y))
x_sm,y_sm=sm.fit_resample(x,y)
print("Balanced :",Counter(y_sm))
