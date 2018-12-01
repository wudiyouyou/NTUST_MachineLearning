
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/bank/bank-full.csv",sep=';')

marital_mapping = {'single':1, 'married':2, 'divorced':3}
df['marital'] = df['marital'].map(marital_mapping)

job_mapping = {"admin.":1,"unknown":2,"unemployed":3,"management":4,"housemaid":5,"entrepreneur":6,"student":7,"blue-collar":8,
               "self-employed":9,"retired":10,"technician":11,"services":12}
df['job'] = df['job'].map(job_mapping)

education_mapping = {"unknown":1,"secondary":2,"primary":3,"tertiary":4}
df['education'] = df['education'].map(education_mapping)

default_mapping = {'yes':1, 'no':2}
df['default'] = df['default'].map(default_mapping)

housing_mapping = {'yes':1, 'no':2}
df['housing'] = df['housing'].map(housing_mapping)

loan_mapping = {'yes':1, 'no':2}
df['loan'] = df['loan'].map(loan_mapping)

contact_mapping = {"unknown":1, "telephone":2,"cellular":3}
df['contact'] = df['contact'].map(contact_mapping)

month_mapping = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, 
                 "nov":11, "dec":12}
df['month'] = df['month'].map(month_mapping)

poutcome_mapping = {"unknown":1,"other":2,"failure":3,"success":4}
df['poutcome'] = df['poutcome'].map(poutcome_mapping)

y_mapping = {'yes':1,'no':2}
df['y'] = df['y'].map(y_mapping)

X=df[['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration',
      'campaign','pdays','previous','poutcome']]
Y=df[['y']]

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=27)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5, 5), random_state=1)

print("———— Executing ————")

clf.fit(x_train, y_train)

print("———— Done      ————")

y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)
print("accurancy : %f"%accuracy_score(y_test, y_pred))
print (clf.n_layers_)
print (clf.n_iter_)
print (clf.loss_)
print (clf.out_activation_)


cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)

lcc_ = len(clf.coefs_)

lcc0 = (clf.coefs_[0])

lmi0 = len(clf.intercepts_[0])

