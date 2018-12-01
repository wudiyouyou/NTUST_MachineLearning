
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import math

dftr = pd.read_csv("./data/all/train.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dftr["text"])

analyze = vectorizer.build_analyzer()

word = vectorizer.get_feature_names()


from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(X)

dftr["text"] = tfidf


savee = pd.DataFrame(dftr["text"])

A_mapping = {'EAP' : 0, 'HPL' : 1, 'MWS' : 2}
dftr['author'] = dftr['author'].map(A_mapping)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(tfidf, dftr["author"], test_size= 0.2, random_state=1)

sc = StandardScaler(copy=True, with_mean=False, with_std=True)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(40,40), random_state=1)

# clf = MLPClassifier(solver='adam', activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,max_iter=50,verbose=10,learning_rate_init=.1)

print("———— Executing ————")

clf.fit(x_train, y_train)

print("———— Done      ————")


y_pred = clf.predict(x_test)

accuracy_score(y_test, y_pred)
print("accuracy : %f"%accuracy_score(y_test, y_pred))
print (clf.n_layers_)
print (clf.n_iter_)
print (clf.loss_)
print (clf.out_activation_)

