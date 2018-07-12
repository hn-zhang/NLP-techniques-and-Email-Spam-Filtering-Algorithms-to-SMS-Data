import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords



# read data
sms = pd.read_csv("/Users/hainingzhang/Desktop/class/6494 learning/project/spam.csv",encoding='latin-1')
sms.head()

sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
sms = sms.rename(columns = {'v1':'label','v2':'body'})
sms.head()

sms.groupby('label').describe()


# get more features
sms['length'] = sms['body'].apply(lambda x : len(x)- x.count(" "))
sms.hist(column='length', by='label', bins=50)
plt.show()

sms['number'] = sms['body'].apply(lambda x : len(re.findall(r'\d+',x)))
sms['number_port']= sms['number']/sms['length']
sms.hist(column='number_port', by='label', bins=50)
plt.show()


sms['capital'] = sms['body'].apply(lambda x : len(re.findall(r'([A-Z])',x)))
sms['capital_port']= sms['capital']/sms['length']
sms.hist(column='capital_port', by='label', bins=50)
plt.show()


# remove stopwords
text_copy = sms['body'].copy()

stop_words = stopwords.words('english')
stopwords = [str(stop_words[x]) for x in range(len(stop_words))]
print(stopwords[0:10])
stop_words = set(stop_words)

def text_clean(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text_filter = []
    for w in text.split():
        if w.lower() not in stop_words:
            text_filter.append(w)
    words = ""
    for i in text_filter:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "

    return words


text_cleaned = text_copy.apply(text_clean)

#  Convert Text to Sparse Matrices with Vectorizers

vectorizer1 = CountVectorizer("english")
features_count = vectorizer1.fit_transform(text_cleaned)
vectorizer2 = TfidfVectorizer("english", norm = "l2")
features_tfidf = vectorizer2.fit_transform(text_cleaned)




# split to get train and test sets

x_train1,x_test1,y_train1,y_test1 = train_test_split(features_count,sms["label"],
                                                     test_size = 0.2, random_state = 10)
x_train2,x_test2,y_train2,y_test2 = train_test_split(features_tfidf,sms["label"],
                                                     test_size = 0.2, random_state = 10)








from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier



# CountVectorizer, tune parameters for each algrithm
# MultinomialNB
pred_scores = []
for i in np.linspace(0.05, 1, num=20):
    mnb = MultinomialNB(alpha=i)
    mnb.fit(x_train1, y_train1)
    pred = mnb.predict(x_test1)
    pred_scores.append((i, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# Logistic regression with l2 penalty
slvr = {'newton-cg' : 'newton-cg', 'lbfgs': 'lbfgs', 'liblinear': 'liblinear', 'sag': 'sag'}
pred_scores = []
for k,v in slvr.items():
    lrc = LogisticRegression(solver=v, penalty='l2')
    lrc.fit(x_train1, y_train1)
    pred = lrc.predict(x_test1)
    pred_scores.append((k, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]


# Logistic regression with l1 penalty
lrc2 = LogisticRegression(solver='liblinear', penalty='l1')
lrc2.fit(x_train1, y_train1)
pred = lrc2.predict(x_test1)
accuracy_score(y_test1,pred)


# Logistic regression with elasticnet penalty
lrc3 = SGDClassifier(loss='log', penalty='elasticnet')
lrc3.fit(x_train1, y_train1)
pred = lrc3.predict(x_test1)
accuracy_score(y_test1,pred)


# SVM
pred_scores = []
krnl = {'rbf' : 'rbf','polynominal' : 'poly', 'sigmoid': 'sigmoid'}
for k,v in krnl.items():
    for i in np.linspace(0.05, 1, num=20):
        svc = SVC(kernel=v, gamma=i)
        svc.fit(x_train1, y_train1)
        pred = svc.predict(x_test1)
        pred_scores.append((k, [i, accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Gamma','Score'])
df[df['Score'] == df['Score'].max()]

# KNN
pred_scores = []
for i in range(3,50):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(x_train1, y_train1)
    pred = knc.predict(x_test1)
    pred_scores.append((i, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# decision tree
pred_scores = []
for i in range(2,21):
    dtc = DecisionTreeClassifier(min_samples_split=i, random_state=111)
    dtc.fit(x_train1, y_train1)
    pred = dtc.predict(x_test1)
    pred_scores.append((i, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]


# random forest
pred_scores = []
for i in range(2,36):
    rfc = RandomForestClassifier(n_estimators=i, random_state=111)
    rfc.fit(x_train1, y_train1)
    pred = rfc.predict(x_test1)
    pred_scores.append((i, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# AdaBoosting
pred_scores = []
for i in range(25,76):
    abc = AdaBoostClassifier(n_estimators=i, random_state=111)
    abc.fit(x_train1, y_train1)
    pred = abc.predict(x_test1)
    pred_scores.append((i, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# bagging
pred_scores = []
for i in range(2,21):
    bc = BaggingClassifier(n_estimators=i, random_state=111)
    bc.fit(x_train1, y_train1)
    pred = bc.predict(x_test1)
    pred_scores.append((i, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]





# TfidfVectorizer, tune parapmeters for each algorithm
# MultinomialNB
pred_scores = []
for i in np.linspace(0.05, 1, num=20):
    mnb = MultinomialNB(alpha=i)
    mnb.fit(x_train2, y_train2)
    pred = mnb.predict(x_test2)
    pred_scores.append((i, [accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]


# Logistic regression with l2 penalty
pred_scores = []
for k,v in slvr.items():
    lrc = LogisticRegression(solver=v, penalty='l2')
    lrc.fit(x_train2, y_train2)
    pred = lrc.predict(x_test2)
    pred_scores.append((k, [accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# Logistic regression with l1 penalty
lrc2 = LogisticRegression(solver='liblinear', penalty='l1')
lrc2.fit(x_train2, y_train2)
pred = lrc2.predict(x_test2)
accuracy_score(y_test2,pred)

# Logistic regression with elasticnet penalty
lrc3 =   SGDClassifier(loss='log', penalty='elasticnet')
lrc3.fit(x_train2, y_train2)
pred = lrc3.predict(x_test2)
accuracy_score(y_test2,pred)


# SVM
pred_scores = []
krnl = {'rbf' : 'rbf','polynominal' : 'poly', 'sigmoid': 'sigmoid'}
for k,v in krnl.items():
    for i in np.linspace(0.05, 1, num=20):
        svc = SVC(kernel=v, gamma=i)
        svc.fit(x_train2, y_train2)
        pred = svc.predict(x_test2)
        pred_scores.append((k, [i, accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Gamma','Score'])
df[df['Score'] == df['Score'].max()]

# KNN
pred_scores = []
for i in range(3,50):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(x_train2, y_train2)
    pred = knc.predict(x_test2)
    pred_scores.append((i, [accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# decision tree
pred_scores = []
for i in range(2,21):
    dtc = DecisionTreeClassifier(min_samples_split=i, random_state=111)
    dtc.fit(x_train2, y_train2)
    pred = dtc.predict(x_test2)
    pred_scores.append((i, [accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# random forest
pred_scores = []
for i in range(2,36):
    rfc = RandomForestClassifier(n_estimators=i, random_state=111)
    rfc.fit(x_train2, y_train2)
    pred = rfc.predict(x_test2)
    pred_scores.append((i, [accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# AdaBoosting
pred_scores = []
for i in range(25,76):
    abc = AdaBoostClassifier(n_estimators=i, random_state=111)
    abc.fit(x_train2, y_train2)
    pred = abc.predict(x_test2)
    pred_scores.append((i, [accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]

# bagging
pred_scores = []
for i in range(2,21):
    bc = BaggingClassifier(n_estimators=i, random_state=111)
    bc.fit(x_train2, y_train2)
    pred = bc.predict(x_test2)
    pred_scores.append((i, [accuracy_score(y_test2,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df[df['Score'] == df['Score'].max()]





# fit the model for CountVectorizer

mnb = MultinomialNB(alpha=0.05)
svc = SVC(kernel='rbf', gamma=0.05)
knc = KNeighborsClassifier(n_neighbors=3)
dtc = DecisionTreeClassifier(min_samples_split=5, random_state=111)
lrc = SGDClassifier(loss='log', penalty='elasticnet')
rfc = RandomForestClassifier(n_estimators=19, random_state=111)
abc = AdaBoostClassifier(n_estimators=54, random_state=111)
bc = BaggingClassifier(n_estimators=11, random_state=111)

# fit the models and show accuracy


clfs = {'NB': mnb, 'SVC' : svc,'DT': dtc, 'LR': lrc,'KN' : knc, 'RF': rfc, 'AdaBoost': abc, 'BgC': bc}

def train_classifier(clf, feature_train, labels_train):
    clf.fit(feature_train, labels_train)

def predict_labels(clf, features):
    return (clf.predict(features))

pred_scores = []
for k,v in clfs.items():
    train_classifier(v, x_train1, y_train1)
    pred = predict_labels(v,x_test1)
    pred_scores.append((k, [accuracy_score(y_test1,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df





# fit the model for TfidfVectorizer

mnb = MultinomialNB(alpha=0.35)
svc = SVC(kernel='sigmoid', gamma=0.95)
knc = KNeighborsClassifier(n_neighbors=45)
dtc = DecisionTreeClassifier(min_samples_split=4, random_state=111)
lrc = SGDClassifier(loss='log', penalty='elasticnet')
rfc = RandomForestClassifier(n_estimators=9, random_state=111)
abc = AdaBoostClassifier(n_estimators=56, random_state=111)
bc = BaggingClassifier(n_estimators=12, random_state=111)




pred_scores = []
for k,v in clfs.items():
    train_classifier(v, x_train2, y_train2)
    pred = predict_labels(v,x_test2)
    pred_scores.append((k, [accuracy_score(y_test2,pred)]))
df2 = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score2'])
df = pd.concat([df,df2],axis=1)
df



# only use first four classifiers to include additional features

clfs = {'NB': mnb,'SVC' : svc, 'DT': dtc, 'LR': lrc}

lf = sms['length'].as_matrix()
newfeat = np.hstack((features_count.todense(),lf[:, None]))
x_train1,x_test1,y_train1,y_test1 = train_test_split(newfeat,sms["label"],
                                                     test_size = 0.2, random_state = 10)

pred_scores = []
for k,v in clfs.items():
    train_classifier(v, x_train1, y_train1)
    pred = predict_labels(v,x_test1)
    pred_scores.append((k, [accuracy_score(y_test1,pred)]))
df3 = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score3'])
df3


nf = sms['number_port'].as_matrix()
newfeat = np.hstack((features_count.todense(),nf[:, None]))
x_train2,x_test2,y_train2,y_test2 = train_test_split(newfeat,sms["label"],
                                                     test_size = 0.2, random_state = 10)

pred_scores = []
for k,v in clfs.items():
    train_classifier(v, x_train2, y_train2)
    pred = predict_labels(v,x_test2)
    pred_scores.append((k, [accuracy_score(y_test2,pred)]))
df4 = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score4'])
df4

cf = sms['capital_port'].as_matrix()
newfeat = np.hstack((features_count.todense(),cf[:, None]))
x_train3,x_test3,y_train3,y_test3 = train_test_split(newfeat,sms["label"],
                                                     test_size = 0.2, random_state = 10)

pred_scores = []
for k,v in clfs.items():
    train_classifier(v, x_train3, y_train3)
    pred = predict_labels(v,x_test3)
    pred_scores.append((k, [accuracy_score(y_test3,pred)]))
df5 = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score5'])
df5
