import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

##1. Load the Dataset

df = pd.read_table("C:\Machine learning\SMS spam detection\smsspamcollection\SMSSpamCollection",header = None,encoding = 'utf-8')

#print useful information
print(df.info())
#print(df.head())

#check class distribution
classes = df[0]
print(classes.value_counts())
#print(classes.count())
outlier_fraction = 747/classes.count()
print(outlier_fraction)
##2. Preprocessing

#convert class labels to numeric 0 = ham 1 = spam

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

Y = encoder.fit_transform(classes)

#print(Y[:10])

#Store the SMS data

text_messages = df[1]
#print(text_messages[:10])

# use regex to replace emails, urls , phone nos, other numbers and symbols

#replace email addreses with 'emailaddr'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')
#replace urls with 'webaddr'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')
#replace money symbols with money sym
processed = processed.str.replace(r'£|\$', 'moneysymb')

#replace 10 digit phone nos with 'phonenumbr'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
#replace normal numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

#remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

#replace white spaces with single space
processed = processed.str.replace(r'\s+', ' ')

#remove leading and trailing white spaces
processed = processed.str.replace(r'^\s+|\s+?$', '')

#change words to lower case
processed = processed.str.lower()

#print(processed)

#remove stop words from SMS
nltk.download('stopwords')
from nltk.corpus import stopwords
#print(stopwords)

stop_words = set(stopwords.words('english'))
print(len(stop_words))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

#remove word stems using a Porter Stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

print(processed)

nltk.download('punkt')
#nltk.download('word_tokenize')
from nltk.tokenize import word_tokenize

#creating a bag of words
all_words =[]

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

#print total number of words and the 15 most common
print('Number of Words: {}'.format(len(all_words)))
print('Most Common Words: {}'.format(all_words.most_common(15)))

#use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]


#define a find features fn.
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


#lets see an example
features = find_features(processed[0])
for key,value in features.items():
    if value:
        print(key)
print(processed[0])

#find features for all messages
messages = zip(processed,Y)

#define a seed for reproducibility
seed = 1
np.random.seed = seed
#np.random.shuffle(messages)

#call find features for each SMS
featuresets = [(find_features(text),label) for (text,label) in messages]

#split training and testing datasets using sklearn
from sklearn import model_selection

training,testing = model_selection.train_test_split(featuresets,test_size=0.25,random_state=seed)

print('Training: {}'.format(len(training)))
print('Testing: {}'.format(len(testing)))

#4. Scikit Learn Classifiers with NLTK

from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

#Define models to train
names = ['K Nearest Neighbor', 'Decision Tree', 'Random Forest', 'Logistic Regression',
         'SGD Classifier', 'Naive Bayes', 'SVM Linear']
classifiers = [
    KNeighborsClassifier(),

    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names,classifiers))

#wrap models in NLTK
from nltk.classify.scikitlearn import SklearnClassifier

for name,model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model,testing)*100
    print('{}: Accuracy:{}'.format(name,accuracy))

#ensemble method - Voting Classifier
from sklearn.ensemble import VotingClassifier

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models,voting ='hard',n_jobs = -1))
nltk_ensemble.train(training)

accuracy = nltk.classify.accuracy(nltk_ensemble,testing)*100
print('Ensemble Model Accuracy:{}'.format(accuracy))

#make class label prediction for testing set
txt_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(txt_features)

# print a confusion matrix and classification report
print(classification_report(labels,prediction))

print(pd.DataFrame(
    confusion_matrix(labels,prediction),
    index = [['actual','actual'],['ham','spam']],
    columns = [['predicted','predicted'],['ham','spam']]
))



