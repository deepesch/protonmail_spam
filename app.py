from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from features import Features
from flask import request

app = Flask(__name__)

max_features=20000
model_file="models/spam_mnb.pkl"
vocabulary_file="models/spam_vocabulary.pkl"
data_directory = "data/processed"
model_threshold = 0.6


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    spam_detect=Spam_Detect()
    features_extract = Features(vocabulary_file)
    if request.method == 'POST':
        if 'train' in request.form:
            print ('Predict and Train')
            train_nb_spam()
        email = request.form['email']
        data = [email]
        featurevectors=features_extract.extract(data)
        my_prediction = spam_detect.detect(featurevectors)
        return render_template('result.html', prediction=my_prediction)


def load_files():
# We read the files and append them to the ham and spam list
    print ("Generating spam and ham list")
    ham_list = []
    spam_list = []
    for directories, subdirs, files in os.walk(data_directory):
        if (os.path.split(directories)[1]  == 'ham'):
            for file_name in files:      
                with open(os.path.join(directories, file_name), encoding="latin-1") as f:
                    email = f.read()
                    ham_list.append(email)

        if (os.path.split(directories)[1]  == 'spam'):
            for file_name in files:
                with open(os.path.join(directories, file_name), encoding="latin-1") as f:
                    email = f.read()
                    spam_list.append(email)

                    
    return ham_list, spam_list

def train_nb_spam():
    X, y = get_features_by_wordbag()  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)   
    nb_wordbag(X_train, X_test, y_train, y_test)


def nb_wordbag(X_train, X_test, y_train, y_test):
    print ("Naive Bays and wordbag")

    clf = MultinomialNB()
    print  (clf)
    clf.fit(X_train, y_train)
    joblib.dump(clf,model_file)   
    y_pred = clf.predict(X_test)
    print ("f1_score:")
    print (metrics.f1_score(y_test, y_pred))


def get_features_by_wordbag():
    ham_list, spam_list=load_files()
    X=ham_list + spam_list
    y=[0]*len(ham_list)+[1]*len(spam_list)
    vectorizer=None

    if os.path.exists(vocabulary_file):
        vocabulary=joblib.load(vocabulary_file)
        vectorizer = CountVectorizer(
                                     decode_error='ignore',
                                     vocabulary=vocabulary,
                                     strip_accents='ascii',
                                     max_features=max_features,
                                     stop_words='english',
                                     max_df=1.0,
                                     min_df=1 )
    else:
        vectorizer = CountVectorizer(
                                     decode_error='ignore',
                                     strip_accents='ascii',
                                     max_features=max_features,
                                     stop_words='english',
                                     max_df=1.0,
                                     min_df=1 )

    X=vectorizer.fit_transform(X)
    X=X.toarray()

    if not os.path.exists(vocabulary_file):
        vocabulary=vectorizer.vocabulary
        joblib.dump(vocabulary,vocabulary_file)

    return X,y

            
class Spam_Detect(object):

    def __init__(self):
        self.name="Spam_Detect"
        self.clf=joblib.load(model_file)

    def detect(self,featurevectors):
        y_pred = self.clf.predict_proba([featurevectors])[0,-1]
        label = float(y_pred >= model_threshold)
        return label



if __name__ == '__main__':
    app.run(debug=True)



