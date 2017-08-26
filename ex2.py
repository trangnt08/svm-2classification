# -*- coding: utf-8 -*-
from my_regex import my_regex
from pyvi.pyvi import ViTokenizer
import random
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from io import open
import utils
import json
import os
from sklearn.svm import SVC
from sklearn.externals import joblib


class quick_response:
    def __init__(self):
        self.clf = None
        self.re = my_regex()
        self.vectorizer = None


    def preprocessing(self, msg):
        content = self.re.rm_datetime.sub(u' ', msg)
        content = self.re.rm_url.sub(u' ', content)
        content = self.re.rm_email.sub(u' ', content)
        content = self.re.rm_number.sub(u' ', content)
        content = self.re.rm_special_chars.sub(u' ', content)
        content = self.re.normalize_space.sub(u' ', content)
        return content


    def build_vocab(self, dataset):
        utils.mkdir('model')
        self.vectorizer = self.load_model('model/vectorizer.pkl')
        total_vector = self.load_model('model/total_vector.pkl')
        if self.vectorizer != None and total_vector != None:
            return total_vector
        print('build vocab ....')
        list_comment = []; total_vector = {}
        with open(dataset, 'r', encoding='utf-8') as f:
            j = json.load(f)
            for d in j:
                id = d['id']; msg = d['message']
                msg = ViTokenizer.tokenize(msg)
                msg = self.preprocessing(msg)
                list_comment.append(msg.lower())
                total_vector.update({id:msg.lower()})
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
        self.vectorizer.fit(list_comment)
        for k, v in total_vector.items():
            total_vector[k] = self.vectorizer.transform([v])
        joblib.dump(self.vectorizer, 'model/vectorizer.pkl')
        joblib.dump(total_vector, 'model/total_vector.pkl')
        return total_vector


    def load_model(self, model):
        print('loading model ...')
        if os.path.isfile(model):
            return joblib.load(model)
        else:
            return None


    def training(self, dataset, total_vector):
        self.clf = self.load_model('model/model.pkl')
        if self.clf != None:
            return
        X = []; y = []
        count = 0
        with open(dataset, 'r', encoding='utf-8') as f:
            j = json.load(f)
            for d in j:
                try:
                    id = d['id']
                    v = total_vector[id]
                    cmt = d['labels'][0]
                    for c in cmt.values():
                        category = c[0]['id']
                        X.append(v); y.append(category)
                except:
                    print('Sample exception: %s' % (id))
                    count += 1
        print('There are %d sample\'s exception' % (count))
        X_train, y_train, X_test, y_test = self.split_sample(X, y)
        self.fit(X_train, y_train)
        self.testing(X_test, y_test)


    def fit(self, X_train, y_train):
        print('fit model ...')
        utils.mkdir('model')
        self.clf = SVC(C=5e3, cache_size=2048)
        self.clf.fit(X_train, y_train)
        joblib.dump(self.clf, 'model/model.pkl')


    def testing(self, X_test, y_test):
        print('testing ...')
        y_predict = self.clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_predict)
        print ('accuracy = %f' % (accuracy))


    def split_sample(self, X, y):
        X_train = X; y_train = y
        X_test = None; y_test = []
        N = int(len(X) * 0.2)
        print('total_samples = %d - test_sample = %d' % (len(X), N))
        indices = random.sample(xrange(N), N)
        for i, v in enumerate(indices):
            if i == 0:
                X_test = csr_matrix(X[v])
            else:
                X_test = utils.append_csr_matrix(X_test, X[v])
            X_train = X_train[:v] + X_train[v+1:]
            y_test.append(y[v])
            y_train = y_train[:v] + y_train[v+1:]
        X_train = utils.list2matrix(X_train)
        return X_train, y_train, X_test, y_test


    def predict(self, cmt):
        msg = utils.make_unicode(cmt)
        msg = ViTokenizer.tokenize(msg)
        msg = self.preprocessing(msg).lower()
        v = self.vectorizer.transform([msg])
        label = self.clf.predict(v)[0]
        return label


if __name__ == '__main__':
    qr = quick_response()
    total_vectors = qr.build_vocab('data/labels_quick_response_p2.json')
    qr.training('data/labels_quick_response_p2.json', total_vectors)