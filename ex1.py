# -*- encoding: utf8 -*-
import re
from operator import itemgetter
import numpy as np
import pandas as pd
from pyvi.pyvi import ViTokenizer, ViPosTagger
import numpy as np
import os
import json

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# print ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội")



def clean_str_vn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[~`@#$%^&*-+]", " ", string)
    def sharp(str):
        b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
        while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
        b = re.sub(r'\s\.\s', ' # ', b)
        return b
    string = sharp(string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    # return string.strip().lower()


def review_to_words(review):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    #
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    with open('data/vietnamese-stopwords-dash.txt', "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()

    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in array]
    # print meaningful_words
    #
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

# def print_words_frequency(train_data_features):
#     # Take a look at the words in the vocabulary
#     vocab = vectorizer.get_feature_names()
#     print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    # dist = np.sum(train_data_features, axis=0)
    #
    # # For each, print the vocabulary word and the number of times it
    # # appears in the training set
    # print "Words frequency..."
    # for tag, count in zip(vocab, dist):
    #     print count, tag

def cleaning_data(dataset, file_name):

    # Get the number of reviews based on the dataframe column size
    num_reviews = dataset["message"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review
    for i in xrange(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if (i + 1) % 10000 == 0:
            print "Review %d of %d\n" % (i + 1, num_reviews)

        # Call our function for each one, and add the result to the list of
        # clean reviews
        id = str(dataset["id"][i])
        message = review_to_words(str(dataset["message"][i]))

        clean_train_reviews.append(message+ "\t" + id )

    print "Writing clean train reviews..."
    with open(file_name, "w") as f:
        f.write("message,id\n")
        for review in clean_train_reviews:
            f.write("%s\n" % review)

def load_model(model):
    print('loading model ...')
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def clean_data():
    with open('data/labels_quick_response_p2.json') as data_file:
        data = json.load(data_file)
        count = 0
        mlist = []
        i = []
        clean_train_reviews = []
        col1 = [];
        col2 = []

        for item in data:
            try:
                id1 = item['id']
                id1 = int(id1)
                message = item['message']
                message = ViTokenizer.tokenize(message).encode('utf8')
                message = clean_str_vn(message)
                message = review_to_words(message)
                i.append(message)
                try:
                    # c = (((item.get("labels")[0]).get("5923dd8056837c2c7f06eef0"))[0]).get("id")
                    c = item['labels'][0]
                    for d in c.values():
                        e = d[0]['id']
                    i.append(e)

                    j = np.array(i)
                    if (j.shape != (2,)):
                        print j
                    else:
                        # if e==1:
                        #     mlist.append(i)
                        #     col1.append(message)
                        #     col2.append(e)
                        # else:
                        #     if id1%4==0:
                        #         mlist.append(i)
                        #         col1.append(message)
                        #         col2.append(e)
                        mlist.append(i)
                        col1.append(message)
                        col2.append(e)
                except:
                    count += 1
                    pass
                i = []
            except:
                print('Sample exception: %s' % (id1))
                count += 1
        dictionary = dict(zip(col1, col2))

        list2 = np.array(mlist) # chuyen mlist thanh array
        print "Data dimensions:", list2.shape
        # print list2
        b = 0
        for v in list2:
            v1 = np.array(v)
            if (v1.shape != (2,)):
                b += 1
                print v1

        d = {"message": col1, "id": col2}

        train = pd.DataFrame(d)
        print "Data dimensions:", train.shape
        print "List features:", train.columns.values
        print "First review:", train["message"][0], "|", train["id"][0]
        count1 = 0;count2 =0
        for item in train["id"]:
            if item == 1:
                count1 += 1
            if item == 0:
                count2 += 1
        print count1, count2
        return train

def training():
    clean_train_reviews = clean_data()
    print clean_train_reviews

    train, test = train_test_split(clean_train_reviews, test_size=0.2)

    print "Creating the bag of words...\n"
    # vectorizer = vector = CountVectorizer(analyzer="word",
    #                  tokenizer=None,
    #                  preprocessor=None,
    #                  stop_words=None,
    #                  max_features=1000)

    vectorizer = load_model('model/vectorizer.pkl')
    if vectorizer==None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)

    train_text = train["message"].values
    test_text = test["message"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)

    X_train = X_train.toarray()
    y_train = train["id"]

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["id"]
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    fit(X_train,y_train)
    clf = load_model('model/clf.pkl')
    y_pred = clf.predict(X_test)
    print y_pred
    print "accuracy: %0.3f" % accuracy_score(y_test, y_pred)


def fit(X_train, y_train):
    clf = SVC(kernel='rbf', C=100)
    clf.fit(X_train,y_train)
    joblib.dump(clf, 'model/clf.pkl')

def predict_ex( mes):
    vectorizer = load_model('model/vectorizer.pkl')
    # vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    if vectorizer==None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)

    clf = load_model('model/clf.pkl')
    if clf==None:
        training()

    clf = load_model('model/clf.pkl')
    mes = unicode(mes, encoding='utf-8')
    test_message = ViTokenizer.tokenize(mes).encode('utf8')
    test_message = clean_str_vn(test_message)
    test_message = review_to_words(test_message)
    clean_test_reviews = []
    clean_test_reviews.append(test_message)
    d2 = {"message": clean_test_reviews}
    test2 = pd.DataFrame(d2)
    test_text2 = test2["message"].values.astype('str')
    test_data_features = vectorizer.transform(test_text2)
    test_data_features = test_data_features.toarray()
    # print test_data_features
    s = clf.predict(test_data_features)
    s2 = np.array(s)
    s3 = str(s2[0])
    return s3
