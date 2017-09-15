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
    return string.strip().lower()

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

def vector():
    try:
        clf = joblib.load('model/clf.pkl')
        print 'load model completed !!!'
        return clf
    except:
        clf = None
        if clf == None:
            clf = SVC(kernel='rbf',C=100)

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

                clean_train_reviews = train
                # clean_train_reviews["sentiment"] = clean_train_reviews["id"] == 1
                print clean_train_reviews

                train, test = train_test_split(clean_train_reviews, test_size=0.2)

                print "Creating the bag of words...\n"
                # vectorizer = vector = CountVectorizer(analyzer="word",
                #                  tokenizer=None,
                #                  preprocessor=None,
                #                  stop_words=None,
                #                  max_features=1000)

                vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
                print "AAAA "

                train_text = train["message"].values.astype('str')
                test_text = test["message"].values.astype('str')

                X_train = vectorizer.fit_transform(train_text)
                X_train = X_train.toarray()
                y_train = train["id"]
                print "BBBB "
                # y_train = train["sentiment"]

                X_test = vectorizer.fit_transform(test_text)
                X_test = X_test.toarray()
                y_test = test["id"]
                # y_test = test["sentiment"]
                print "EEEEEEE "
                joblib.dump(vectorizer, 'model/vectorizer.pkl')
                print "FFFFFF"
                # joblib.dump(X_train, 'model/X_train.pkl')
                # joblib.dump(X_test, 'model/X_test.pkl')

                print "---------------------------"
                print "Training"
                print "---------------------------"
                names = ["RBF SVC"]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print y_pred
                print "accuracy: %0.3f" % clf.score(X_test, y_test)
                joblib.dump(clf, 'model/clf.pkl')
                return y_test


def training():
    clf = load_model('model/vectorizer.pkl')
    if clf != None:
        return
    X = load_model('model/X_train.pkl')
    y = []



def fit(mes):
    try:
        vectorizer = load_model('model/vectorizer.pkl')
    except:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)

    # y_test = vector()
    clf = load_model('model/clf.pkl')
    # X_test = load_model('model/X_test.pkl')


                # print "---------------------------"
                # print "Training"
                # print "---------------------------"
                # names = ["RBF SVC"]
                #
                # clf.fit(X_train, y_train)
                # joblib.dump(clf, 'svm.pkl')
    # y_pred = clf.predict(X_test)
    # print y_pred
    # print "predict: %0.3f" % clf.score(X_test,y_test)
    # print "accuracy: %0.3f" % clf.score(X_test, y_test)
    print 'query ', mes
    query = unicode(mes, encoding='utf-8')
    test_message = ViTokenizer.tokenize(query).encode('utf8')
    print "test_message", test_message
    test_message = clean_str_vn(test_message)
    test_message = review_to_words(test_message)
    clean_test_reviews = []
    clean_test_reviews.append(test_message)
    d2 = {"message": clean_test_reviews}
    test2 = pd.DataFrame(d2)
    print test2
    test_text2 = test2["message"].values.astype('str')
    print test_text2
    test_text2 = test2["message"].values
    print test_text2
    test_data_features = vectorizer.transform(test_text2)
    print test_data_features
    test_data_features = test_data_features.toarray()
    print test_data_features

    s = clf.predict(test_data_features)
    print s
    s2 = np.array(s)
    s3 = str(s2[0])
    return s3



# def predict_ex(clf, query):
#     test_message = ViTokenizer.tokenize(query).encode('utf8')
#     test_message = review_to_words(test_message)
#     clean_test_reviews = []
#     clean_test_reviews.append(test_message)
#     d2 = {"message": clean_test_reviews}
#     test2 = pd.DataFrame(d2)
#     test_text2 = test2["message"].values.astype('str')
#     test_data_features = vectorizer.transform(test_text2)
#     test_data_features = test_data_features.toarray()
#
#     s = clf.predict(test_data_features)
#     print "ssssss ", s
#     # print "1 ",type(s)
#     s2 = np.array(s)
#     print "aaaaaaaaa", s2[0]
#     s3 = str(s2[0])
#     print "2 ",type(s3)
#     # s3 = ''.join(s2)
#     # print "3 ",type(s3)
#
#     # print s3
#     return s3

# X1=["message"]; y1=["id"]
# # print ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội"))
#
# # with open('data/labels_quick_response_p2.json') as data_file, open('data/f2.csv', "w") as f2:
# with open('data/labels_quick_response_p2.json') as data_file:
#     data = json.load(data_file)
#     count=0
#     mlist = []
#     i = []
#     # f2.write("message\tid\n")
#     clean_train_reviews=[]
#     col1=[]; col2 = []
#
#     for item in data:
#         try:
#             id1 = item['id']
#             message = item['message']
#             message = ViTokenizer.tokenize(message).encode('utf8')
#             # print message
#             # message = clean_str_vn(message)
#             message = review_to_words(message)
#             # print message
#             # print message
#             i.append(message)
#             try:
#                 # c = (((item.get("labels")[0]).get("5923dd8056837c2c7f06eef0"))[0]).get("id")
#                 c = item['labels'][0]
#                 for d in c.values():
#                     e = d[0]['id']
#                 i.append(e)
#
#                 j = np.array(i)
#                 if(j.shape != (2,)):
#                     print j
#                 else:
#                     mlist.append(i)
#                     col1.append(message)
#                     col2.append(e)
#             except:
#                 count += 1
#                 pass
#
#             # f2.write(message+ "\t" + str(e) + "\n")
#             i = []
#         except:
#             print('Sample exception: %s' % (id1))
#             count += 1
#
#     print len(col1)
#     print "len ", len(col2)
#     # col3=["abc ơi","xyz","cô ơi", "haha"]; col4=[4,3,2,6]
#     # print zip(col3,col4)
#     # print dict(zip(col3, col4))
#     dictionary = dict(zip(col1,col2))
#
#     list2 = np.array(mlist)
#     print "Data dimensions:", list2.shape
#     # print list2
#     b=0
#     for v in list2:
#         v1 = np.array(v)
#         if(v1.shape != (2,)):
#             b+=1
#             print v1
#             print v1.shape
#     # print "bbbbbbbb ",b
#
#
#     # train = get_reviews_data("data/f2.csv")
#     # train = list2
#     # train = dictionary
#     # print "Data dimensions:", train.shape
#     # print train[0]
#
#     # cleaning_data(train, "data/clean_train_reviews.csv")
#
#     # clean_train_reviews = pd.read_csv("data/clean_train_reviews.csv", nrows=1000, error_bad_lines= False)
#
#     d = {"message": col1, "id":col2}
#
#     train = pd.DataFrame(d)
#     print "Data dimensions:", train.shape
#     print "List features:", train.columns.values
#     print "First review:", train["message"][0], "|", train["id"][0]
#     # print train
#
#
#     # clean_train_reviews = get_reviews_data("data/f2.csv")
#     # print "Data dimensions:", clean_train_reviews.shape
#     # print clean_train_reviews
#     # print clean_train_reviews
#
#     clean_train_reviews = train
#     clean_train_reviews["sentiment"] = clean_train_reviews["id"] == 1
#     print clean_train_reviews
#
#     train, test = train_test_split(clean_train_reviews, test_size=0.2)
#     print "train\n", train
#     print "test\n", test
#     # print test
#     #
#     print "Creating the bag of words...\n"
#     vectorizer = CountVectorizer(analyzer="word",
#                                  tokenizer=None,
#                                  preprocessor=None,
#                                  stop_words=None,
#                                  max_features=1000)
#
#     # vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=2000)
#     #
#     train_text = train["message"].values.astype('str')
#     test_text = test["message"].values.astype('str')
#
#     str1 = u"tư vấn cho em với"
#     str2 = u"tư vấn cho mình với 01252985049, về học phí, cách học"
#     str3 = u"tư vấn cho mình với 01252985049, về học phí, cách học."
#
#     test_message = ViTokenizer.tokenize(str1).encode('utf8')
#     test_message2 = ViTokenizer.tokenize(str2).encode('utf8')
#     test_message3 = ViTokenizer.tokenize(str3).encode('utf8')
#
#     # test_message = clean_str_vn(test_message)
#     test_message = review_to_words(test_message)
#     print  test_message
#
#     # test_message2 = clean_str_vn(test_message2)
#     test_message2 = review_to_words(test_message2)
#     print  test_message2
#
#     # test_message3 = clean_str_vn(test_message3)
#     test_message3 = review_to_words(test_message3)
#     print  test_message3
#
#     clean_test_reviews = []
#     clean_test_reviews.append(test_message)
#     clean_test_reviews.append(test_message2)
#     clean_test_reviews.append(test_message3)
#     print "aaaaaaaaa ",clean_test_reviews
#
#     d2 = {"message": clean_test_reviews}
#     test2 = pd.DataFrame(d2)
#
#     test_text2 = test2["message"].values.astype('str')
#     print "bbbbbb ",test_text2
#
#
#     # hàm fit_transform() để chuyển đổi thành ma trận term - document làm input cho các hàm phân lớp
#     # convert data-set to term-document matrix
#
#     X_train = vectorizer.fit_transform(train_text)
#     # print "X_train ", X_train
#     X_train = X_train.toarray()
#     # print "X_train ", X_train
#     # np.asarray(X_train)
#     y_train = train["sentiment"]
#
#     X_test = vectorizer.fit_transform(test_text)
#     X_test = X_test.toarray()
#     print "dddd\n ", X_test
#     # np.asarray(X_test)
#     y_test = test["sentiment"]
#     # print "ccccccccc ", X_test
#
#     test_data_features = vectorizer.transform(test_text2)
#     test_data_features = test_data_features.toarray()
#     print "test_data_features\n", test_data_features
#
#     # print_words_frequency(X_train) # in ra danh sách các từ kèm tần suất xuất hiện của chúng
#
#     """
#     Training
#     """
#
#     print "---------------------------"
#     print "Training"
#     print "---------------------------"
#     names = ["RBF SVC"]
#
#     classifiers = [SVC(kernel='rbf', C=1000)]
#
#     # iterate over classifiers
#     results = {}
#     kq = {}
#     clf = SVC(kernel='rbf',C=100)
#     # clf = RandomForestClassifier(n_estimators = 100)
#     clf.fit(X_train,y_train)
#     print "aaaaaaaaaaaa ",clf.predict(X_test)
#     # clf.fit(X_test, y_test)
#     # print "aaaaaaaaaaaa ", clf.predict(X_test)
#     print y_test
#
#     # print "accuracy: %0.3f" % clf.score(X_test,y_test)
#     print test_data_features
#     print "bbbbbb ",clf.predict(test_data_features)
#
#     print " accuracy: %0.3f" % clf.score(X_test,y_test)
#     print " accuracy: %0.3f" % clf.score(X_train,y_train)
#
#
#
#
#     # for name, clf in zip(names, classifiers):
#     #     print "Training " + name + " classifier..."
#         # clf.fit(X_train, y_train)
#         # score = clf.score(X_test, y_test)
#         # results[name] = score
#         # r = clf.predict(test_data_features)
#
#         # r = clf.predict(X_test)
#         # kq[name] = r
#
#     print "---------------------------"
#     print "Evaluation results"
#     print "---------------------------"
#
#     # sorting results and print out
#     # sorted(results.items(), key=itemgetter(1))
#     # for name in results:
#     #     print name + " accuracy: %0.3f" % results[name]
#     #     print "kqqqqq", kq[name]
#     #     print y_test
#
