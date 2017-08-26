# -*- encoding: utf8 -*-
import re
from operator import itemgetter
import numpy as np
import pandas as pd
from pyvi.pyvi import ViTokenizer
import numpy as np
import os
from sklearn import svm
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

# print ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội")

# ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội")

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

def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "Words frequency..."
    for tag, count in zip(vocab, dist):
        print count, tag

def get_reviews_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        df = pd.read_csv(file_name, header=0, delimiter="\t", quoting=3, error_bad_lines=False)
        # df = list(open("data/f2.csv").readlines())
    return df

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


X1=["message"]; y1=["id"]


# with open('data/labels_quick_response_p2.json') as data_file, open('data/f2.csv', "w") as f2:
with open('data/labels_quick_response_p2.json') as data_file:
    data = json.load(data_file)
    count=0
    mlist = []
    i = []
    # f2.write("message\tid\n")
    clean_train_reviews=[]
    col1=[]; col2 = []

    for item in data:
        try:
            id1 = item['id']
            message = item['message']
            message = ViTokenizer.tokenize(message).encode('utf8')
            message = clean_str_vn(message)
            message = review_to_words(message)
            # print message
            i.append(message)
            try:
                # c = (((item.get("labels")[0]).get("5923dd8056837c2c7f06eef0"))[0]).get("id")
                c = item['labels'][0]
                for d in c.values():
                    e = d[0]['id']
                i.append(e)

                j = np.array(i)
                if(j.shape != (2,)):
                    print j
                else:
                    mlist.append(i)
                    col1.append(message)
                    col2.append(e)
            except:
                count += 1
                pass

            # f2.write(message+ "\t" + str(e) + "\n")
            i = []
        except:
            print('Sample exception: %s' % (id1))
            count += 1

    print len(col1)
    print "len ", len(col2)
    col3=["abc ơi","xyz","cô ơi", "haha"]; col4=[4,3,2,6]
    print zip(col3,col4)
    print dict(zip(col3, col4))
    dictionary = dict(zip(col1,col2))

    list2 = np.array(mlist)
    print "Data dimensions:", list2.shape
    # print list2
    b=0
    for v in list2:
        v1 = np.array(v)
        if(v1.shape != (2,)):
            b+=1
            print v1
            print v1.shape
    print "bbbbbbbb ",b


    # train = get_reviews_data("data/f2.csv")
    # train = list2
    # train = dictionary
    # print "Data dimensions:", train.shape
    # print train[0]

    # cleaning_data(train, "data/clean_train_reviews.csv")

    # clean_train_reviews = pd.read_csv("data/clean_train_reviews.csv", nrows=1000, error_bad_lines= False)

    d = {"message": col1, "id":col2}

    train = pd.DataFrame(d)
    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["message"][0], "|", train["id"][0]
    print train


    # clean_train_reviews = get_reviews_data("data/f2.csv")
    # print "Data dimensions:", clean_train_reviews.shape
    # print clean_train_reviews
    # print clean_train_reviews

    clean_train_reviews = train
    clean_train_reviews["sentiment"] = clean_train_reviews["id"] == 1
    # print clean_train_reviews["sentiment"]

    train, test = train_test_split(clean_train_reviews, test_size=0.2)

    # print test
    #
    print "Creating the bag of words...\n"
    # vectorizer = CountVectorizer(analyzer="word",
    #                              tokenizer=None,
    #                              preprocessor=None,
    #                              stop_words=None,
    #                              max_features=1000)

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    #
    train_text = train["message"].values.astype('str')
    test_text = test["message"].values.astype('str')

    # hàm fit_transform() để chuyển đổi thành ma trận term - document làm input cho các hàm phân lớp
    # convert data-set to term-document matrix
    X_train = vectorizer.fit_transform(train_text).toarray()
    y_train = train["sentiment"]

    X_test = vectorizer.fit_transform(test_text).toarray()
    y_test = test["sentiment"]

    # print_words_frequency(X_train) # in ra danh sách các từ kèm tần suất xuất hiện của chúng

    """
    Training
    """

    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["Linear SVC"]

    classifiers = [SVC(kernel='rbf', C=100)]

    # iterate over classifiers
    results = {}
    for name, clf in zip(names, classifiers):
        print "Training " + name + " classifier..."
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results[name] = score

    print "---------------------------"
    print "Evaluation results"
    print "---------------------------"

    # sorting results and print out
    sorted(results.items(), key=itemgetter(1))
    for name in results:
        print name + " accuracy: %0.3f" % results[name]