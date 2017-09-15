# -*- encoding: utf-8 -*-
__author__ = 'trangnt'

from flask import Flask, request, flash, render_template
from sklearn.externals import joblib
import ex1
from io import open


app = Flask('crf')

with open('home.html', 'r', encoding='utf-8') as f:
	data1 = f.read()

@app.route('/',methods = ['GET','POST'])
def homepage():
    try:
        error = None
        if request.method == "GET":
            return data1
        if request.method == "POST":
            # data2 = 'annnn'
            # data2 = request.form['text']
            data2 = request.get_data()
            print data2

            kq = ex1.fit(data2)
            print 'kq ',kq
            return kq
            # return data2
            # print svm
            # return "svm"
    except:
        return render_template("home.html", error = error)
	return data
#
# @app.route('/user/<username>')
# def show_user_profile(username):
#     # show the user profile for that user
#     return 'User %s' % username

# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     # show the post with the given id, the id is an integer
#     return 'Post %d' % post_id

@app.route('/svm/', methods=['POST'])
def process_request():
    data = request.form['input']
    flash(data)
    svm = ex1.fit()
    pass
    return ex1.predict(svm, data)



if __name__ == '__main__':
    app.run(port=8000)