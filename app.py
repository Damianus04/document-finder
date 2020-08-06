from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

sw = stopwords.words('indonesian') + stopwords.words('english')

app = Flask(__name__)
bootstrap = Bootstrap(app)
df = pd.read_csv("data/bank_central_asia_news.csv", encoding='iso-8859-1')
tfidf = TfidfVectorizer(ngram_range=(
    1, 2), tokenizer=word_tokenize, stop_words=sw)
tfidf_matrix = tfidf.fit_transform(df['Hit Sentence'].values.astype('U'))


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/result", methods=['POST'])
def result():
    if request.method == "POST":
        query = request.form['text']
        # find the most similar document
        vec = tfidf.transform([query])
        dist = cosine_distances(vec, tfidf_matrix)
        result_series = dist.argsort()[0, :10]
        result_list = result_series.tolist()
        result = df['Hit Sentence'][result_list]
        document_list = result.tolist()
    return render_template('result.html', document_list=document_list)
