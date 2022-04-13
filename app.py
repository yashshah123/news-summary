from flask import Flask,render_template,url_for,request
from flask_material import Material
import nltk
from newspaper import Article
from htmldate import find_date
from textblob import TextBlob
import joblib
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
news_classification = joblib.load('data/news_classification.pkl')
loaded_vectorizer = joblib.load(open('data/vectorizer.pickle', 'rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/',methods=["POST"])
def analyze():
    if request.method == 'POST':
        url = request.form['url']
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        data=[]
        analysis = TextBlob(article.text)
        data = loaded_vectorizer.transform([article.title]).toarray()
        title = article.title
        author = article.authors
        date = find_date(url)
        news_type = news_classification.predict(data)
        keywords = article.keywords
        summary = article.summary
        if analysis.polarity > 0: 
            sentiment = "Positive" 
        elif analysis.polarity < 0:
            sentiment="Negative"
        else:
            sentiment="Neutral"
    
    return render_template('index.html',url = url,
            title = title,
            author = author,
            date = date,
            news_type = news_type,
            keywords = keywords,
            summary = summary,
            sentiment = sentiment)

if __name__ == '__main__':
	app.run(debug=True)
