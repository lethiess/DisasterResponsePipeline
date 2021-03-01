import json
import plotly
import pandas as pd
import random
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
db_name = "DisasterResponse"
engine = create_engine("sqlite:///../data/" + db_name + ".DB")
df = pd.read_sql_table(db_name, engine)

# load model
model_name = "DisasterResponseModel"
model = joblib.load("../models/" + model_name + ".pkl")

# color lists
colorlist_gerne = None
colorlist_category = None

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # inspiration for the random colorlist 
    # https://stackoverflow.com/questions/28999287/generate-random-colors-rgb/50218895
    get_n_colors = lambda n : ["#"+''.join([random.choice('0123456789ABCDEF')
                            for j in range(6)]) for i in range(n)]

    #genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index.str.capitalize())
    global colorlist_gerne
    if not colorlist_gerne:
        colorlist_gerne = get_n_colors(len(genre_names))

    # extract data needed for visuals: categories
    category_sum = df.iloc[: , 5:].sum()
    category_name = df.iloc[: , 5:].columns.str.replace("_", " ").str.capitalize()
    global colorlist_category
    if not colorlist_category:
        colorlist_category = get_n_colors(len(category_name))

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = category_name,
                    y = category_sum,
                    marker = dict(color = colorlist_category)
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 33
                }
            }
        },
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts,
                    marker = dict(color = colorlist_gerne)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()