import sys
from collections import defaultdict 
import pickle

import nltk
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



def load_data(database_filepath):
    '''
    Load data from a datebase and split it into variables
    for training a machine learning model. The data in the
    database is the preprocessed data from the ETL pipeline.

    Args:
        database_filename: The filepath and name of the database.

    Returns:
        X: This pandas series contrains all messages.
        Y: This pandas series contrains the response to X.
        categories: This list contains all categories of Y.
    '''
    # open database and read data to a pandas dataframe
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)

    # split dataframe in predictor and outcome variables    
    X = df["message"]   
    Y = df.iloc[: , 4:]     
    categories = Y.columns

    return X, Y, categories


def tokenize(text):
    '''
    Split the text of each disaster message into useful tokens.

    Args:
        text: The text of the disaster messages.

    Returns:
        A tokenized version of the original disaster messages.
    '''
    # use nltk to tokenize the message text
    tokens = word_tokenize(text)

    # lemmatize the tokens of each message 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Building a machine leanring model using 
    sklearn's pipeline.

    Args:
        None
    
    Returns:
        The pipeline of the model.
    '''
    # text processing and model pipeline
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    # create gridsearch object and return as final model pipeline

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predict the outcome of the model using the unknown test data.
    The predictions will be evaluated with scikit-learn functions,
    the focus is on f1-socre and accurary.    

    Args:
        model:  The model (pipeline) which should be evaluated.
        X_test: Training data input (desaster messages).
        Y_test: Training data output (disaster category)
        category_names: Names of the disaster categories.

    Returns:
        None
    '''
    # predict 
    Y_pred = pd.Dataframe(pipeline.predict(X_test))

    # evaluate each predicted category
    reports = {}
    for category in category_names:
        output = classification_report(Y_test[category], Y_pred_df[category], output_dict=True)
        
        reports[category] = {}
        for out in output:
            if out == "accuracy":
                reports[category]["accuracy"] = output[category]
            else:
                reports[category]["f1_score_"+out] = output[category]["f1-score"]
                reports[category]["precision_"+out] = output[category]["precision"]
                reports[category]["recall_"+out] = output[category]["recall"]

    # convert reports to pandas dataframe
    df_reports = pd.DataFrame(reports).T
    df_reports_mean = df_reports.mean()
    print(df_reports_mean)


def save_model(model, model_filepath):
    '''


    Args:
        model:
        model_filepath:
    
    Returns:

    '''
    pickle.dump(model, open(filename, "wb"))


def main():
    # download the required nltk packages
    nltk.download(["punkt", "wordnet"])

    # check if the number of arguments are correct
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        #print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()