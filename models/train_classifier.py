import sys
from collections import defaultdict 
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def getFileName(filepath):
    '''
    Extract the file name from the file path. This includes
    removing the path and the file extension.

    Args:
        database_filepath: The path to the file 

    Return:
        Name of the Database without path and file extension.
    '''
    file_name = ""
    # split path from file name
    try:
        file_name = filepath.rsplit("\\",1)[1]
    except:
        file_name = filepath
    # split file extension and return name 
    return file_name.split(".")[0]


def load_data(database_filepath, random_drop = False, random_drop_percentace = 0.75):
    '''
    Load data from a datebase and split it into variables
    for training a machine learning model. The data in the
    database is the preprocessed data from the ETL pipeline.

    Args:
        database_filepath: The filepath and name of the database.
        random_drop:       Optional parameter flag for randomly drop rows.
        random_drop_percentage: Optional parameter which defines how many rows will be dropped.
    Returns:
        X: This pandas series contrains all messages.
        Y: This pandas series contrains the response to X.
        categories: This list contains all categories of Y.
    '''
    # open database and read data to a pandas dataframe
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("SELECT * FROM " + getFileName(database_filepath), engine)

    # for test purposes on the ML pipeline it is useful to shrink the dataset
    if random_drop:
        drop_n = int(df.shape[0] * random_drop_percentace)
        drop_indices = np.random.choice(df.index, drop_n, replace = False)
        df = df.drop(drop_indices)

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
    sklearn's pipeline and Gridsearch for optimal
    parameters.

    Args:
        None
    
    Returns:
        A gridsearch object of the ML pipeline.
    '''
    # text processing and model pipeline
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ("clf", MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
            'clf__estimator__n_estimators': [25, 50, 75, 100, 200],
            'clf__estimator__learning_rate': [0.25, 0.5, 0.75, 1.0]
        }


    # create gridsearch object and return as final model pipeline
    return GridSearchCV(pipeline, param_grid=parameters, n_jobs = 3, verbose=3)


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
    Y_pred = pd.DataFrame(model.predict(X_test))
    Y_pred.columns = category_names

    # evaluate each predicted category
    reports = {}
    for category in category_names:
        report = classification_report(Y_test[category], Y_pred[category], output_dict = True)
        
        reports[category] = {}
        for var in report:
            if var == "accuracy":
                reports[category]["accuracy"] = report[var]
            else:
                reports[category]["f1_score_" + var] = report[var]["f1-score"]
                reports[category]["precision_" + var] = report[var]["precision"]
                reports[category]["recall_" + var] = report[var]["recall"]

    # convert reports to pandas dataframe
    df_reports = pd.DataFrame(reports).T
    df_reports = df_reports[["f1_score_0", "f1_score_1", "f1_score_macro avg", "f1_score_weighted avg"]]
    df_report_mean = df_reports.mean()
    print(df_report_mean)


def save_model(model, model_filepath):
    '''
    Save the model as a pickle file, so it can be used again
    without new training.

    Args:
        model:
        model_filepath:
    
    Returns:
        None
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    '''
    Run the Machine Learning Pipeline.
    '''
    # download the required nltk packages if not available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/wordnet')
    except LookupError:
        nltk.download(["punkt", "wordnet"])

    # check if the number of arguments are correct
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Model parameters:')
        print(model.best_params_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()