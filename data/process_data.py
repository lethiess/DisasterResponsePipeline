import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from collections import defaultdict

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


def load_data(messages_filepath, categories_filepath):
    '''
    Extract the disaster messages with corresponding categories from file.

    Args:
        messages_filepath:      file path to the messages csv-file
        categories_filepath:    file path to the categories csv-file  

    Returns:
        A pandas dataframe containing messages and corresponding categories.
    '''
    # read in messages and categories files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the messages and categories data frames
    return pd.merge(messages, categories, how = "inner", on = "id")
    

def clean_data(df):
    '''
    Transform the data in the merged pandas dataframe. The categories are the
    problem in this dataframe. Each value of the categories columns contrains
    a string with all categories and values in it.

    The categories are separated by a ";" whereas each substring contains the
    category name and value separated by a "-".

    e.g.: "related-1;request-0;offer-0;aid_related-0;medical_help-0;......"

    Args:
        df: This dataframe contrains the disaster messages and categories.

    Return:
        A clean dataframe with separated categories.
    '''
    # split the category value for each row into separate category strings
    # each categorystring contains the name and value of the corresponding category
    # e.g. "related-1" or "water-0" 
    category_rows = [category.split(";") for category in df.categories]

    # create an empty default dictionary of lists
    d = defaultdict(list)
    for i, category_row in enumerate(category_rows):
        # split the category name and value
        category_value_list = [x.split("-") for x in category_row]
        # store/update the values in a dictionry
        d["id"].append(df["id"][i])
        for category in category_value_list:
            d[category[0]].append(category[1])

    # create a new dataframe based on the category dict
    df_categories = pd.DataFrame(d)
    
    # remove the original category column
    df.drop(columns="categories", axis=1, inplace=True)
    
    # merge the updated input dataframe and the categories dataframe
    df_merged = pd.merge(df, df_categories, how = "inner", on = "id")
    
    # drop duplicates
    df_merged.drop_duplicates(inplace=True)

    return df_merged


def save_data(df, database_filepath):
    '''
    Load the cleaned data from the a pandas dataframe into a SQLiete database.
    The database will be stored in the data folder.

    Args:
        df: The pandas dataframe which should be stored in the database.
        database_filepath: The file path and name of the database.
    Return:
        None.
    '''
    try:
        engine = create_engine("sqlite:///" + database_filepath)
        df.to_sql(getFileName(database_filepath), engine, if_exists="replace")  
    except:
        print("Error while writing data to SQLite-database.")


def main():
    '''
    Run the ETL pipeline for the disaster response project.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()