import sys
import re
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - string containing path to the csv containing the messages
    categories_filepath - string containing path to the csv containing the categories
    
    OUTPUT
    df - pandas dataframe with messages and categories
    '''
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='left', on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    
    # select first row of categories to extract column names
    row = categories[:1]
    category_colnames = []
    for category in row:
        category_colnames.append(row[category].str.split('-', expand=True)[0][0])
        
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]
        
    # replace all category column null values with 0 and any other non-binary characters with 1
        categories[column] = categories[column].fillna(0)
        categories[column] = categories[column].str.replace(r'[^0-1]', '1', regex=True)
    
    # convert category column values from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the messages and categories dataframes
    df = pd.concat([messages, categories], axis=1)
    
    return df

def clean_data(df):
    '''
    INPUT
    df - pandas dataframe with messages and categories
    
    OUTPUT
    df - pandas dataframe with messages and categories after cleaning
    '''
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    INPUT
    df - pandas dataframe with messages and categories
    database_filename - string, file name of the database where the dataframe will be saved
    
    OUTPUT
    NONE
    
    Saves the dataframe to the specified database
    '''
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace') 


def main():
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