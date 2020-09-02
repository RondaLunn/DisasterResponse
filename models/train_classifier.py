import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    INPUT
    database_filepath - string, the filepath for the database from which to load the data
    
    OUTPUT
    X - array of strings, the messages to be classified
    Y - array of integers, the correct categories for each message
        represented in binary 1 = in category, 0 = not in category
    category_names - array of strings, the names for the category columns
    '''
    # load data from sql database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    
    # set x to messages
    X = df['message'].values
    
    # set y to category columns
    Y_df = df.drop(columns=['id', 'message', 'original', 'genre'])
    Y = Y_df.values
    category_names = Y_df.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT
    text - array of strings, the messages to be tokenized and lemmed for classification
    
    OUTPUT
    lemmed - array of strings, the tokenized and lemmed strings ready for the machine learning algorithm
    '''
    words = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).strip())
    words = [w for w in words if w not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, 'v') for w in lemmed]
    lemmed = [WordNetLemmatizer().lemmatize(w, 'a') for w in lemmed]
    lemmed = [WordNetLemmatizer().lemmatize(w, 'r') for w in lemmed]
    return lemmed

def build_model():
    '''
    INPUT
    None
    
    OUTPUT
    cv - sklearn model, the best fit model for the text classification
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),

        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        # for speed unused parameters are commented out, and best parameters are used
        # uncomment parameters to retest
        # CountVectorizer: 
        #'features__nlp_pipeline__vect__max_df': [0.5, 0.75, 1.0],
        'features__nlp_pipeline__vect__max_df': [0.5],
        #'features__nlp_pipeline__vect__min_df': [0.0, 0.25],

        # Tfidf: 
        #'features__nlp_pipeline__tfidf__norm': ['l2', 'l1'],
        'features__nlp_pipeline__tfidf__norm': ['l1'],
        #'features__nlp_pipeline__tfidf__sublinear_tf': [False, True],

    #     # RandomForestClassifier: 
    #     'clf__estimator__n_estimators': [100, 50, 25, 10, 5, 1],
    #     'clf__estimator__max_depth': [None, 5, 10, 25, 100],
    #     'clf__estimator__criterion': ['gini', 'entropy'],

    #     # KNeighborsClassifier: 
    #     'clf__estimator__n_neighbors': [1, 2, 5, 10, 20],
    #     'clf__estimator__weights': ['uniform', 'distance'],
    #     'clf__estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

        # AdaBoostClassifier: 
        #'clf__estimator__n_estimators': [100, 50],
        'clf__estimator__n_estimators': [100],
    }

    cv = GridSearchCV(pipeline, parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - sklearn model, the classification model for tha data
    X_test - array of strings, the test set of messages to classify
    Y_test - array of integers, the test set of correct categories for the test messages
            represented in binary 1 = in category, 0 = not in category
    category_names - array of strings, the names of the categories
    
    OUTPUT
    None - The classification report is printed to the terminal
    '''
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names = category_names)
    accuracy = (Y_pred == Y_test).mean()
    print("Classification Report:\n", class_report)
    print("Accuracy:", accuracy)
    print(model.best_params_)

def save_model(model, model_filepath):
    '''
    INPUT
    model - sklearn model, the classification model for the messages
    model_filepath - string, the pathname where the model will be saved
    
    OUTPUT
    None - The model is saved to a pickle file 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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