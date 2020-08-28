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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_data(database_filepath):
    # load data from sql database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    
    # set x to messages
    X = df['message'].values
    
    # set y to category columns
    Y_df = df.drop(columns=['id', 'message', 'original', 'genre', 'related'])
    Y = Y_df.values
    category_names = Y_df.columns
    
    return X, Y, category_names

def tokenize(text):
    words = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).strip())
    words = [w for w in words if w not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, 'v') for w in lemmed]
    lemmed = [WordNetLemmatizer().lemmatize(w, 'a') for w in lemmed]
    lemmed = [WordNetLemmatizer().lemmatize(w, 'r') for w in lemmed]
    return lemmed

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),

        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {}

    cv = GridSearchCV(pipeline, parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names = category_names)
    accuracy = (Y_pred, == Y_test).mean()
    print("Classification Report:\n", class_report)
    print("Accuracy:", accuracy)

def save_model(model, model_filepath):
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