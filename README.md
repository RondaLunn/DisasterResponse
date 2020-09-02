# Disaster Response Pipeline Project
This project uses python with pandas, numpy, nltk, and sklearn to build a model to classify messages. 
Disaster response-related messages are sorted into categories to allow the appropriate teams to respond.
The data is collected from a csv file and is saved in an sqlite database after cleaning. 
The model is saved in a pickle file for use in the web app. 
A Flask web app provides visualizations to better understand the data. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # categories data to process 
|- disaster_messages.csv  # messages data to process
|- process_data.py  # python file that processes the data and prepares it for machine learning
|- DisasterResponse.db   # database where cleaned data is saved

- models
|- train_classifier.py  # python file that builds the machine learning model to classify the messages
|- classifier.pkl  # saved classification model 

- README.md  # This file
