from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
#import h2o
#from h2o.automl import H2OAutoML
from sklearn.pipeline import Pipeline
import subprocess


# define a Gaussain NB classifier

# define the class encodings and reverse encodings
classes = {0: 'Bad', 1: 'Good'}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    # X, y = datasets.load_iris(return_X_y=True)
    df = pd.read_csv('dataset/german.data', sep=' ', header=None)
    y = df[20]
    X = df.drop(20, axis=1)

    # one hot encode categorical features only 
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    ct = ColumnTransformer([('o',OneHotEncoder(),cat_ix)], remainder='passthrough')

    clf = GaussianNB()
    
    global pipe
    pipe = Pipeline([
        ("ct", ct),
        ("clf", clf)
    ])

    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    pipe.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"Model trained with accuracy: {round(acc, 3)}")

    #Generating Explainability File
    subprocess.call(["jupyter","nbconvert","--to","notebook","--inplace","--execute","dataset/explainable_AI_starter.ipynb"])
    subprocess.call(["jupyter","nbconvert","dataset/explainable_AI_starter.ipynb","--no-input","--to","html"])
    print("Explainability file generated")

# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    X = pd.DataFrame([x])

    global pipe
    prediction = pipe.predict(X)[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# unction to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.loan] for d in data]

    # fit the classifier again based on the new data obtained
    global pipes
    pipe.fit(X, y)