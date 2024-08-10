import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import os

def load_data():
    # Load datasets from files
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    return X_train, X_test, y_train, y_test


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train model
    rfc = RandomForestClassifier()

    #Finding the best params from the random forest (modifable params here)
    rand_forest_grid = {
        'bootstrap': [True, False],
        'max_depth': [50, 100, 150],
        'n_estimators': [200, 300, 400]
    }

    rand_forest_gridsearch = GridSearchCV(estimator = rfc, 
                                        param_grid = rand_forest_grid, 
                                        cv = 3, 
                                        n_jobs = -1, 
                                        verbose = 1)
    
    # Fitting the model
    rand_forest_gridsearch.fit(X_train, y_train)

    #print the best params to take note
    print(rand_forest_gridsearch.best_params_)

    # dumping the model
    filename = 'random_forest_model.pkl'
    saved_model_dir = 'saved_model'

    # Create the saved_model directory if it does not exist
    os.makedirs(saved_model_dir, exist_ok=True)

    # Define the filepath to save the pickled model
    model_filepath = os.path.join(saved_model_dir, filename)

    # Pickle dump the model into the saved_model directory
    with open(model_filepath, 'wb') as f:
        pickle.dump(rand_forest_gridsearch, f)

    # compare/evaluate results
    y_pred = rand_forest_gridsearch.predict(X_test)
    class_labels = sorted(y_test['AdoptionSpeed'].unique().astype(str))
    print(classification_report(y_test, y_pred, target_names=class_labels))

    

if __name__ == "__main__":
    main()
