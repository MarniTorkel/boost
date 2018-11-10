import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import time, warnings

# If MODE = 1 then the classifier is trained and evaluated with optimal parameters
# If MODE = 2 then the experiment corresponding to EXPERIMENT_NUM is run
MODE = 1

# Determines the experiment to run if MODE = 2
EXPERIMENT_NUM = 5

# Determines the number of jobs to run in parallel when performing the grid search
NUM_THREADS = 1

def get_experiment_1():
    classifier = AdaBoostClassifier(algorithm="SAMME", learning_rate=0.1, random_state=5)
    param_grid = {
        'n_estimators': [100, 200, 300]
    }
    return (classifier, param_grid)

def get_experiment_2():
    classifier = DecisionTreeClassifier()
    param_grid = {
        'max_depth': [5, 10, 15],
        'criterion': ['entropy', 'gini'],
        'min_samples_leaf': [5, 10, 15],
        'max_leaf_nodes': [20, 30, 40]
    }
    return (classifier, param_grid)

def get_experiment_3():
    dt = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=20,min_samples_leaf=5,random_state=5)
    classifier = AdaBoostClassifier(base_estimator=dt,n_estimators=500,algorithm='SAMME',learning_rate=0.1,random_state=5)
    param_grid = {
        'base_estimator__max_depth': [5,7]
    }
    return (classifier, param_grid)

def get_experiment_4():
    classifier = ExtraTreeClassifier(random_state=5)
    param_grid = {
        'max_depth': [20,25,30],
        'criterion': ['entropy', 'gini'],
        'min_samples_leaf': [1,2,3],
        'max_leaf_nodes': [1200, 1500, 2000]
    }
    return (classifier, param_grid)

def get_experiment_5():
    et = ExtraTreeClassifier(criterion='entropy',max_leaf_nodes=1200,min_samples_leaf=1,random_state=5)
    classifier = AdaBoostClassifier(base_estimator=et,n_estimators=500,algorithm='SAMME',learning_rate=0.1,random_state=5)
    param_grid = {
        'base_estimator__max_depth': [5,7]
    }
    return (classifier, param_grid)

def get_experiment(returnParamGrid=True):
    if EXPERIMENT_NUM == 1:
        classifier, param_grid = get_experiment_1()
    elif EXPERIMENT_NUM == 2:
        classifier, param_grid = get_experiment_2()
    elif EXPERIMENT_NUM == 3:
        classifier, param_grid = get_experiment_3()
    elif EXPERIMENT_NUM == 4:
        classifier, param_grid = get_experiment_4()
    elif EXPERIMENT_NUM == 5:
        classifier, param_grid = get_experiment_5()

    if returnParamGrid:
        return (classifier, param_grid)
    else:
        return classifier

# Parameter tuning
def get_optimal_parameters(features, classes):
    # Instantiate the classifier
    classifier, param_grid = get_experiment()

    # Instantiate the grid search and fit the data to perform the search
    gridSearch = GridSearchCV(classifier, param_grid, cv=5, scoring="f1_weighted", n_jobs=NUM_THREADS)
    gridSearch.fit(features, classes)

    # Output the score results of the grid search
    for params, mean_score, scores in gridSearch.grid_scores_:
        print("Average F1 Score {:0.3f} for {}".format(mean_score, params))

    # Return the optimal parameters
    return gridSearch.best_params_


def evaluate_tuned_classifier(features, classes, optimal_parameters):
    # Obtain the classifier for the current experiment
    classifier = get_experiment(returnParamGrid=False)

    classifier.set_params(**optimal_parameters)

    # Split the data set into training and test set
    train_X, test_X, train_Y, test_Y = train_test_split(features, classes, test_size=0.2)

    # Fit the training data to the model
    classifier.fit(train_X, train_Y)

    # Predict the classes of the test set
    predicted_classes = classifier.predict(test_X)

    # Compute the confusion matrix
    matrix = confusion_matrix(test_Y, predicted_classes)
    print("Confusion matrix:")
    print(matrix)

    # Compute the weighted F1-Score
    f1_measure = f1_score(test_Y, predicted_classes, average="weighted")
    print("Weighted F1-Score: {:0.3f}".format(f1_measure))

    # Compute a classification report
    report = classification_report(test_Y, predicted_classes)
    print("Classification report:")
    print(report)

def evaluate_optimal_classifier(features, classes):
    # Obtain the classifier for the current experiment
    et = ExtraTreeClassifier(criterion='entropy',max_leaf_nodes=1200,min_samples_leaf=1,max_depth=7,random_state=5)
    classifier = AdaBoostClassifier(base_estimator=et,n_estimators=500,algorithm='SAMME',learning_rate=0.1,random_state=5)

    # Split the data set into training and test set
    train_X, test_X, train_Y, test_Y = train_test_split(features, classes, test_size=0.2)

    # Fit the training data to the model
    classifier.fit(train_X, train_Y)

    # Predict the classes of the test set
    predicted_classes = classifier.predict(test_X)

    # Compute the confusion matrix
    matrix = confusion_matrix(test_Y, predicted_classes)
    print("Confusion matrix:")
    print(matrix)

    # Compute the weighted F1-Score
    f1_measure = f1_score(test_Y, predicted_classes, average="weighted")
    print("Weighted F1-Score: {:0.3f}".format(f1_measure))

    # Compute a classification report
    report = classification_report(test_Y, predicted_classes)
    print("Classification report:")
    print(report)

# Main function
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    start = time.time()

    # Read in the CSV data
    data = pd.read_csv("../input/winequality-white.csv", delimiter=";").values

    # Slice the data to separate the features from the classes
    features = data[:,:11]
    classes = data[:,11]

    sc = StandardScaler()
    sc.fit(features)
    features = sc.transform(features)

    if MODE == 1:
        evaluate_optimal_classifier(features, classes)
        end = time.time()
        print("Training & evaluation time = {}".format(end-start))
    else:
        # Get the optimal parameters for the classifier
        optimal_parameters = get_optimal_parameters(features, classes)
        print("Optimal parameters are: {}".format(optimal_parameters))

        mid = time.time()
        print("Tuning time = {}".format(mid-start))

        # Evaluate the classifier with the tuned parameters & output results
        evaluate_tuned_classifier(features, classes, optimal_parameters)

        end = time.time()
        print("Training & evaluation time = {}".format(end-mid))
        print("Entire time = {}".format(end-start))







