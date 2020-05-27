# %matplotlib qt

import pandas as pd
import numpy as np
import seaborn as sns

# sklearn imports
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# miscellaneous
import time


# function to get TP, FP, TN
def perf_measure(y_test, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_test[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_test[i] != y_pred[i]:
            FP += 1
        if y_test[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_test[i] != y_pred[i]:
            FN += 1

    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))


# function to call classification algorithms
def run_models(c_names, classifiers, X_train, X_test, y_train, y_test, save=0, save_index=0, tune=0, grid_parameters=[], feature_selection=0, f_selection_method=0):
    result = np.nan
    for name, clf in zip(c_names, classifiers):
        start_time = time.time()
        feature_sel_txt = ""
        print("Currently on: " + name + '...')
        result = clf.fit(X_train, y_train)

        # get y_pred
        y_pred = result.predict(X_test)

        # print accuracy
        print("Accuracy For " + name + ":" +
              metrics.accuracy_score(y_test, y_pred).astype(str))
        perf_measure(y_test.tolist(), y_pred.tolist())

        # get the time it took to run
        end_time = time.time()
        print(name + " took: " + str(end_time - start_time) + "seconds")
# -- end of run_models function


# function to convert categorical data to dummy data
def handle_cat_data(cat_feats, data):
    for f in cat_feats:
        to_add = pd.get_dummies(data[f], prefix=f, drop_first=True)
        merged_list = data.join(
            to_add, how='left', lsuffix='_left', rsuffix='_right')
        data = merged_list

    # then drop the categorical features
    data.drop(cat_feats, axis=1, inplace=True)

    return data


data = pd.read_csv(
    r'C:/Users/Emmanuel/Documents/projects/Python/Students Data Analysis/Dataset/student.csv')
student_data = pd.DataFrame(data)

# drop all null data
student_data.dropna(inplace=True)

# array of categorical values
cat_data = ['school', 'sex', 'address', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'activities', 'nursery', 'fatherd', 'Pstatus', 'higher', 'internet', 'romantic', 'famrel',
            'freetime', 'goout', 'Dalc', 'Walc', 'health', 'Medu', 'famsup']

student_data = handle_cat_data(cat_data, student_data)

# divide data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(student_data.drop(
    'failures', axis=1), student_data.failures, test_size=0.25, stratify=student_data.failures)

# classification section
classifier_names = ["K Nearest Neighbour",
                    "Neural Networks", "Adaboost Classifier"]

# classifiers
classifiers = [
    KNeighborsClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
]
# run classification model
run_models(classifier_names, classifiers, X_train, X_test, y_train, y_test)
