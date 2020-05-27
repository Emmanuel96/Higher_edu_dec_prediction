# Higher_edu_dec_prediction

Classification Models To Predict The Decision Of A Student To go On For Higher Education

## Objective

This project focuses on the use of 3 different algorithms (Multi-layered Perceptron Classifier, K-Nearest Neighbor and Adaboost Ensemble) to predict a students decision to go on to Higher Education

## Requirements

1. Python 3.7 or any working version

2. VS Code Or Spyder

## Implementation

Once our environment is set up, we first import our Panda Libraries as follow:

    import pandas as pd
    import numpy as np
    import seaborn as sns

Next, we import the necessary scikit-learn libraries:

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

### Classification

Firstly, we read our csv file,create a data frame out of it and drop our null values as follows:

    data = pd.read_csv(
        r'C:/Users/Emmanuel/Documents/projects/Python/Students Data Analysis/Dataset/student.csv')
    student_data = pd.DataFrame(data)

    # drop all null data
    student_data.dropna(inplace=True)

#### Handle Categorical Values

Classification algorithms don't work well with categorical values, hence we need to convert them to numeric binary values. In this case we use Pandas dummy_data function with our custom made function as follows:

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

    # drop all null data
    student_data.dropna(inplace=True)

    # array of categorical values
    cat_data = ['school', 'sex', 'address', 'famsize', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'activities', 'nursery', 'fatherd', 'Pstatus', 'higher', 'internet', 'romantic', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'Medu', 'famsup']

    student_data = handle_cat_data(cat_data, student_data)

#### Split Dataset To Test and Train Data

We can't use our entire dataset directly for both training and testing, hence we split it into 80% for training and 20% for testing with Pythons train_test_split method. We select our higher column as our target column and leave the other columns, except the target column to train our model with. We achieve this with the code snippet below:

    # divide data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(student_data.drop(
    'failures', axis=1), student_data.failures, test_size=0.25, stratify=student_data.failures)

    # classifier names for run model function
    classifier_names = ["K Nearest Neighbour",
    "Neural Networks", "Adaboost Classifier"]

    # classifiers
    classifiers = [
    KNeighborsClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    ]

Finally we run our model by calling the run_models function as shown below:

    # run classification model
    run_models(classifier_names, classifiers, X_train, X_test, y_train, y_test)

## Results

The table below shows the accuracy of the different algorithms used. These values may vary slightly from what you get based on different factors i.e. your machine.

| Classifier         | Accuracy |
| ------------------ | -------- |
| K-Nearest Neighbor | 82%      |
| MLP                | 80%      |
| Adaboost           | 80%      |
