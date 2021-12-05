import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from naive_gaussian_bayes import NaiveBayesGaussian

df = pd.read_csv('winequality-white.csv', sep=';').reset_index().dropna()


def get_features_targets(dataframe):
    return dataframe.drop(['quality'], axis=1), dataframe['quality']


def train_test_sets_method():
    bayes = NaiveBayesGaussian()
    features, targets = get_features_targets(df)
    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=0.4, random_state=54)
    bayes.fit(features_train, targets_train)
    predictions = bayes.predict(features_test)
    accuracy = bayes.accuracy(targets_test, predictions)
    print("Train & test sets model accuracy: %.4f" % accuracy)


def cross_validation_method(k=5):
    bayes = NaiveBayesGaussian()
    accuracies = []
    shuffled_df = df.sample(frac=1, random_state=50)
    slice_size = int(len(shuffled_df) / k)
    for i in range(k):
        df_test_slice = shuffled_df[i * slice_size:(i + 1) * slice_size]
        df_train_slice = shuffled_df.drop(df_test_slice.index)
        features_test, targets_test = get_features_targets(df_test_slice)
        features_train, targets_train = get_features_targets(df_train_slice)
        bayes.fit(features_train, targets_train)
        predictions = bayes.predict(features_test)
        accuracies.append(bayes.accuracy(targets_test, predictions))

    print("Cross validation accuracies per iteration:")
    for index, acc in enumerate(accuracies):
        print(f"{index}: %.4f" % acc)
    print("Cross validation average accuracy: %.4f" % np.mean(accuracies))


# train_test_sets_method()
# cross_validation_method(k=5)
# print(df['quality'].value_counts())