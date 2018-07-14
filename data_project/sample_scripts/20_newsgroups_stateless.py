import six; from six.moves import cPickle

import warnings

import numpy as np
import sklearn.datasets
import sklearn.externals.joblib
import sklearn.feature_extraction.text
import sklearn.feature_selection
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

import sys

is_python_3 = sys.version_info[0] >= 3

def report(results, n_top=3):
    print('-' * 42)
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_train_test():
    train = sklearn.datasets.fetch_20newsgroups(subset='train', remove=('headers'))
    test = sklearn.datasets.fetch_20newsgroups(subset='test', remove=('headers'))
    return train, test


def model_selection(train, test, cv=3, test_size=0.33):
    """
    Run model selection with the given data and targets.
    :param data: n-element list of newsgroup posts.
    :param targets: n-element list of newgroups label.
    :param target_names: textual description of newsgroup labels.
    :return: cross validated classification pipeline
    """
    pipeline = sklearn.pipeline.Pipeline(
        [
            ('vectorizer', sklearn.feature_extraction.text.HashingVectorizer()),
            ('clf', sklearn.linear_model.LogisticRegression())
        ]
    )

    parameters = {
        'vectorizer__ngram_range': [(1,1), (1, 2)],
        'vectorizer__n_features': [100000],
        'vectorizer__analyzer': ['word'],
    }

    data_train, data_val, labels_train, labels_val = sklearn.model_selection.train_test_split(
        train.data,
        train.target,
        test_size=test_size,
        random_state=42
    )

    clf = sklearn.model_selection.GridSearchCV(
        pipeline,
        parameters,
        scoring=sklearn.metrics.make_scorer(sklearn.metrics.precision_score, **{'average': 'weighted'}),
        refit=True,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf.fit(data_train, labels_train)

    report(clf.cv_results_)
    best_pipeline = clf.best_estimator_

    #print(sklearn.metrics.classification_report(labels_val, best_pipeline.predict(data_val)))

    return best_pipeline


if __name__ == "__main__":
    train, test = get_train_test()
    best_pipeline = model_selection(train, test, cv=2, test_size=0.4)

    print()
    print('saving model...')
    sklearn.externals.joblib.dump(best_pipeline, 'pipeline.pkl')
    cPickle.dump(train.target_names, open('train_target_names.pkl', 'wb'))
    print('done.')

