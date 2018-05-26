from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

reader = pd.read_json('~/Downloads/reviews_Electronics_5.json.gz', lines=True, chunksize=20000)

for val in reader:
    break

#reader.chunksize = 100

hv = HashingVectorizer(analyzer='word', ngram_range=(1,2), n_features=2**18)
clf = SGDClassifier(loss='hinge', max_iter=1)

X_val = hv.transform(val['reviewText'])

ci = 0
for chunk in reader:
    print('chunk {}'.format(ci))
    ci+=1
    X = hv.transform(chunk['reviewText'])
    y = chunk['overall']
    clf.partial_fit(X, y, classes=[1, 2, 3, 4, 5])
    val_p = clf.predict(X_val)
    report = classification_report(val['overall'], val_p)
    print(report)
    print(confusion_matrix(val['overall'], val_p))

