import six; from six.moves import cPickle

import numpy as np
import sklearn.externals.joblib
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

# parser = reqparse.RequestParser()
# parser.add_argument('task')

pipeline = sklearn.externals.joblib.load('pipeline.pkl')
label_names = cPickle.load(open('train_target_names.pkl', 'rb'))


class NewsgroupService(Resource):
    def post(self):
        text = request.get_json()['post_text']
        probas = pipeline.predict_proba([text])
        labels_with_probas = [{'name': n, 'proba': p} for (p, n) in zip(probas[0], label_names)]
        predicted_newsgroup = sorted(labels_with_probas, key=lambda e: e['proba'])[-1]

        return {
            'predicted_newsgroup': predicted_newsgroup,
            'labels_with_probas': labels_with_probas
        }


api.add_resource(NewsgroupService, '/')

if __name__ == '__main__':
    app.run(debug=True)

