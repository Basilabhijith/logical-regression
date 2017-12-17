import pandas as pd
df = pd.read_csv('titanic.csv')
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include]
categoricals = []

for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)
from sklearn.ensemble import RandomForestClassifier as rf

dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])
y = df_ohe[dependent_variable]

clf = rf()
clf.fit(x, y)
from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')
clf = joblib.load('model.pkl')

from flask import Flask, jsonify
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = clf.predict(query)
     return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     app.run(port=8080)

model_columns = list(x.columns)
joblib.dumps(model_columns, 'model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)

     for col in model_columns:
          if col not in query.columns:
               query[col] = 0

     prediction = clf.predict(query)
     return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     model_columns = joblib.load('model_columns.pkl')
     app.run(port=8080)