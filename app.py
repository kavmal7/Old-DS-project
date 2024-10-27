import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from waitress import serve

app = Flask(__name__)

# We can load our model here
# Path may need to be changed depending on the PC
model = pickle.load(open('/Users/kavya/INF161/project/model.pkl', 'rb'))

# Because of the dummy variables, I have to recreate the same dataframe and pad missing values with 0
cols = ['Dato/Tid', 'Volum', 'Solskinstid', 'Lufttemperatur', 'Vindstyrke',
       'Måned', 'Dag', 'Ukedag', 'Time', 'Måned_1', 'Måned_10', 'Måned_11',
       'Måned_12', 'Måned_2', 'Måned_3', 'Måned_4', 'Måned_5', 'Måned_6',
       'Måned_7', 'Måned_8', 'Måned_9', 'Ukedag_0', 'Ukedag_1', 'Ukedag_2',
       'Ukedag_3', 'Ukedag_4', 'Ukedag_5', 'Ukedag_6', 'Time_0', 'Time_1',
       'Time_10', 'Time_11', 'Time_12', 'Time_13', 'Time_14', 'Time_15',
       'Time_16', 'Time_17', 'Time_18', 'Time_19', 'Time_2', 'Time_20',
       'Time_21', 'Time_22', 'Time_23', 'Time_3', 'Time_4', 'Time_5', 'Time_6',
       'Time_7', 'Time_8', 'Time_9']


df_dict = {}

for c in cols:
    df_dict[c] = np.nan

# Pad empty dataframe using the columns above with NaN
df_all_columns = pd.DataFrame([df_dict])

@app.route('/')
def home():
    return render_template('./index_bikes.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = dict(request.form)
    print(features)

    # Get dummy values based on the given input
    all_features = ['Solskinstid', 'Lufttemperatur', 'Vindstyrke', 'Måned', 'Dag', 'Ukedag', 'Time']

    features_df = pd.DataFrame(features, index=[0]).loc[:, all_features]
    features_dummies = pd.get_dummies(features_df[['Måned', 'Ukedag', 'Time']].astype(str))
    features_df = pd.concat([features_df, features_dummies], axis=1)

    global df_all_columns

    # Merge columns and fill all NaN with 0s
    df_all_columns.update(features_df)
    df_all_columns = df_all_columns.fillna(0)

    print(df_all_columns)

    # Edge cases for impossible values
    if float(features['Solskinstid']) < 0 or float(features['Solskinstid']) > 60:
        return render_template('./index_bikes.html', prediction_text='The sunshine time has to be between 0 and 60')
    
    if float(features['Vindstyrke']) < 0:
        return render_template('./index_bikes.html', prediction_text='The wind speed cannot be negative')

    if float(features['Måned']) < 1 or float(features['Måned']) > 12:
        return render_template('./index_bikes.html', prediction_text='The month has to be between 1 and 12')

    if float(features['Dag']) < 1 or float(features['Dag']) > 31:
        return render_template('./index_bikes.html', prediction_text='The day has to be between 1 and 31 (max)')

    if float(features['Ukedag']) < 1 or float(features['Ukedag']) > 7:
        return render_template('./index_bikes.html', prediction_text='The weekday has to be between 1 and 7')
    
    if float(features['Time']) < 0 or float(features['Time']) > 24:
        return render_template('./index_bikes.html', prediction_text='The time has to be between 0 and 24')

    # Predicting volum
    prediction = model.predict(df_all_columns.drop(['Dato/Tid', 'Volum'], axis=1).to_numpy())
    prediction = np.round(prediction[0])

    # Make sure that negative values are filtered to 0
    if prediction < 0:
        prediction = 0
    
    prediction = np.clip(prediction, 0, np.inf)

    # Reset the dataframe so that it can work recursively
    df_all_columns = pd.DataFrame([df_dict])

    # Return the predicted value
    return render_template('./index_bikes.html', prediction_text=f'Predicted count: {int(prediction)}')


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)