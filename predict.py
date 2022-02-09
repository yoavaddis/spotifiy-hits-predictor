# Made by @yyoavv

import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

# Load model
model = keras.models.load_model('path')

print(model.summary())

# Load test data
data = pd.read_csv('spotify.csv')

# Select coloumns to use
x_test = data.dropna()
y_test = x_test.pop('track_popularity')

artists = x['track_artist']

# Collect all numaric features
num_features =  ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']

# Collect all categorical features
cat_features = ['playlist_genre']

# Apply StandardScaler on num_features and OneHotEncoder on cat_features
preprocessor = make_column_transformer((StandardScaler(),num_features),(OneHotEncoder(),cat_features))

x_test = pd.DataFrame(preprocessor.fit_transform(x_test))
y_test = y_test / 100

# Print predictions
print(model.predict(x_test))

# Print accuracy of the model (by using test sets)
accuracy = model.score(x_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 3)},%')
