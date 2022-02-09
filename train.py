# Made by @yyoavv
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

# Load data
data = pd.read_csv('spotify.csv')

# Select coloumns to use
x = data.dropna()
y = x.pop('track_popularity')

artists = x['track_artist']

# Collect all numaric features
num_features =  ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']

# Collect all categorical features
cat_features = ['playlist_genre']

# Apply StandardScaler on num_features and OneHotEncoder on cat_features
preprocessor = make_column_transformer((StandardScaler(),num_features),(OneHotEncoder(),cat_features))

def group_split (x,y, group, train_size = 0.75):
    # Functionality: split the data and keeps all of an artist's songs in the same split to prevent leakage.
    # Input: x - features, y - target, train_size - ratio of split.
    # Output: x_train, y_train, x_valid, y_valid
    splitter = GroupShuffleSplit(train_size = train_size)
    train,valid = next(splitter.split(x,y,groups = group))
    return (x.iloc[train], y.iloc[train], x.iloc[valid], y.iloc[valid])

x_train, y_train, x_valid, y_valid = group_split(x,y, artists)

x_train = pd.DataFrame(preprocessor.fit_transform(x_train))
x_valid = pd.DataFrame(preprocessor.transform(x_valid))

# The features values are between 0 to 1, changing the target to 0 to 1 (instead of 0 - 100)
y_train = y_train / 100
y_valid = y_valid / 100

x_shape = [x_train.shape[1]]

# Create model - ANN algorithm
model = keras.Sequential([
    layers.Dense(256, activation = 'relu', input_shape = x_shape),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(1)
])

# Adding optimizer and loss function with compile method
model.compile(optimizer = 'adam', loss = 'mae')

# minimium amount of change to count as an improvement, how many epochs to wait before stopping
early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=5, restore_best_weights=True)

# Train model, save training loss and validation loss of each batch using history method
history = model.fit(x_train, y_train, validation_data = (x_valid, y_valid),
                    batch_size = 512, epochs = 50, callbacks=[early_stopping])

# Plotting training loss and validation loss
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
plt.show()

# Print model's summary
print(model.summary())

# Save model
model.save('path')
