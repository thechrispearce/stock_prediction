import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical

# import data
goog = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/GOOG_2010.csv")
nflx = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/NFLX_2010.csv")
goog['Date'] = pd.to_datetime(goog.Date)
nflx['Date'] = pd.to_datetime(nflx.Date)
goog['Target_diff'] = goog['Close'].diff()
goog['Target_tic'] = goog['Target_diff'].apply(lambda row: 1 if row > 0 else 0)

df = pd.DataFrame()
df['Date'] = goog['Date']
df['Predictor'] = goog['Close']
#df['Predictor'] = goog['Target_diff']
#df['Predictor'] = goog['Target_tic']
df['Target'] = goog['Target_tic']
# Line to amend date range
#df = df[df['Date'] > '2019-01-01'].reset_index()

class MLP:
    # target as 'Target' and predictor as 'Predictor'
    def __init__(self, data, target, predictor):
        self.data = data
        self.target_lab = target
        self.pred_labs = predictor
        self.target_data = data.filter([target])
        self.input_data = data.filter([predictor])

    def preprocess(self, train_perc, valid_perc, timesteps):
        self.ts = timesteps
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.target_vals = self.target_data.values
        self.input_vals = self.input_data.values

        # split training data into x and y
        # x will be data used to predict values
        # y will be actual values, used to optimise model
        x = []
        y = []
        # in x we want the previous 'ts' values for each data point
        # in y we want the actual value
        for i in range(self.ts, len(self.target_vals)):
            x.append(self.input_vals[i - self.ts:i])
            y.append(self.target_vals[i])
            # ,0 is to make sure we add the value, not the array
        # Convert x and y to numpy arrays
        x, y = np.array(x), np.array(y)
        x = np.reshape(x, (x.shape[0], x.shape[1]))
        y = to_categorical(y)

        # Define intervals for training, validation and testing
        self.train_data_len = math.ceil(len(y) * self.train_perc)
        self.validation_data_len = math.ceil(len(y) * (self.valid_perc + self.train_perc))

        # Partition data
        self.x_train = x[:self.train_data_len, :]
        self.y_train = y[:self.train_data_len]
        self.x_valid = x[self.train_data_len : self.validation_data_len, :]
        self.y_valid = y[self.train_data_len : self.validation_data_len]
        self.x_test = x[self.validation_data_len :, :]
        self.y_test = y[self.validation_data_len :]

        return self.x_test.shape, self.y_test.shape, self.y_train[0:5]

    def model(self, epochs):
        self.epochs = epochs

        # Create model
        self.model = Sequential()
        # Regular layer
        self.model.add(Dense(25, activation = 'sigmoid', input_dim = self.ts))
        # Regular layer
        self.model.add(Dense(50, activation='sigmoid'))
        # Dropout layer
        self.model.add(Dropout(rate = 0.5))
        # Regular layer
        self.model.add(Dense(50, activation='sigmoid'))
        # Dropout layer
        self.model.add(Dropout(rate=0.5))
        # Output node
        self.model.add(Dense(2, activation = 'softmax'))
        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
        # Train the model
        self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid), batch_size=1, epochs=self.epochs)

        return self.history.history

    def pred(self):

        # Evaluate model
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=1)
        # Get model predictions
        self.pred = self.model.predict(self.x_test)

        # Remember we are now working with the original data which is ts elements longer than the preprocessed data
        # Index on the length plus timesteps, otherwise indexing error
        self.train = self.target_data[:self.train_data_len + self.ts]
        self.valid = self.target_data[self.train_data_len + self.ts:self.validation_data_len + self.ts]
        self.actual = self.target_data[self.validation_data_len + self.ts:].reset_index()
        self.actual['Predictions'] = [0 if self.pred[i, 0] > self.pred[i, 1] else 1 for i in range(len(self.pred))]

        return 'Test loss.: {:.4f}  Test acc.:{:.4f}'.format(*results)

    def analytics(self):
        history = self.history.history

        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(history['loss'], lw=2)
        plt.plot(history['val_loss'], lw=2)
        plt.legend(['Train loss', 'Validation loss'], fontsize=8)
        ax.set_xlabel('Epochs', size = 8)

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(history['accuracy'], lw=2)
        plt.plot(history['val_accuracy'], lw=2)
        plt.legend(['Train accuracy', 'Validation accuracy'], fontsize=8)
        ax.set_xlabel('Epochs', size=8)

        plt.show()

MLP = MLP(df, 'Target', 'Predictor')
MLP.preprocess(0.5, 0.3, 20)
MLP.model(20)
print(MLP.pred())
MLP.analytics()