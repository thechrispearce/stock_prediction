import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# import data
goog = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/GOOG_2010.csv")
nflx = pd.read_csv("/Users/chrispearce/Documents/Python Mini Projects/Datasets/NFLX_2010.csv")
goog['Date'] = pd.to_datetime(goog.Date)
nflx['Date'] = pd.to_datetime(nflx.Date)

df = pd.DataFrame()
df['Date'] = goog['Date']
df['Predictor_1'] = goog['Close']
df['Predictor_2'] = nflx['Close']
df['Target'] = goog['Close']
# Line to amend date range
#df = df[df['Date'] > '2019-01-01'].reset_index()

class RNN:
    # target as 'Target' and predictors as ['Predictor_1', 'Predictor_2',...]
    def __init__(self, data, target, predictors):
        self.data = data
        self.target_lab = target
        self.pred_labs = predictors
        self.target_data = data.filter([target])
        self.input_data = data.filter(predictors)

    def preprocess(self, train_perc, valid_perc, timesteps):
        self.ts = timesteps
        self.train_perc = train_perc
        self.valid_perc = valid_perc
        self.target_vals = self.target_data.values
        self.input_vals = self.input_data.values
        # Scale data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.input_scaled = self.scaler.fit_transform(self.input_vals)
        self.target_scaled = self.scaler.fit_transform(self.target_vals)

        # split training data into x and y
        # x will be data used to predict values
        # y will be actual values, used to optimise model
        x = []
        y = []
        # in x we want the previous 'ts' values for each data point
        # in y we want the actual value
        for i in range(self.ts, len(self.input_scaled)):
            x.append(self.input_scaled[i - self.ts:i, 0])
            y.append(self.target_scaled[i, 0])
            # ,0 is to make sure we add the value, not the array
        # Convert x and y to numpy arrays
        x, y = np.array(x), np.array(y)
        # Reshape the data: the model expects 3D input: samples, timesteps, features
        # Samples is no. of previous vals
        # Timesteps is no. of vals to be predicted
        # Features is no. of variables (1 for TSF)

        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        # Define intervals for training, validation and testing
        self.train_data_len = math.ceil(len(y) * self.train_perc)
        self.validation_data_len = math.ceil(len(y) * (self.valid_perc + self.train_perc))

        # Partition data
        self.x_train = x[0:self.train_data_len, :]
        self.y_train = y[0:self.train_data_len]
        self.x_valid = x[self.train_data_len : self.validation_data_len, :]
        self.y_valid = y[self.train_data_len : self.validation_data_len]
        self.x_test = x[self.validation_data_len :, :]
        self.y_test = y[self.validation_data_len :]

    def model(self, epochs):
        self.epochs = epochs

        # Create model
        self.model = Sequential()
        # LSTM layer
        self.model.add(LSTM(20, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        # LSTM layer
        self.model.add(LSTM(20, return_sequences=False))
        # Regular layer
        self.model.add(Dense(25))
        # Output node
        self.model.add(Dense(1))
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        self.history = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_valid, self.y_valid), batch_size=1, epochs=self.epochs)

        return self.history.history

    def pred(self):

        # Get model predictions
        self.pred = self.model.predict(self.x_test)
        self.pred = self.scaler.inverse_transform(self.pred)

        # Remember we are now working with the original data which is ts elements longer than the preprocessed data
        # Index on the length plus timesteps, otherwwise indexing error
        self.train = self.target_data[:self.train_data_len + self.ts]
        self.valid = self.target_data[self.train_data_len + self.ts:self.validation_data_len + self.ts]
        self.actual = self.target_data[self.validation_data_len + self.ts:]
        self.actual['Predictions'] = self.pred

        return self.actual['Predictions'].corr(self.actual[self.target_lab]), np.sqrt(mean_squared_error(self.actual[self.target_lab], self.actual['Predictions']))

    def plot(self):

        plt.figure(figsize=(10, 6))
        plt.xlabel('Date')
        plt.ylabel('Close ($)')
        # Plot training and validation
        plt.plot(self.train[self.target_lab])
        plt.plot(self.valid[self.target_lab])
        # Plot target on top of predictions
        plt.plot(self.actual[[self.target_lab, 'Predictions']])
        plt.legend(['Training Actual', 'Validation Actual', 'Actual', 'Predicted'])
        plt.show()

        # Return complete data for reference
        return self.train, self.valid, self.actual

    def analytics(self):
        history = self.history.history

        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(history['loss'], lw=2)
        plt.plot(history['val_loss'], lw=2)
        plt.legend(['Train loss', 'Validation loss'], fontsize=8)
        ax.set_xlabel('Epochs', size = 8)

        ax = fig.add_subplot(1, 2, 2)
        plt.xlabel('Date')
        plt.ylabel('Close ($)')
        # Plot training and validation
        plt.plot(self.train[self.target_lab])
        plt.plot(self.valid[self.target_lab])
        # Plot target on top of predictions
        plt.plot(self.actual[[self.target_lab, 'Predictions']])
        plt.legend(['Training Actual', 'Validation Actual', 'Actual', 'Predicted'])

        plt.show()

RNN = RNN(df, 'Target', ['Predictor_1','Predictor_2'])
RNN.preprocess(0.5, 0.3, 5)
RNN.model(5)
print(RNN.pred())
RNN.plot()
RNN.analytics()