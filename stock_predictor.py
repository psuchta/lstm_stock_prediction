import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from time import time
import tensorflow as tf


class StockPredictor:

    def __init__(self, batch_size=10, back_step_size=30):
        self.batch_size = batch_size
        self.back_step_size = back_step_size
        self.neuron_units = 50
        self.dropout_factor = 0.7
        self.network_model = None
        self.eps = 2
        self.optimizer = 'adam'
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.07, momentum=0.0, nesterov=False, name="SGD")

        self.last_prediction_time = None

    def create_learned_model(self, training_data):
        x_train, y_train = self.get_learning_data(training_data)

        start_time = time()
        
        self.set_model(x_train)
        self.learn_model(x_train, y_train)
        
        end_time = time()
        self.last_prediction_time = round(end_time - start_time, 2)

    # Returns two arrays:
    #   * x_data - array which contains the 10 previous days for the day on which we want to train the model 
    #   * y_data - array which contains stock value for each value we want to train from
    def get_learning_data(self, data):
        x_data = []
        y_data = []
        for i in range(self.back_step_size, len(data)):
            # Here for i day observation we are getting 10 (back_step_size) past observations.
            x_data.append(data[i - self.back_step_size:i, 0])
            # We add current i stock value to y array
            y_data.append(data[i, 0])
        # x_data has following structure [array([0.97, 0.92473118]), array([1., 0.96])]
        x_data, y_data = np.array(x_data), np.array(y_data)
        # x_data has following shape - (116, 10)
        # line billow adds each observation to one element array
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
        return x_data, y_data

    def set_model(self, x_train):
        model = Sequential()
        model.add(LSTM(units=self.neuron_units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(rate=self.dropout_factor))
        model.add(LSTM(units=self.neuron_units, return_sequences=True))
        model.add(Dropout(rate=self.dropout_factor))
        model.add(LSTM(units=self.neuron_units, return_sequences=True))
        model.add(Dropout(rate=self.dropout_factor))
        model.add(LSTM(units=self.neuron_units, return_sequences=True))
        model.add(Dropout(rate=self.dropout_factor))
        model.add(LSTM(units=self.neuron_units, return_sequences=True))
        model.add(Dropout(rate=self.dropout_factor))
        model.add(LSTM(units=self.neuron_units, return_sequences=False))
        model.add(Dropout(rate=self.dropout_factor))
        model.add(Dense(units=1))
        self.network_model = model

    def learn_model(self, x_train, y_train):
        self.network_model.compile(optimizer=self.optimizer, loss='mean_squared_error')
        self.network_model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.eps)

    def predict_values(self, data_set):
        x_set, y_set = self.get_learning_data(data_set)
        prediction = self.network_model.predict(x_set)

        return prediction, y_set


    def get_description(self):
        return f'''
               LiveChat stock test prediction within {self.back_step_size} time steps with {self.eps} epochs
               with {self.neuron_units} neurons with elapsed time {self.last_prediction_time} seconds,
               dropout {self.dropout_factor}
           '''
