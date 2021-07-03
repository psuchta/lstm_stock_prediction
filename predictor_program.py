import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from stock_predictor import StockPredictor

TRAIN_SET_PERCENT = 0.8
X_LABEL = 'Time in days'
Y_LABEL = 'Close price of stock'


def load_file(file_name, columns):
    file_data = pd.read_csv(file_name,
                            index_col='Data',
                            squeeze=True,
                            usecols=columns,
                            parse_dates=True)
    return file_data.iloc[::-1]

def split_to_train_and_test_sets(data):
    slice_range = round(len(data) * TRAIN_SET_PERCENT)
    train_set = data.iloc[:slice_range]
    test_set = data.iloc[slice_range:]
    return train_set, test_set

# Returns scaled data with range (0 - 1)
def scale_data(file_data, sc):
    return sc.fit_transform(np.array(file_data).reshape(-1, 1))


def prepare_for_plot(data_set, scaler):
    data_set = data_set.reshape(-1, 1)
    return scaler.inverse_transform(data_set)


def draw_predicted_set(real_set, predicted_set, sc):
    plt.plot(prepare_for_plot(predicted_set, sc), color='red', label='Prediction')
    plt.plot(prepare_for_plot(real_set, sc), color='blue', label='Real values')
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.legend()
    plt.title(stock_predictor.get_description())
    plt.show()

# def predict_for_10_days(stock_data):


if __name__ == '__main__':
    # Convert CSV data into DataFrames
    file_data = load_file('stock_year.csv',
                          ['Data', 'Zamknięcie'])
    print('original size', len(file_data))
    # Split Data into training and test subsets
    training_set, test_set = split_to_train_and_test_sets(file_data)

    sc = MinMaxScaler(feature_range=(0, 1))
    
    scaled_training_data = scale_data(training_set, sc)
    scaled_test_data = scale_data(test_set, sc)
    # scaled_training_data has following shape - (126, 1), structure - [[0.53], [0.73]..[0.98]]

    # Create and learn model
    stock_predictor = StockPredictor()
    stock_predictor.create_learned_model(scaled_training_data)

    # Predict for test set and Draw plot
    # predicted_test, y_predict_set = stock_predictor.predict_values(scaled_test_data)
    # draw_predicted_set(y_predict_set, predicted_test, sc)

    # Predict for 10 days
    x = scale_data(file_data, sc)
    full_data = np.copy(x)
    # print(full_data)
    for i in range(0, 30):
        additional = np.array([[0]])
        temp_dataset = np.concatenate((full_data, additional))
        x_predicted, y_predicted = stock_predictor.predict_values(temp_dataset)
        full_data = np.append(full_data, [x_predicted[-1]], axis=0)

    draw_predicted_set(x, full_data, sc)