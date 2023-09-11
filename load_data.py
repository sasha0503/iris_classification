import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    file_path = 'IRIS.csv'
    iris = pd.read_csv(file_path)

    x = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values

    y = np.where(y == 'Iris-setosa', 0, y)
    y = np.where(y == 'Iris-versicolor', 1, y)
    y = np.where(y == 'Iris-virginica', 2, y)

    y = y.astype('int')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test
