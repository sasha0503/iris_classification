import pickle

import numpy as np

from load_data import load_data


class KNN:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_pred = []
        for x in x_test:
            distances = np.sqrt(np.sum((x - self.x_train)**2, axis=1))

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)

            y_pred.append(most_common)

        return y_pred


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    
    knn = KNN(k=3)
    knn.fit(x_train, y_train)
    
    model_path = 'models/knn_custom_model.sav'
    pickle.dump(knn, open(model_path, 'wb'))
    print("Model saved to disk")

    y_pred = knn.predict(x_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
