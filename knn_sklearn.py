import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from load_data import load_data

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)

    filename = 'models/knn_sklearn_model.sav'
    pickle.dump(knn, open(filename, 'wb'))
    print("Model saved to disk")

    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
