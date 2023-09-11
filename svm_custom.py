import copy
import pickle

import numpy as np

from load_data import load_data


class SVM:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.random.randn(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * 1 / self.n_iters * self.w)
                else:
                    self.w -= self.lr * (2 * 1 / self.n_iters * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


class MulticlassSVM:
    def __init__(self, learning_rate=0.01, n_iters=1000, n_classes=3):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.classifiers = [SVM(learning_rate, n_iters) for _ in range(n_classes)]

    def fit(self, X, y):
        for i in range(self.n_classes):
            # Convert the problem to binary classification
            binary_y = np.where(y == i, 1, -1)
            self.classifiers[i].fit(X, binary_y)

    def predict(self, X):
        scores = np.zeros((X.shape[0], self.n_classes))
        for i in range(self.n_classes):
            scores[:, i] = self.classifiers[i].predict(X)
        # Assign the class with the highest score
        return np.argmax(scores, axis=1)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    SVM = MulticlassSVM()
    SVM.fit(x_train, y_train)

    model_path = 'models/svm_custom_model.pkl'
    SVM_save = copy.copy(SVM)
    SVM_save.classifiers = [dict(w=clf.w, b=clf.b) for clf in SVM.classifiers]
    pickle.dump(SVM_save, open(model_path, 'wb'))
    print("Model saved to disk")

    y_pred = SVM.predict(x_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
