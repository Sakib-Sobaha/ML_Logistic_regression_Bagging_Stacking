import numpy as np



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            # print(f"Type of linear_model: {type(linear_model)}") 
            y_predicted = sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls 
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_pred)

        return np.column_stack((1 - probabilities, probabilities))
    
    def predict_log_proba(self, X):
        probabilities = self.predict_proba(X)
        log_probabilities = np.log(probabilities)
        return log_probabilities