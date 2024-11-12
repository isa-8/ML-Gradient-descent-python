import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Mean Squared Error (MSE) and Mean Absolute Error (MAE) calculation
def compute_loss(X, y, weights, loss_type="MSE"):
    pred = np.dot(X, weights)
    error = y - pred
    # number of examples
    n = X.shape[0]
    MSE = np.sum(error ** 2) / (2 * n)
    MAE = np.sum(np.abs(error)) / (2 * n)
    return MSE, MAE


def gradientDescent(X_train, y_train, iterations=100, lr= 0.001):
    cur_weights = np.random.randn(X_train.shape[1])  # Random initialization

    for i in range(iterations):
        gradient = derivative(X_train, y_train, cur_weights)
        cur_weights -= gradient * lr
        if i % 10 == 0:
            MSE, MAE = compute_loss(X_train, y_train, cur_weights)
            print(f"Iteration {i} - MSE/MAE: {MSE}/{MAE}")

    return cur_weights


# computing the gradient for all samples
def derivative(X, y, weights):
    # predicted values
    pred = np.dot(X, weights)
    error = y - pred
    gradient = X.T.dot(error) / X.shape[0]
    return gradient


# Predict on test data
def predict(X, weights):
    return [sum(X[i][j] * weights[j] for j in range(len(weights))) for i in range(len(X))]


def best_lr(X, y):
    learning_rates = [0.1, 0.01, 0.001, 0.05]
    best_lr = 0.01
    cur_cost = 100000000

    for lr in learning_rates:
        weights = gradientDescent(X, y, 100, lr)
        loss = compute_loss(X, y, weights)
        if loss < cur_cost:
            cur_cost = loss
            best_lr = lr

    return best_lr


if __name__ == '__main__':

    # Generate the dataset with 1000 samples and 3 features
    X, y = make_regression(n_samples=1000, n_features=3, noise=10)

    # Split the dataset into training and testing sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate and print MSE and MAE on test data
    # y_pred_test = predict(X_test, weights)
    # mse_test = compute_loss(X_test, y_test, weights, loss_type="MSE")
    # mae_test = compute_loss(X_test, y_test, weights, loss_type="MAE")
    # print(f"Test MSE: {mse_test:.4f}")
    # print(f"Test MAE: {mae_test:.4f}")

    optimized_weights = gradientDescent(X_train, y_train)
    # best_lr = best_lr(X_train, y_train)
    test_loss = compute_loss(X_test, y_test, optimized_weights)
    print(f"test loss = {test_loss}")