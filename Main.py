import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Mean Squared Error (MSE) and Mean Absolute Error (MAE) calculation
def compute_loss(X, y, weights):
    pred = np.dot(X, weights)
    error = y - pred
    n = X.shape[0]
    MSE = np.sum(error ** 2) / ( 2* n )
    MAE = np.sum(np.abs(error)) / ( 2* n )
    return MSE, MAE


def gradientDescent(X_train, y_train, iterations=100, lr=0.1):
    cur_weights = np.random.randn(X_train.shape[1])  # Random initialization
    for i in range(iterations):
        gradient = derivative(X_train, y_train, cur_weights)
        cur_weights -= gradient * lr
        if i % 10 == 0:

            MSE, MAE = compute_loss(X_train, y_train, cur_weights)
            print(f"Iteration {i} - MSE: {MSE:.4f}, MAE: {MAE:.4f}")
    return cur_weights


# Compute gradient for all samples
def derivative(X, y, weights):
    pred = np.dot(X, weights)
    error = y - pred
    gradient = -X.T.dot(error) / X.shape[0]
    return gradient


# Predict on test data
def predict(X, weights):
    return np.dot(X, weights)


# Find best learning rate
def best_lr(X, y):
    learning_rates = [0.1, 0.01, 0.001, 0.05]
    best_lr = 0.085
    cur_cost = float('inf')
    for lr in learning_rates:
        weights = gradientDescent(X, y, iterations=100, lr=lr)
        mse, _ = compute_loss(X, y, weights)
        if mse < cur_cost:
            cur_cost = mse
            best_lr = lr
    return best_lr


if __name__ == '__main__':


    X, y = make_regression(n_samples=1000, n_features=3, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    optimized_weights = gradientDescent(X_train, y_train)
    print(f"optimized weights = {optimized_weights}")
    # Calculate and print MSE and MAE on test data
    mse_test, mae_test = compute_loss(X_test, y_test, optimized_weights)
    print(f"Test MSE: {mse_test:.4f}")
    print(f"Test MAE: {mae_test:.4f}")