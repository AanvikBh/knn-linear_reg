
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

class LinearRegression:
    def __init__(self, reg_lambda, poly_degree, num_features, learning_rate, plot_save, num_iter=100, regularization=None, min_delta = 1e-4):
        self.reg_lambda = reg_lambda  # Regularization parameter
        self.poly_degree = poly_degree  # Degree of polynomial features
        self.num_features = num_features  # Number of features in the dataset
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.num_iterations = num_iter  # Number of iterations for gradient descent
        self.weights = None  # Weights for the model
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.regularization = regularization  # Regularization type: 'L1', 'L2', or None
        self.grad_desc_iterations = list()
        self.plot_save = False
        self.convergence = False
        self.min_mse = 0
        self.convergence_delta = min_delta
        self.convergence_iter = 0
        if plot_save == 1:
            self.plot_save = True

    def _polynomial_features(self, X):
        """Generate polynomial features up to the specified degree."""
        X_poly = X.copy()
        for degree in range(2, self.poly_degree + 1):
            X_poly = np.hstack((X_poly, np.power(X, degree)))

        X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))
        return X_poly

    def fit(self, train_data_path, val_data_path, test_data_path, figures_directory):
        train_data = pd.read_csv(train_data_path)

        # Split the data into features (X) and target (y)
        X_train_pts = train_data.iloc[:, :-1].values  # Assuming all columns except last are features
        Y_train_pts = train_data.iloc[:, -1].values  # Assuming last column is the target

        # Generate polynomial features
        X_poly_train = self._polynomial_features(X_train_pts)
        # Training set
        self.X_train = X_poly_train
        self.y_train = Y_train_pts

        val_data = pd.read_csv(val_data_path)

        # Split the data into features (X) and target (y)
        X_val_pts = val_data.iloc[:, :-1].values  # Assuming all columns except last are features
        Y_val_pts = val_data.iloc[:, -1].values  # Assuming last column is the target

        # Generate polynomial features
        X_poly_val = self._polynomial_features(X_val_pts)
        # Validation set
        self.X_val = X_poly_val
        self.y_val = Y_val_pts

        test_data = pd.read_csv(test_data_path)

        # Split the data into features (X) and target (y)
        X_test_pts = test_data.iloc[:, :-1].values  # Assuming all columns except last are features
        Y_test_pts = test_data.iloc[:, -1].values  # Assuming last column is the target

        # Generate polynomial features
        X_poly_test = self._polynomial_features(X_test_pts)
        # testing set
        self.X_test = X_poly_test
        self.y_test = Y_test_pts

        # Initialize weights (including bias)
        self.weights = np.random.randn(self.X_train.shape[1])

        # Gradient descent on training set
        for i in range(self.num_iterations):
            # Compute predictions
            y_pred_train = np.dot(self.X_train, self.weights)

            # Compute gradients
            error_train = y_pred_train - self.y_train
            dw = (2 / len(self.X_train)) * np.dot(self.X_train.T, error_train)

            # Add regularization term
            if self.regularization == 'L2':
                dw += (2 * self.reg_lambda * self.weights)
            elif self.regularization == 'L1':
                dw += self.reg_lambda * np.sign(self.weights)

            
            # Update weights
            self.weights -= self.learning_rate * dw

            # Predict on the validation set using the current weights
            y_pred_val = np.dot(self.X_val, self.weights)


            mse = self.mean_squared_error(self.y_val, y_pred_val)
            var = self.variance(y_pred_val)
            std_dev = self.standard_deviation(y_pred_val)
            self.grad_desc_iterations.append({"Iteration_Num":i, "MSE": mse, "Variance": var, "Standard Deviation": std_dev, "Weights": self.weights.copy()})
            # plot the image and save it in the figures_directory too with the name 'Figure_{i}'. Figure should contain the line that 
            # is fitted to the data, training points, MSE, std and var 

            # Plot the training data, fitted line, and metrics, and save the plot as 'Figure_{i}'
            if self.plot_save:
                self._plot_iteration(i, figures_directory)

            self.min_mse = min(self.min_mse, mse)
            if self.min_mse - mse < self.convergence_delta:
                self.convergence = False
                self.min_mse = mse
            else:
                self.convergence = True
                self.convergence_iter = i + 1

            if self.convergence and self.plot_save == False:
                print(f"Breaking at iteration {i+1}")
                break

        return self

    def _plot_iteration(self, iteration, figures_directory):
        """Plots the training data, fitted line, MSE, variance, and standard deviation for each iteration."""
        plt.figure(figsize=(15, 10))

        # Subplot 1: Training Data and Fitted Line
        plt.subplot(2, 2, 1)
        plt.scatter(self.X_train[:, 1], self.y_train, color='blue', label='Training Data')
        x_range = np.linspace(min(self.X_train[:, 1]), max(self.X_train[:, 1]), 10000)
        X_poly = np.vander(x_range, N=self.poly_degree + 1, increasing=True)
        y_range_pred = np.dot(X_poly, self.weights)
        plt.plot(x_range, y_range_pred, color='red', label=f'Fitted Line (Iteration {iteration})')
        plt.title('Training Data and Fitted Line')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()

        # Subplot 2: MSE vs Iterations
        plt.subplot(2, 2, 2)
        mse_values = [entry["MSE"] for entry in self.grad_desc_iterations]
        plt.plot(range(iteration + 1), mse_values, color='green')
        plt.title('MSE vs Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')

        # Subplot 3: Variance vs Iterations
        plt.subplot(2, 2, 3)
        var_values = [entry["Variance"] for entry in self.grad_desc_iterations]
        plt.plot(range(iteration + 1), var_values, color='purple')
        plt.title('Variance vs Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Variance')

        # Subplot 4: Standard Deviation vs Iterations
        plt.subplot(2, 2, 4)
        std_values = [entry["Standard Deviation"] for entry in self.grad_desc_iterations]
        plt.plot(range(iteration + 1), std_values, color='orange')
        plt.title('Standard Deviation vs Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Standard Deviation')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot with the iteration number
        plt.savefig(os.path.join(figures_directory, f'{iteration}.png'))

        # Close the plot to free memory
        plt.close()

    def predict(self, X):
        """Predicts output using the trained model."""
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.weights)

    def mean_squared_error(self, y_true, y_pred):
        """Calculate the Mean Squared Error (MSE)."""
        return np.mean((y_true - y_pred) ** 2)

    def variance(self, y):
        """Calculate the variance of the data points."""
        return np.var(y)

    def standard_deviation(self, y):
        """Calculate the standard deviation of the data points."""
        return np.std(y)

    def evaluate(self):
        """Evaluate the model on the test set."""
        y_pred_test = self.predict(self.X_test)
        mse = self.mean_squared_error(self.y_test, y_pred_test)
        var = self.variance(self.y_test)
        std_dev = self.standard_deviation(self.y_test)
        return {"MSE": mse, "Variance": var, "Standard Deviation": std_dev}
