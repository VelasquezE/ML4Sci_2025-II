import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.costs = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass: compute predictions
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
          
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Create and train model
            model = SGDRegressor(max_iter=1000, tol=1e-3)
            # Compute cost (for monitoring)
            cost = self.compute_cost(y, y_pred)
            self.costs.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters using gradient descent
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
            # Print progress
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost:.4f}")
    
    def compute_cost(self, y_true, y_pred):
        """Compute cross-entropy cost"""
        m = len(y_true)
        # Avoid log(0) by clipping predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        return cost
    
    def predict(self, X):
        """Make predictions"""
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and train model
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    print(f"Final weights: {model.weights}")
    print(f"Final bias: {model.bias}")
    print(f"Accuracy: {np.mean(predictions == y):.4f}")
    
    # Plot cost function
    plt.figure(figsize=(10, 6))
    plt.plot(model.costs)
    plt.title('Cost Function Over Training, SDG')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig("exercise6_2.pdf", dpi = 400)