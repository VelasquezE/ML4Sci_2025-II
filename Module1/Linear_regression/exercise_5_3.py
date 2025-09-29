# Step 1: Generate some experimental data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# -------------- Using spring model

N = 1000

# Let's assume the true spring constant k is 4.5 N/m
k_true = 4.5
np.random.seed(42) # for reproducibility

# Displacement (x) in meters. This is our feature X.
# The .reshape(-1, 1) is needed because scikit-learn expects 2D arrays for features.
xdata = np.linspace(0, 2, N)
x_displacement = xdata.reshape(-1, 1)

# Force (F) in Newtons. This is our target y.
# We'll calculate the true force and add some random "measurement noise"
noise = np.random.normal(0, 0.5, x_displacement.shape)
y_force = k_true * x_displacement + noise

# ------------- Training

# Split the data
X_train, x_test, Y_train, y_test = train_test_split(x_displacement, y_force, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(x_test)

# Create model. 
model = SGDRegressor(max_iter=1000, tol=1e-3)

MAX = 1000
epochs = np.arange(1, MAX, 0.5)

loss_function = np.zeros_like(epochs)

counter = 0
for ii in epochs:
    model.partial_fit(X_train, Y_train.ravel())
    y_prediction = model.predict(x_test)

    mse = mean_squared_error(y_test, y_prediction)
    loss_function[counter] = mse
    counter = counter + 1

r2 = r2_score(y_test, y_prediction)

plt.plot(epochs, loss_function, label = 
         fr"$R^{2}$: {r2:.4f}, mse: {loss_function[-1]:.3f}")
plt.xlabel(f'Iterations (N = {len(epochs)})')
plt.ylabel('Loss function')
plt.title('Loss (mse) vs. Iterations for Hooke’s law')
plt.xscale('log')
plt.legend()
plt.savefig("exercise5_3.pdf", dpi = 400)


