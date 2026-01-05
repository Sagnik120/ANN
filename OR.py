import numpy as np

# ----------------------
# Step 1: Dataset (OR)
# ----------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [1]])

# ----------------------
# Step 2: Initialize weights & bias
# ----------------------
np.random.seed(42)
weights = np.random.randn(2, 1)
bias = np.random.randn(1)

# ----------------------
# Step 3: Activation function
# ----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# ----------------------
# Step 4: Training
# ----------------------
learning_rate = 0.1
epochs = 5000

for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)

    # Error
    error = y - y_pred

    # Backpropagation
    d_pred = error * sigmoid_derivative(y_pred)

    weights += learning_rate * np.dot(X.T, d_pred)
    bias += learning_rate * np.sum(d_pred)

# ----------------------
# Step 5: Testing
# ----------------------
print("Trained Weights:\n", weights)
print("Trained Bias:\n", bias)

print("\nOR Gate Predictions:")
for i in range(len(X)):
    output = sigmoid(np.dot(X[i], weights) + bias)
    print(f"{X[i]} -> {int(output >= 0.5)}")
