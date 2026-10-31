
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Convert 28x28 images into 784-element vectors
X_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
X_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

# ==========================================
# 2. DEFINE NETWORK ARCHITECTURE
# ==========================================

input_size = 784
hidden_size = 128
output_size = 10

learning_rate = 0.1
epochs = 10
batch_size = 64

rng = np.random.default_rng(42)

# He initialization for ReLU layers
W1 = (
    rng.standard_normal((input_size, hidden_size))
    * np.sqrt(2.0 / input_size)
)
b1 = np.zeros((1, hidden_size))

W2 = (
    rng.standard_normal((hidden_size, output_size))
    * np.sqrt(2.0 / hidden_size)
)
b2 = np.zeros((1, output_size))

# ==========================================
# 3. ACTIVATION FUNCTIONS
# ==========================================

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ==========================================
# 4. FORWARD PROPAGATION
# ==========================================

def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    P = softmax(Z2)

    return Z1, A1, P

# ==========================================
# 5. LOSS FUNCTION
# ==========================================

def cross_entropy(P, y):
    m = len(y)
    correct_probabilities = P[np.arange(m), y]

    return -np.mean(
        np.log(np.clip(correct_probabilities, 1e-12, 1.0))
    )

# ==========================================
# 6. TRAIN THE NEURAL NETWORK
# ==========================================

loss_history = []
accuracy_history = []

start_time = time.perf_counter()

for epoch in range(epochs):

    # Shuffle training data each epoch
    indices = rng.permutation(len(X_train))

    for start in range(0, len(X_train), batch_size):

        batch_indices = indices[start:start + batch_size]

        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        m = len(X_batch)

        # Forward propagation
        Z1, A1, P = forward(X_batch)

        # ==================================
        # BACKPROPAGATION
        # ==================================

        # Output layer gradient
        dZ2 = P.copy()
        dZ2[np.arange(m), y_batch] -= 1
        dZ2 /= m

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer gradient
        dZ1 = (dZ2 @ W2.T) * relu_derivative(Z1)

        dW1 = X_batch.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # ==================================
        # GRADIENT DESCENT
        # ==================================

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # Evaluate after each epoch
    _, _, train_probabilities = forward(X_train)

    loss = cross_entropy(train_probabilities, y_train)

    predictions = np.argmax(train_probabilities, axis=1)
    accuracy = np.mean(predictions == y_train)

    loss_history.append(loss)
    accuracy_history.append(accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {loss:.4f} | "
        f"Training accuracy: {accuracy * 100:.2f}%"
    )

training_time = time.perf_counter() - start_time

# ==========================================
# 7. EVALUATE ON TEST DATA
# ==========================================

_, _, test_probabilities = forward(X_test)

test_predictions = np.argmax(test_probabilities, axis=1)

test_accuracy = np.mean(test_predictions == y_test)
test_loss = cross_entropy(test_probabilities, y_test)

print("\nFINAL RESULTS")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Training time: {training_time:.2f} seconds")

# ==========================================
# 8. PLOT TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training Loss - NumPy Neural Network")
plt.grid(True)
plt.tight_layout()
plt.savefig("scratch_loss.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(
    range(1, epochs + 1),
    np.array(accuracy_history) * 100,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy - NumPy Neural Network")
plt.grid(True)
plt.tight_layout()
plt.savefig("scratch_accuracy.png")
plt.show()

# ==========================================
# 9. SHOW SOME TEST PREDICTIONS
# ==========================================

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")

    predicted = test_predictions[i]
    actual = y_test[i]

    color = "green" if predicted == actual else "red"

    plt.title(
        f"Pred: {predicted} | True: {actual}",
        color=color
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("scratch_predictions.png")
plt.show()

# Save model parameters for later use
np.savez(
    "scratch_weights.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("Model weights and graphs saved successfully.")
