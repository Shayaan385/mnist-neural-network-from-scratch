
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

X_train = x_train.astype("float32") / 255.0
X_test = x_test.astype("float32") / 255.0

# ==========================================
# 2. MODEL CONFIGURATION
# ==========================================

hidden_size = 128
learning_rate = 0.5
epochs = 20
batch_size = 128

tf.keras.utils.set_random_seed(42)

# ==========================================
# 3. BUILD THE NEURAL NETWORK
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        hidden_size,
        activation="relu",
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.Dense(
        10,
        activation="softmax",
        kernel_initializer="he_normal"
    )
])

# ==========================================
# 4. COMPILE THE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 5. TRAIN THE MODEL
# ==========================================

start_time = time.perf_counter()

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

training_time = time.perf_counter() - start_time

# ==========================================
# 6. EVALUATE THE MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nLIBRARY MODEL RESULTS")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Training time: {training_time:.2f} seconds")

# ==========================================
# 7. TRAINING LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 4))
plt.plot(
    range(1, epochs + 1),
    history.history["loss"],
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy Loss")
plt.title("Training Loss - Keras")
plt.grid(True)
plt.tight_layout()
plt.savefig("library_loss.png")
plt.show()

# ==========================================
# 8. TRAINING ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 4))
plt.plot(
    range(1, epochs + 1),
    np.array(history.history["accuracy"]) * 100,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy - Keras")
plt.grid(True)
plt.tight_layout()
plt.savefig("library_accuracy.png")
plt.show()

# ==========================================
# 9. DISPLAY PREDICTIONS
# ==========================================

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")

    predicted = predictions[i]
    actual = y_test[i]

    color = "green" if predicted == actual else "red"

    plt.title(
        f"Pred: {predicted} | True: {actual}",
        color=color
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("library_predictions.png")
plt.show()

# ==========================================
# 10. SAVE THE TRAINED MODEL
# ==========================================

model.save("library_model.keras")

print("Library model and graphs saved successfully.")
