# mnist_test.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis]

# Load saved model
model = tf.keras.models.load_model("mnist_cnn.h5")

# Evaluate full test set
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {acc:.4f}, Loss: {loss:.4f}")

# --- Example Prediction (for README or debugging) ---
# Pick one random sample from the test set
index = np.random.randint(0, len(x_test))
sample = x_test[index]
label = y_test[index]

# Predict the digit
prediction = np.argmax(model.predict(sample[np.newaxis, ...]))
print(f"Example prediction -> True Label: {label}, Predicted: {prediction}")

# Show and save the result (optional for README)
plt.imshow(sample.squeeze(), cmap='gray')
plt.title(f"Predicted: {prediction}")
plt.axis('off')

# Save image for README (optional)
plt.savefig("docs/sample_prediction.png", bbox_inches='tight')
plt.show()
