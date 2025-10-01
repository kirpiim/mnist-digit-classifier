# mnist_test.py
import tensorflow as tf

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis]

# Load saved model
model = tf.keras.models.load_model("mnist_cnn.h5")

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f" Test accuracy: {acc:.4f}, Loss: {loss:.4f}")
