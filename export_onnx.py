import tensorflow as tf
import tf2onnx

# Load your trained Keras model
model = tf.keras.models.load_model("mnist_cnn.h5")

# Export to TF SavedModel format (needed for tf2onnx)
model.export("mnist_cnn_saved_model")

# Convert to ONNX
spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input_layer"),)
output_path = "mnist_cnn.onnx"

model_proto, _ = tf2onnx.convert.from_saved_model(
    "mnist_cnn_saved_model",
    input_signature=spec,
    output_path=output_path
)

print(f"Exported to {output_path}")
