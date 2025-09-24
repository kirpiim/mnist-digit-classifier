import tensorflow as tf

model = tf.keras.models.load_model("mnist_cnn.h5")
model.save("mnist_cnn_saved_model")  # save as TF SavedModel

import tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(model)
with open("mnist_cnn.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Exported to mnist_cnn.onnx")
