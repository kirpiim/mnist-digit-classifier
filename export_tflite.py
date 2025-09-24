import tensorflow as tf

model = tf.keras.models.load_model("mnist_cnn.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("mnist_cnn.tflite", "wb") as f:
    f.write(tflite_model)

print("Exported to mnist_cnn.tflite")
