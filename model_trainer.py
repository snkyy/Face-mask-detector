import os
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import sklearn.model_selection
from matplotlib import pyplot as plt

# READING DATA AND PREPROCESSING IMAGES

data = []
labels = []
path1 = "/home/snky/tcs/si/project/dataset/with_mask"
path2 = "/home/snky/tcs/si/project/dataset/without_mask"

for image in os.listdir(path1):
    image = tf.keras.preprocessing.image.load_img(os.path.join(path1, image), target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    data.append(image)
    labels.append("with_mask")

for image in os.listdir(path2):
    image = tf.keras.preprocessing.image.load_img(os.path.join(path2, image), target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    data.append(image)
    labels.append("without_mask")

# LABELS ONE HOT ENCODING
label_binarizer = sklearn.preprocessing.LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)
labels = np.array(labels)
data = np.array(data)

# SPLITTING DATA
(train_data, test_data, train_output, test_output) = sklearn.model_selection.train_test_split(
    data, labels, test_size=0.2, random_state=42)

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

# CREATING OUR MODEL
base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], weights="imagenet", include_top=False)

head_model = tf.keras.layers.AveragePooling2D()(base_model.output)
head_model = tf.keras.layers.Flatten()(head_model)
head_model = tf.keras.layers.Dense(64, activation="relu")(head_model)
head_model = tf.keras.layers.Dropout(0.3)(head_model)
head_model = tf.keras.layers.Dense(2, activation="sigmoid")(head_model)

model = tf.keras.models.Model(inputs=base_model.input, outputs=head_model)

# TRAINING TIME
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
res = model.fit(
    data_generator.flow(train_data, train_output, batch_size=32), epochs=10, steps_per_epoch=len(train_data) // 32,
    validation_steps=len(train_data) // 32)

# RUNNING TRAINED MODEL ON TEST DATA
prediction = model.predict(test_data, batch_size=32)
prediction = np.argmax(prediction, axis=1)

model.evaluate(test_data, test_output, batch_size=32)  # 0.9935

print(sklearn.metrics.classification_report(
    test_output.argmax(axis=1), prediction, target_names=label_binarizer.classes_))

# CREATING PLOT
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 10), res.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), res.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# SAVING THE MODEL
# model.save("mask_detection_model", save_format="h5")
