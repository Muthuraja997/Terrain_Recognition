import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input


base_dir ="/content/drive/MyDrive/Colab Notebooks/terrain dataset"
grass_dir = os.path.join(base_dir, "Grassy_Terrain")
marshy_dir = os.path.join(base_dir, "Marshy_Terrain")
rocky_dir = os.path.join(base_dir, "Rocky_Terrain")
sandy_dir = os.path.join(base_dir, "Sandy_Terrain")
other_dir = os.path.join(base_dir, "Other_Image")

def load_and_preprocess_image(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels)

grass_images, grass_labels = load_and_preprocess_image(grass_dir, "Grassy_Terrain")
marshy_images, marshy_labels = load_and_preprocess_image(marshy_dir, "Marshy_Terrain")
rocky_images, rocky_labels = load_and_preprocess_image(rocky_dir, "Rocky_Terrain")
sandy_images, sandy_labels = load_and_preprocess_image(sandy_dir, "Sandy_Terrain")
other_images, other_labels = load_and_preprocess_image(other_dir, "Other_Image")

all_images = np.concatenate([grass_images, marshy_images, rocky_images, sandy_images, other_images], axis=0)
all_labels = np.concatenate([grass_labels, marshy_labels, rocky_labels, sandy_labels, other_labels], axis=0)

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)


X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels_encoded, test_size=0.2, random_state=42)


input_tensor = layers.Input(shape=(224, 224, 3))
base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(5, activation='softmax')(x)

final_model = models.Model(inputs=base_model.input, outputs=output)

final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = final_model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt




y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


conf_matrix = confusion_matrix(y_test, y_pred_classes)


class_names = label_encoder.classes_
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


print(classification_report(y_test, y_pred_classes, target_names=class_names))


import matplotlib.pyplot as plt





# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()






