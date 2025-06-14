# IMAGE CLASSIFICATION MODEL

This project demonstrates the implementation and testing of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. CNNs are highly effective in recognizing patterns in image data and are a foundational concept in deep learning.


## Internship Information

- Company: CODTECH IT SOLUTIONS PVT. LTD

- Name: Peethani Satya Durga Rao

- Intern ID: CT06DF395

- Domain: Machine Learning

- Duration: 6 Weeks

- Mentor: Neela Santhosh Kumar


## Task Description

The goal of this task is to build and evaluate a CNN model using TensorFlow/Keras to classify images from the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32×32 color images in 10 different classes. This project walks through loading the data, building a CNN architecture, training the model, and evaluating its performance with visualizations and real-world image predictions.

### Objectives:

- Understand the architecture and layers of CNNs.

- Train a CNN model on CIFAR-10.

- Evaluate the model using accuracy and loss plots.

- Test the model using images from a URL.

- Interpret model predictions.


## Steps

### Data Loading:

CIFAR-10 dataset via TensorFlow/Keras API.

```python

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

```

> Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
> 170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step

---

### Building The Model: 

Built using Sequential, with Conv2D, MaxPooling2D, Flatten, and Dense layers.

```python

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

```

> /usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

---

### Compiling The Model:

```python

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

### Training: 

Trained for 10 epochs with validation on test data.

```python

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

```

> Epoch 1/10
> 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 74s 45ms/step - accuracy: 0.3325 - loss: 1.7916 - val_accuracy: 0.5536 - val_loss: 1.2428
Epoch 2/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 69s 44ms/step - accuracy: 0.5670 - loss: 1.2080 - val_accuracy: 0.5777 - val_loss: 1.2194
Epoch 3/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 81s 44ms/step - accuracy: 0.6261 - loss: 1.0508 - val_accuracy: 0.6499 - val_loss: 0.9987
Epoch 4/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 82s 44ms/step - accuracy: 0.6722 - loss: 0.9374 - val_accuracy: 0.6679 - val_loss: 0.9371
Epoch 5/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 82s 44ms/step - accuracy: 0.6976 - loss: 0.8589 - val_accuracy: 0.6819 - val_loss: 0.9239
Epoch 6/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 69s 44ms/step - accuracy: 0.7258 - loss: 0.7811 - val_accuracy: 0.6968 - val_loss: 0.8652
Epoch 7/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 70s 45ms/step - accuracy: 0.7450 - loss: 0.7282 - val_accuracy: 0.6946 - val_loss: 0.8966
Epoch 8/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 80s 43ms/step - accuracy: 0.7607 - loss: 0.6868 - val_accuracy: 0.7103 - val_loss: 0.8404
Epoch 9/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 83s 44ms/step - accuracy: 0.7755 - loss: 0.6450 - val_accuracy: 0.6967 - val_loss: 0.8696
Epoch 10/10
> 1563/1563 ━━━━━━━━━━━━━━━━━━━━ 80s 43ms/step - accuracy: 0.7855 - loss: 0.6055 - val_accuracy: 0.7021 - val_loss: 0.8828

---

### Evaluation: 

Plotted training/validation accuracy and loss.

```python

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

```
> 313/313 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.7076 - loss: 0.8685
Test accuracy: 0.7020999789237976

---

### Visualization of Accuracy and Loss

```python

# Accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy")
plt.show()

# Loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Loss")
plt.show()

```
> ![Model Accuracy](https://github.com/user-attachments/assets/b05635e2-42ab-4c8e-be3b-06ad1b486ed2)
  ![Model Loss](https://github.com/user-attachments/assets/94c60252-cca9-4da0-9702-7d264d42e129)

---

### Real Image Testing: 

Load and classify an external image using its URL.

#### Testing image: 

![Image](https://github.com/user-attachments/assets/07705374-6140-498a-a727-abcff679398e)

#### URL: 

https://github.com/user-attachments/assets/07705374-6140-498a-a727-abcff679398e

#### Code:

```python

import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Replace with your image URL
image_url = "https://example.com/path-to-image.jpg"

# Load image from URL
response = requests.get(image_url)
img = Image.open(BytesIO(response.content)).convert('RGB')
img = img.resize((32, 32))  # Resize to match CIFAR-10 input shape

# Preprocess the image
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Class names (CIFAR-10)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Show result
plt.imshow(img)
plt.title(f"Predicted: {class_names[predicted_class]}")
plt.axis('off')
plt.show()

```
#### Output:

> ![Predicted Output](https://github.com/user-attachments/assets/1bb24041-98d3-476e-a007-ed845cb19b37)
