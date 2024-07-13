# Pizza vs Steak Binary Classification

This project aims to classify images as either pizza or steak using a Convolutional Neural Network (CNN). The dataset is organized into training and testing sets, with images categorized into 'pizza' and 'steak'. The project explores different CNN models and techniques to improve classification accuracy.

## Contents

- [Description](#description)
- [Dataset Information](#dataset-information)
- [Libraries Used](#libraries-used)
- [Data Preparation](#data-preparation)
- [Models and Techniques](#models-and-techniques)
- [Predictions](#predictions)
- [Contact](#contact)

## Description

The goal of this project is to build a binary classification model that can distinguish between images of pizza and steak. Various CNN architectures and data augmentation techniques are used to improve the model's performance.

## Dataset Information

The dataset is organized into the following directories:

- **pizza_steak**
  - **test**
    - **pizza**: 250 images
    - **steak**: 250 images
  - **train**
    - **pizza**: 750 images
    - **steak**: 750 images

## Libraries Used

- tensorflow
- keras
- matplotlib
- numpy
- pandas
- sklearn

## Data Preparation

To prepare the data, the following steps were taken:

Data Loading and Normalization: The `ImageDataGenerator` class from `tensorflow.keras.preprocessing.image` is used to load and normalize the image data.

```python
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'pizza_steak/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'pizza_steak/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
```
## Models and Techniques

Several CNN models were built and evaluated:

1. Model 1 (Tiny VGG): A basic CNN model with Convolutional layers, ReLU activations, pooling layers, and a fully connected output layer.
2. Model 2: Similar to Model 1 but with more MaxPool layers to reduce overfitting.
3. Model 3: Same as Model 2 but with data augmentation (without shuffling).
4. Model 4: Same as Models 2 and 3 but with shuffled augmented data.

### Model 1: Tiny VGG Architecture

```python
model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Input((224,224,3)),
  tf.keras.layers.Conv2D(filters=10, 
                         kernel_size=3, 
                         activation="relu"), 
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2,
                            padding="valid"), 
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

````

### Data Augmentation for Models 3 and 4

```python
train_datagen_augmented = ImageDataGenerator(rescale = 1./255,
                                             rotation_range= 20,
                                             shear_range=-0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)

train_generator_augmented = train_datagen_augmented.flow_from_directory(
    'pizza_steak/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=True # for model 4
    #shuffle= False - model 3)
``` 
## Predictions

Predictions were made using the trained models, and the results were visualized with random images from Internet.

## Contact

If you have any questions or suggestions, you can contact me at radi2035@gmail.com.