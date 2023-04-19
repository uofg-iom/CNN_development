import numpy as np
import tensorflow as tf
# from tensorflow import keras
from matplotlib import pyplot as plt
from enum import Enum

class ModelType(Enum):
    CustomModel = 0
    MobileNet = 1
    ResNet50 = 2
    ResNet152 = 3
    VGG16 = 4
    Xception = 5

# Data Augmentation layer can take place as a model layer (synchronous)
# or to the dataset (asynchronously) before being passed to the model
# The synchronous option benefits from GPU acceleration, however it will be slower when run on a CPU
# The asynchronous option is better if we are training on a CPU

SYNCHRONOUS_AUGM = False
dataset_path = "00 - Datasets split by class - Watermark Removed"

class SkinTypeModel():
    
    def __init__(self, path, image_x = 300, image_y = 300, batch_size = 32):
        self.dataset_path = path
        
        
        # Load dataset as TensorFlow Dataset Object
        # Shuffle argument shuffles all images in all classes and places them into batches
        # If shuffle is false, then the data is placed into batches based on the order in which they are loaded 
        # Image size downsizes the image to the specified resolution, it doesn't crop
        # If crop_to_aspect_ratio is selected then the image is cropped
        self.rescale_image_size = (image_x, image_y)
        self.ds_batch_size = batch_size

        self.train_ds, self.val_ds  = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="both",
            batch_size=self.ds_batch_size,
            image_size=self.rescale_image_size,
            crop_to_aspect_ratio=False,
            color_mode='rgb',
            shuffle=False,
            seed=100)

        self.class_names = self.train_ds.class_names
        print("Classifications: \n", self.class_names)

        # For demonstration, iterate over the batches yielded by the dataset.
        # for data, labels in train_ds:
        #    print(data.shape)  # (64, 200, 200, 3)
        #    print(data.dtype)  # float32
        #    print(labels.shape)  # (64,)
        #    print(labels.dtype)  # int32

    def _add_data_augmentation(self):
        # "Filters" to be applied sequentially to each image
        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                # tf.keras.layers.RandomCrop(200, 200),
                tf.keras.layers.RandomBrightness(0.1),
                tf.keras.layers.RandomContrast(0.2), # Computationally expensive
                # tf.keras.layers.RandomZoom(0.2, 0.2) # Computationally expensive
                # tf.keras.layers.RandomTranslation(height_factor= 0.2, width_factor= 0.2, fill_mode="nearest") # Perhaps too much for this application
            ]
        )


    def build_model(self, num_classes, model_type, input_image_size = None):
        
        if input_image_size == None:
            in_w = self.rescale_image_size[0]
            in_h = self.rescale_image_size[1]
        else:            
            in_w = input_image_size[0]
            in_h = input_image_size[1]
            
        # When input data size is variable
        inputs = tf.keras.Input(shape=(in_w, in_h, 3))
        # x = tf.keras.layers.CenterCrop(height=200, width=200)(inputs)
        # If synchronous, add data augmentation layer as part of the model
        if(SYNCHRONOUS_AUGM):
            x = self.data_augmentation(inputs)
            # Scaling Layer (scales data into 0.0 - 1.0 range)
            x = tf.keras.layers.Rescaling(scale=1.0 / 255)(x)
        else:
            # Scaling Layer (scales data into 0.0 - 1.0 range)
            x = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)
        
        if model_type == ModelType.CustomModel:
            
            
            # ----------- Add Model Layers here ---------------
            # Here layers are added randomly as example. See main guide.
                
            # Apply some convolution and pooling layers
            x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            # Apply global average pooling to get flat feature vectors
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # -------------------------------------------------
            
            
            # Dropout layer helps prevent overfitting by setting one input to 0 randomly
            # arguement is float from 0-1, fraction of the input units to drop.
            x = tf.keras.layers.Dropout(0.1)(x)
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        
        elif model_type == ModelType.ResNet152:
            # Following preprocess_input() method is required by ResNet
            # See https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet152V2
            
            x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
            x = tf.keras.layers.Dropout(0.1)(x)
            outputs = tf.keras.applications.ResNet152V2(
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=(in_w, in_h, 3),
                pooling="avg",
                classes=num_classes,
                classifier_activation="softmax")(x)
            
        
        return tf.keras.Model(inputs, outputs)

    def show_train_dataset(self, sel_batch = 6, num_rows = 3, num_cols = 3):
        
        plt.figure(figsize=(10, 10))
        # .take() selects the batch number to show
        for images, labels in self.train_ds.take(sel_batch):
            for i in range(num_rows*num_cols):
                ax = plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.tight_layout()
                # plt.axis("off")
        
    

