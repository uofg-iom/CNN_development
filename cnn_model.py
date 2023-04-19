import numpy as np
import tensorflow as tf
# from tensorflow import keras
from matplotlib import pyplot as plt


# Data Augmentation layer can take place as a model layer (synchronous)
# or to the dataset (asynchronously) before being passed to the model
# The synchronous option benefits from GPU acceleration, however it will be slower when run on a CPU
# The asynchronous option is better if we are training on a CPU

SYNCHRONOUS_AUGM = False
dataset_path = "00 - Datasets split by class - Watermark Removed"

class SkinTypeModel():
    
    def __init__(self, image_x = 300, image_y = 300, batch_size = 32):
        self.dataset_path = ""
        
        
        # Load dataset as TensorFlow Dataset Object
        # Shuffle argument shuffles all images in all classes and places them into batches
        # If shuffle is false, then the data is placed into batches based on the order in which they are loaded 
        # Image size downsizes the image to the specified resolution, it doesn't crop
        # If crop_to_aspect_ratio is selected then the image is cropped
        rescale_image_size = (image_x, image_y)
        self.ds_batch_size = batch_size

        self.train_ds, self.val_ds  = tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="both",
            batch_size=self.ds_batch_size,
            image_size=rescale_image_size,
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
        
    

