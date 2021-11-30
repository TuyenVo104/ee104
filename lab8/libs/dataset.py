from libs.const import ORIGIN_CATEGORIES
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def save_images():
    """Only use if images haven't been loaded yet

        Loads pictures from const.py into a temp directory
    """
    for category in ORIGIN_CATEGORIES:
        for origin, index in zip(ORIGIN_CATEGORIES[category], range(0, len(ORIGIN_CATEGORIES[category]))):
            tf.keras.utils.get_file(f"{category[:-8]}/{category[:-8]}_{index}.jpg".lower(
            ), origin, cache_dir=f"./origins/{category[:-8]}/".lower())
            # if this line fails make sure the corresponding folders exist on the filesystem


def load_images(image_dir="/tmp/.keras/datasets/", plot = False):
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir, image_size=(32, 32), color_mode='rgb',
    )

    plt.figure(figsize=(10, 10))
    x_test = []
    y_test = []
    
    for imgs, labels in test_dataset:
        for img, label in zip(imgs, labels,):
            if plot:
                plt.imshow(img/255)
                # The CIFAR labels happen to be arrays,
                # which is why you need the extra index
                plt.xlabel(test_dataset.class_names[label])
                plt.show()
            x_test.append(img.numpy()/255)
            y_test.append(label.numpy())
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return (x_test,y_test)
