import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import efficientnet.tfkeras as efn
from keras.callbacks import EarlyStopping
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.backend import switch

import libs.const as const
import keras_tuner as kt


def tune_units(hp: kt.HyperParameters, name, min_value=32, max_value=512, step=32): return hp.Int(
    name, min_value=min_value, max_value=max_value, step=step)


def base_model(hp: kt.HyperParameters):
    # with strategy.scope():
    model = models.Sequential()
    model.add(layers.Conv2D(tune_units(hp,'conv1_filters',max_value=64), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(tune_units(hp,'conv2_filters',max_value=256, min_value=64,step=64), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(tune_units(hp,'conv3_filters',max_value=256, min_value=64,step=64), (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(tune_units(hp,'dense_neurons',min_value=64,step=64), activation=hp.Choice('activation',values=["selu", "tanh", "relu",])))
    model.add(layers.Dense(10))


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        metrics=['accuracy'])

    return model


tuner = kt.Hyperband(base_model, objective="val_accuracy", max_epochs=10)


def dropout_model(hp):
    # with strategy.scope(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Flatten())
    model.add(layers.Dense(tune_units(hp), activation='relu'))
    model.add(layers.Dense(10))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        metrics=['accuracy'])
    return model


def augment_model(hp):
    # with strategy.scope(hp):
    model = models.Sequential()
    model.add(layers.RandomFlip(
        "horizontal_and_vertical", input_shape=(32, 32, 3)))
    model.add(layers.RandomRotation(tune_units(
        hp, min_value=0, max_value=0.2, step=0.05)))
    model.add(layers.RandomZoom(tune_units(
        hp, min_value=0, max_value=0.2, step=0.05)))
    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Flatten())

    model.add(layers.Dense(tune_units(hp), activation='relu'))
    model.add(layers.Dense(10))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        metrics=['accuracy'])
    return model


def batch_normalize_model(hp):
    # with strategy.scope(hp):
    model = models.Sequential()
    model.add(layers.RandomFlip("horizontal", input_shape=(32, 32, 3)))
    model.add(layers.RandomRotation(tune_units(
        hp, min_value=0, max_value=0.2, step=0.05)))
    model.add(layers.RandomZoom(tune_units(
        hp, min_value=0, max_value=0.2, step=0.05)))
    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Conv2D(tune_units(hp), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(tune_units(
        hp, min_value=.1, max_value=.4, step=.05)))

    model.add(layers.Flatten())

    model.add(layers.Dense(tune_units(hp), activation='relu'))
    model.add(layers.Dense(10))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        metrics=['accuracy'])
    return model


def efficient_net(hp):
    # with strategy.scope():
    model = models.Sequential()
    model.add(layers.RandomFlip(
        "horizontal_and_vertical", input_shape=(32, 32, 3)))
    model.add(layers.RandomRotation(tune_units(
        hp, min_value=0, max_value=0.2, step=0.05)))
    model.add(layers.RandomZoom(tune_units(
        hp, min_value=0, max_value=0.2, step=0.05)))
    #https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB7
    model.add(efn.EfficientNetB7(include_top=False, 
                                 pooling='avg'))

    # model.add(layers.Dense(256, activation='selu'))
    model.add(layers.Dense(tune_units(hp, min_value=64,
              max_value=1028, step=.64), activation='selu'))

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(10))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        metrics=['accuracy'])
    return model


def _tune_model(model, train_dataset,test_dataset):
    tuner = kt.Hyperband(model, objective='val_accuracy', max_epochs=5)
    tuner.search(train_dataset[0],train_dataset[1],validation_data=test_dataset, epochs=50,batch_size=64, validation_split=0.2,       callbacks=[
                 EarlyStopping(monitor='val_loss', patience=5)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    compiled_tuned_model = tuner.hypermodel.build(best_hps)
    # return compiled_tuned_model


def execute_model(model: models.Sequential, train_dataset, test_dataset, checkpoint_filepath, epochs=300, batch_size=1):
    train_images, train_labels = train_dataset
    test_images, test_labels = test_dataset

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=30)
    ]

    tuned_model = _tune_model(model, train_dataset,test_dataset)

    history = tuned_model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                              validation_data=(test_images, test_labels), callbacks=callbacks)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    return tuned_model


def evaluate_model(model, dataset, show="all"):
    SHOW_OPTIONS = ["all",
                    "incorrect",
                    "correct",
                    "none"
                    ]
    if not show in SHOW_OPTIONS:
        raise TypeError("Must be: 'all','incorrect', or 'correct'")
    x_test, y_test = dataset
    predictions: np.array = model.predict(x_test)
    for row in range(0, len(predictions)):
        if((show == "all") or (show == "incorrect" and np.argmax(predictions[row]) != y_test[row]) or (show == "correct" and np.argmax(predictions[row]) == y_test[row])):
            plt.imshow(x_test[row])
            plt.xlabel(
                f"predicted: {const.CLASS_NAMES[np.argmax(predictions[row])]}, actual: {const.CLASS_NAMES[y_test[row]]}")
            plt.show()
    return predictions


MODEL_METHODS = [
    base_model,
    dropout_model,
    augment_model,
    batch_normalize_model,
    efficient_net,
]


def load_model(model: str, model_dir="", train_dataset=None, test_dataset=None, epochs=1, batch_size=1):
    print(f"Execute Load Model: (model={model}), model_dir={model_dir}")
    model_path = model_dir + '/' + f"{model}.h5"
    try:
        return keras.models.load_model(model_path)
    except BaseException as err:
        print(
            f"Could not load saved {model}, compiling {model_path} now. Error: {str(err)}")
        model_to_build = MODEL_METHODS[const.MODELS.index(model)]
        return execute_model(_tune_model(model_to_build, train_dataset), train_dataset, test_dataset, model_path, batch_size=batch_size, epochs=epochs)
