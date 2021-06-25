import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import save_model
from keras import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import keras
# from tensorflow import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from keras.applications import imagenet_utils

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
# config.log_device_placement = True
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)


IMAGE_SIZE = 224


class MobileNetMod(object):
    def __init__(self, classMap):
        self.model = self.__buildModel()
        # if os.path.isfile('models/mobilenet.h5'):
        #     self.model.load_weights('models/mobilenet.h5')
        self.__classMap = classMap
        self.graph = tf.get_default_graph()

    def loadModel(self, modelPath):
        if os.path.isfile(modelPath):
            self.model.load_weights(modelPath)

    def __prepare_image(self, file):
        img = image.load_img(file, target_size=(
            IMAGE_SIZE, IMAGE_SIZE))
        img_array = image.img_to_array(img)
        return self.__preProcImg(img_array)
        # img_expanded_dims = np.expand_dims(img_array, axis=0)
        # return keras.applications.mobilenet.preprocess_input(img_expanded_dims)

    def __preProcImg(self, img_array):
        dim = (IMAGE_SIZE, IMAGE_SIZE)
        img_array = cv2.resize(img_array, dim, interpolation=cv2.INTER_AREA)
        img_expanded_dims = np.expand_dims(img_array, axis=0)
        return keras.applications.mobilenet.preprocess_input(img_expanded_dims)

    def __buildModel(self):
        mobilenet = MobileNetV2()
        x = mobilenet.layers[-2].output
        predictions = Dense(8, activation='softmax')(x)
        model = Model(inputs=mobilenet.input, outputs=predictions)
        return model

    def __loadTrainData(self, train_path, val_path):
        train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                                                                                     class_mode='categorical', batch_size=20)
        val_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(val_path, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                                                                                   class_mode='categorical', batch_size=20)
        return train_batches, val_batches

    def __loadTestData(self, test_path):
        test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                                                                                    class_mode='categorical', batch_size=20, shuffle=False)
        return test_batches

    def train(self, train_path, val_path):
        train_batches, val_batches = self.__loadTrainData(train_path, val_path)
        self.model.compile(SGD(lr=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        history = self.model.fit_generator(train_batches, steps_per_epoch=10,
                                           validation_data=val_batches, validation_steps=10, epochs=300, verbose=2)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        # Get the ground truth from generator
        ground_truth = train_batches.classes
        # Get the label to class mapping from the generator
        label2index = train_batches.class_indices
        # Getting the mapping from class index to class label
        idx2label = dict((v, k) for k, v in label2index.items())
        print(idx2label)

        # _, val_labels =  next(val_batches)
        #
        # predictions = model.predict_generator(val_batches, steps=1, verbose=0)
        #
        # cm = confusion_matrix(val_batches, np.round(predictions[:,0]))
        # cm_plot_labels = []
        #
        # for k, v in label2index.items():
        #     cm_plot_labels.append(v)
        #
        # print(cm)

        # serialize model to JSON
        model_json = model.to_json()
        with open("mobilenet.json", "w") as json_file:
            json_file.write(model_json)

        save_model(model, 'mobilenet.h5')

        # from tensorflow.contrib import lite
        # tf.lite.TocoConverter

        converter = tf.lite.TocoConverter.from_keras_model_file("mobilenet.h5")
        tflite_model = converter.convert()
        open("model/mobilenet.tflite", "wb").write(tflite_model)

    def test(self, test_path):
        test_batchs = self.__loadTestData(test_path)
        filenames = test_batchs.filenames
        batch_size = 20
        nb_samples = len(filenames)
        steps = (nb_samples//batch_size) + 1
        predict = self.model.predict_generator(
            test_batchs, steps=steps)  # nb_samples)
        predict = np.argmax(predict, axis=-1)

        print('Confusion Matrix')

        print(confusion_matrix(test_batchs.classes, predict))
        print('Classification Report')
        target_names = ['hand', 'ok', 'paper', 'rock',
                        'scissors', 'the-finger', 'thumbdown', 'thumup']
        print(classification_report(test_batchs.classes,
                                    predict, target_names=target_names))

    def __decode_predictions(self, predictions):
        maxInRows = np.amax(predictions, axis=1)
        res = np.where(predictions == maxInRows)
        classes = list(map(lambda x: self.__classMap[x], res[1]))
        return classes

    def predict_file(self, img_path):
        img = self.__prepare_image(img_path)
        predictions = self.model.predict(img)
        return self.__decode_predictions(predictions)

    def predict(self, img_array):
        img = self.__preProcImg(img_array)
        with self.graph.as_default():
            predictions = self.model.predict(img)
            return self.__decode_predictions(predictions)


if __name__ == '__main__':
    # img_file = '../data/eval/thumbdown/6_2772.jpg'
    classMap = {0: 'hand', 1: 'ok', 2: 'paper', 3: 'rock',
                4: 'scissors', 5: 'the-finger', 6: 'thumbdown', 7: 'thumbup'}
    net = MobileNetMod(classMap)
    # preds = net.predict(img_file)
    # print(preds)

    net.test('../data/eval')
