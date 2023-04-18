import os.path

import Layer
import RBF_BITMAP
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def get_combination_map():
    C3_combination = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
                  [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2], [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
                  [0, 1, 2, 3, 4, 5]]
    return C3_combination

class Lenet5_tf():
    def __init__(self):
        self.model = None
        self.tflite_model = None
        self.tflite_quant_model = None

    def build_model_with_lenet5(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
        self.model.add(layers.MaxPooling2D(2,2))
        self.model.add(layers.Conv2D(16, (5, 5), activation='relu'))
        self.model.add(layers.MaxPooling2D(2,2))
        self.model.add(layers.Conv2D(32, (5, 5), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(10))

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    def train(self, train_images, train_labels, epochs=10, test_images=None, test_labels=None):
        self.build_model_with_lenet5()
        history = self.model.fit(train_images, train_labels, batch_size=64, epochs=epochs, validation_data=(test_images, test_labels))
        # plt.plot(history.history['accuracy'], label='accuracy')
        # plt.plot(history.history['val_accuracy'], label='val_accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.ylim([0.5, 1])
        # plt.legend(loc='lower right')

        epochs_range = range(epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
        plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], label='Training Loss')
        plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        return history

    def eval(self, test_images, test_labels, use_tflite=False, use_tflite_quant=False):
        if use_tflite:
            test_acc = 0
            predictions = list()
            input_details = self.tflite_model.get_input_details()
            output_details = self.tflite_model.get_output_details()

            intput_batch = input_details[0]['shape'][0]
            for idx in range(test_images.shape[0]):
                input_data = test_images[idx:idx+1]
                self.tflite_model.set_tensor(input_details[0]['index'], input_data)
                self.tflite_model.invoke()
                output_data = self.tflite_model.get_tensor(output_details[0]['index'])
                if np.argmax(output_data) == test_labels[idx]:
                    test_acc += 1
            test_acc /= len(test_labels)
        elif use_tflite_quant:
            test_acc = 0
            predictions = list()
            input_details = self.tflite_quant_model.get_input_details()
            output_details = self.tflite_quant_model.get_output_details()

            intput_batch = input_details[0]['shape'][0]
            for idx in range(test_images.shape[0]):
                input_data = test_images[idx:idx+1]
                self.tflite_quant_model.set_tensor(input_details[0]['index'], input_data)
                self.tflite_quant_model.invoke()
                output_data = self.tflite_quant_model.get_tensor(output_details[0]['index'])
                if np.argmax(output_data) == test_labels[idx]:
                    test_acc += 1
            test_acc /= len(test_labels)
        else:
            test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=2)
        print(test_acc)

    def predict(self, img, use_tflite=False, softmax=True):

        if use_tflite:
            predictions = list()
            input_details = self.tflite_model.get_input_details()
            output_details = self.tflite_model.get_output_details()

            intput_batch = input_details[0]['shape'][0]
            for idx in range(img.shape[0]):
                input_data = img[idx:idx+1]
                self.tflite_model.set_tensor(input_details[0]['index'], input_data)
                self.tflite_model.invoke()
                output_data = self.tflite_model.get_tensor(output_details[0]['index'])
                predictions.append(output_data)
        else:
            predictions = self.model.predict(img)

        if softmax:
            predictions = tf.nn.softmax(predictions)
        return predictions

    def save_model(self, model_path='checkpoint/lenet5', to_tflite=False):
        dirname = os.path.dirname(model_path)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        if to_tflite:
            # Convert the model.
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()

            # Save the model.
            with open(model_path+'.tflite', 'wb') as f:
                f.write(tflite_model)
        else:
            self.model.save(model_path+'.h5')

    def convert_to_tflite_quant(self, cal_dataset=None, mode=0):
        def representative_dataset():
            for idx in range(len(cal_dataset)):
                yield [cal_dataset[idx:idx+1]]
        if self.model:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        else:
            raise Exception('Assign model first.')

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if mode == 1:
            converter.representative_dataset = representative_dataset
        elif mode == 2:
            converter.target_spec.supported_types = [tf.float16]
        # if cal_dataset.any():
        #
        tflite_model_content = converter.convert()
        self.tflite_quant_model = tf.lite.Interpreter(model_content=tflite_model_content)
        self.tflite_quant_model.allocate_tensors()

    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model_content = converter.convert()
        self.tflite_model = tf.lite.Interpreter(model_content=tflite_model_content)
        self.tflite_model.allocate_tensors()

    def load_model(self, model_path='checkpoint/lenet5.h5'):
        if model_path.endswith('.h5'):
            self.model = models.load_model(model_path)
        elif model_path.endswith('.tflite'):
            self.tflite_model = tf.lite.Interpreter(model_path=model_path)
            self.tflite_model.allocate_tensors()


class Lenet5():
    def __init__(self):
        self.C1 = Layer.ConvolutionalLayer([5, 5, 1, 6], pad="VALID", activation_function="SQUASHING")
        self.S2 = Layer.PoolingLayer([2, 2, 6], mode="AVERAGE", activation_function="SQUASHING")
        self.C3 = Layer.ConvolutionalCombinationLayer([5, 5, 16], get_combination_map(), pad="VALID", activation_function="SQUASHING")
        self.S4 = Layer.PoolingLayer([2, 2, 16], mode="AVERAGE", activation_function="SQUASHING")
        self.C5 = Layer.ConvolutionalLayer([5, 5, 16, 120], pad="VALID", activation_function="SQUASHING")
        self.F6 = Layer.FullyConnectedLayer([120, 84], activation_function="SQUASHING")
        self.RBF = Layer.RBFLayer(RBF_BITMAP.rbf_bitmap())

    def forward_propagation(self, inputs, labels):
        c1_output = self.C1.forward_propagation(inputs)
        s2_output = self.S2.forward_propagation(c1_output)
        c3_output = self.C3.forward_propagation(s2_output)
        s4_output = self.S4.forward_propagation(c3_output)
        c5_output = self.C5.forward_propagation(s4_output)
        c5_output = np.squeeze(c5_output, axis=(1, 2))
        f6_output = self.F6.forward_propagation(c5_output)
        loss, outputs = self.RBF.forward_propagation(f6_output, labels)
        return loss, outputs

    def backward_propagation(self):
        d_inputs_rbf = self.RBF.backward_propagation()
        d_inputs_f6 = self.F6.backward_propagation(d_inputs_rbf)
        d_inputs_f6 = d_inputs_f6.reshape([d_inputs_f6.shape[0], 1, 1, d_inputs_f6.shape[-1]])
        d_inputs_c5 = self.C5.backward_propagation(d_inputs_f6)
        d_inputs_s4 = self.S4.backward_propagation(d_inputs_c5)
        d_inputs_c3 = self.C3.backward_propagation(d_inputs_s4)
        d_inputs_s2 = self.S2.backward_propagation(d_inputs_c3)
        self.C1.backward_propagation(d_inputs_s2)
    
    def SDLM(self, learning_rate):
        d2_inputs_rbf = self.RBF.SDLM()
        d2_inputs_f6 = self.F6.SDLM(d2_inputs_rbf, learning_rate)
        d2_inputs_f6 = d2_inputs_f6.reshape([d2_inputs_f6.shape[0], 1 , 1, d2_inputs_f6.shape[-1]])
        d2_inputs_c5 = self.C5.SDLM(d2_inputs_f6, learning_rate)
        d2_inputs_s4 = self.S4.SDLM(d2_inputs_c5, learning_rate)
        d2_inputs_c3 = self.C3.SDLM(d2_inputs_s4, learning_rate)
        d2_inputs_s2 = self.S2.SDLM(d2_inputs_c3, learning_rate)
        self.C1.SDLM(d2_inputs_s2, learning_rate)        

