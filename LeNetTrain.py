import numpy as np
import pickle
import Lenet5
import MNIST
import time
import os
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
matplotlib.use('TkAgg',force=True)

# data pre-processing
def get_mnist_and_preprocessing():
    train_data, train_labels, test_data, test_labels = MNIST.load()
    train_data = np.pad(train_data, ((0, ), (2, ), (2, )), "constant")
    test_data = np.pad(test_data, ((0, ), (2, ), (2, )), "constant")

    normalize_min, normalize_max = -0.1, 1.175
    train_data = train_data/255*(normalize_max-normalize_min) + normalize_min
    test_data = test_data/255*(normalize_max-normalize_min) + normalize_min

    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)
    return train_data, train_labels, test_data, test_labels

def train_by_handcraft_lenet5(train_data, train_labels, test_data, test_labels):
    epochs = 20
    batch_size_SDLM = 500
    batch_size_train = 1
    batch_size_test = 2000
    learning_rate = np.array([0.0005]*2 + [0.0002]*3 + [0.0001]*3 + [0.00005]*4 + [0.00001]*8)

    restore_weights_path = "./Weights/LeNetWeightsFinal.pkl"
    if os.path.isfile(restore_weights_path):
        with open(restore_weights_path, "rb") as f:
            lenet5 = pickle.load(f)
    else:
        lenet5 = Lenet5.Lenet5()

    train_loss_list = list()
    train_accuracy_list = list()
    test_loss_list = list()
    test_accuracy_list = list()
    train_data_index = np.arange(len(train_labels))

    for epoch in range(epochs):
        print("=================================================")
        print("epoch \t", epoch+1, "\n")
        np.random.shuffle(train_data_index)

        time_start = time.time()

        lenet5.forward_propagation(train_data[train_data_index[:batch_size_SDLM]], train_labels[train_data_index[:batch_size_SDLM]])
        lenet5.SDLM(learning_rate[epoch])
        print("learning rates in trainable layers:", np.array([lenet5.C1.lr, lenet5.S2.lr, lenet5.C3.lr, lenet5.S4.lr, lenet5.C5.lr, lenet5.F6.lr]))

        for i in range(len(train_labels)//batch_size_train):
            loss, labels_ = lenet5.forward_propagation(train_data[train_data_index[i*batch_size_train:(i+1)*batch_size_train]], train_labels[train_data_index[i*batch_size_train:(i+1)*batch_size_train]])
            lenet5.backward_propagation()

            time_end = time.time()
            if ((i+1) % 25) == 0:
                print("Training Data Num ", (i+1)*batch_size_train)
                print("labels(GT) = ", train_labels[train_data_index[i*batch_size_train:(i+1)*batch_size_train]])
                print("labels(PD) = ", labels_)
                print("Loss = ", loss, "\tTime Elapsed = ", time_end-time_start, "\n")
                if not os.path.isdir("./Weights/"):
                    os.mkdir("./Weights/")
                with open("./Weights/LeNetWeights_"+str((i+1)*batch_size_train)+".pkl", "wb") as f:
                    pickle.dump(lenet5, f)

        print("training time = ", time_end - time_start, "\n")

        time_start = time.time()
        loss = 0
        accuracy = 0
        for i in range(len(train_labels)//batch_size_test):
            loss_, label_ = lenet5.forward_propagation(train_data[i*batch_size_test:(i+1)*batch_size_test], train_labels[i*batch_size_test:(i+1)*batch_size_test])
            loss += loss_
            accuracy += np.sum(np.equal(label_,train_labels[i*batch_size_test:(i+1)*batch_size_test]))
        loss /= len(train_data_index)
        accuracy /= len(train_data_index)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        time_end = time.time()
        print("training loss = ", loss, "\ttraining accuracy = ", accuracy, "\ttime elapse = ", time_end - time_start, "\n")

        time_start = time.time()
        loss = 0
        accuracy = 0
        for i in range(len(test_labels)//batch_size_test):
            loss_, label_ = lenet5.forward_propagation(test_data[i * batch_size_test:(i + 1) * batch_size_test], test_labels[i * batch_size_test:(i + 1) * batch_size_test])
            loss += loss_
            accuracy += np.sum(np.equal(label_, test_labels[i * batch_size_test:(i + 1) * batch_size_test]))
        loss /= len(test_labels)
        accuracy /= len(test_labels)
        test_loss_list.append(loss)
        test_accuracy_list.append(accuracy)
        time_end = time.time()
        print("testing loss = ", loss, "\ttesting accuracy = ", accuracy, "\ttime elapse = ", time_end - time_start, "\n")

    with open("./Weights/LeNetWeightsFinal.pkl", "wb") as f:
        pickle.dump(lenet5, f)

    x = np.arange(epochs)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.plot(x, train_accuracy_list)
    plt.plot(x, test_accuracy_list)
    plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
    plt.show()

def train_by_handcraft_lenet5(train_data, train_labels, test_data, test_labels):
    model = Lenet5.Lenet5_tf()
    history = model.train(train_data, train_labels, test_images=test_data, test_labels=test_labels)

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = get_mnist_and_preprocessing()

    # history = train_by_handcraft_lenet5(train_data, train_labels, test_data, test_labels)

    model = Lenet5.Lenet5_tf()

    # history = model.train(train_data, train_labels, test_images=test_data, test_labels=test_labels)
    # model.save_model()
    # model.save_model(to_tflite=True)

    model.load_model('checkpoint/lenet5.h5')
    model.load_model('checkpoint/lenet5.tflite')
    model.convert_to_tflite_quant(test_data)

    start_time = time.time()
    model.eval(test_data, test_labels)
    end_time = time.time()
    print(f'TF - time elapsed: {end_time-start_time}')

    start_time = time.time()
    model.eval(test_data, test_labels, use_tflite=True)
    end_time = time.time()
    print(f'TFlite - time elapsed: {end_time-start_time}')

    start_time = time.time()
    model.eval(test_data, test_labels, use_tflite_quant=True)
    end_time = time.time()
    print(f'TFLite(DynamicQuant) - time elapsed: {end_time-start_time}')

    model.convert_to_tflite_quant(test_data, 1)
    start_time = time.time()
    model.eval(test_data, test_labels, use_tflite_quant=True)
    end_time = time.time()
    print(f'TFLite(FullInteger) - time elapsed: {end_time-start_time}')

    model.convert_to_tflite_quant(test_data, mode=2)
    start_time = time.time()
    model.eval(test_data, test_labels, use_tflite_quant=True)
    end_time = time.time()
    print(f'TFLite(Float16) - time elapsed: {end_time-start_time}')






    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()