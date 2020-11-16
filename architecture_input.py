# import the necessary packages
import tensorflow as tf
import os
import sys
import random
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt


DATASET = sys.argv[1]
PERCENTAGE_PRIVILEGED = int(sys.argv[2])
N = sys.argv[3] #run number N
PERCENTAGE_TEST = 0.3
EPOCHS = 10

def split_knowledge_set(X, privileged_indices, boolean):
    """
    Get the set of the desired knowledge, privileged or unprivileged
    :param X: dataset to split into privileged and unprivileged knowledge
    :param privileged_indices: indices randomly chosen to become the privileged one
    :param boolean: true => select privileged information
                    false => select unprivileged information
    :return: ndarray with desired information
    """
    if boolean:
        return X[:, privileged_indices[0]]
    else:
        return X[:, privileged_indices[1]]

def split_indices(n_features):
    """
    Provide a list of two arrays, the first contains privileged indices and the second the unprivileged ones.
    The size of the two arrays is given by the PERCENTAGE of privileged knowledge.
    :param n_features: size of the array you want to split
    :return: list [[privileged_indices],[uprivileged_indices]]
    """
    indices = list(range(n_features))
    random.shuffle(indices)
    split_index = int(float(PERCENTAGE_PRIVILEGED) / 100 * n_features)
    return [indices[:split_index], indices[split_index:]]

def save_list_indices(indices):
    """
    It saves privileged and unprivileged indices in two different files
    :param indices: list of indices. In indices[0] --> privileged indices. In indices[1] --> unprivileged indices.
    """
    file = open("privileged_indices.txt", "w")
    file.write(str(indices[0]))
    file.close()
    file = open("unprivileged_indices.txt", "w")
    file.write(str(indices[1]))
    file.close()

def plot(subject, values_toPlot):
    """
    Plot and save images of metrics/losses
    :param subject: metric or loss to plot
    :param values_toPlot: labels of the values to plot
    """
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(len(values_toPlot), 1, figsize=(13, 13))
    # loop over the valuesNames
    for (i, l) in enumerate(values_toPlot):
        # plot the subject
        title = (subject + " for {}").format(l)
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel(subject)
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].legend()
    # save the figure
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\saved_images\\' + subject.format("png"))
    plt.close()

def multi_layer_perceptron(n_features, classes, finalAct):
    input_ = layers.Input(shape=n_features)
    model = layers.Dense(units=256, activation='relu')(input_)
    model = layers.Dense(units=192, activation='relu')(model)
    model = layers.Dense(units=128, activation='relu')(model)
    model = layers.Dense(units=64, activation='relu')(model)
    output_ = layers.Dense(units=classes, activation=finalAct)(model)

    model = models.Model(input_, output_, name="Model_1")
    #model.summary()

    return model

def arch_final(n_features_m2m3, classes, finalAct):
    inputs = layers.Input(shape=n_features_m2m3)

    dense1_2 = layers.Dense(units=256, activation='relu')(inputs)
    dense1_3 = layers.Dense(units=256, activation='relu')(inputs)

    dense2_2 = layers.Dense(units=192, activation='relu')(dense1_2)
    dense2_3 = layers.Dense(units=192, activation='relu')(dense1_3)

    dense3_2 = layers.Dense(units=128, activation='relu')(dense2_2)
    dense3_3 = layers.Dense(units=128, activation='relu')(dense2_3)

    dense4_2 = layers.Dense(units=64, activation='relu')(dense3_2)
    dense4_3 = layers.Dense(units=64, activation='relu')(dense3_3)
    dense4_sum = layers.Dense(units=64, activation='relu')((dense3_2 + dense3_3))

    m2_output = layers.Dense(units=classes, activation=finalAct, name='m2_output')(dense4_2)
    m3_output = layers.Dense(units=classes, activation=finalAct, name='m3_output')(dense4_3)
    m_total = layers.Dense(units=classes, activation=finalAct, name='m_total')(dense4_sum)

    model_fin = models.Model(inputs, [m2_output, m3_output, m_total], name="Architecture")
    #model_fin.summary()

    return model_fin

if __name__ == "__main__":
    mat = scipy.io.loadmat('..\\..\\..\\..\\data\\' + DATASET)

    X = mat['X']  # data
    y = mat['Y']  # label

    # one hot encoding
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(y)

    classes = len(ohe.categories_[0])

    n_samples, n_features = X.shape  # number of samples and number of features
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=PERCENTAGE_TEST)  # Split data into train and test

    p_indices = split_indices(n_features)

    save_list_indices(p_indices)

    x_unprivileged_train = split_knowledge_set(x_train, p_indices, False)  # train set with unprivileged knowledge
    x_unprivileged_test = split_knowledge_set(x_test, p_indices, False)  # test set with unpriviliged knowledge

    y_test_array = np.argwhere(y_test != 0)[:, 1]  # to use during test for model 1, 2 and 3

    # convert numpy to tensors
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)
    x_unprivileged_train = tf.convert_to_tensor(x_unprivileged_train)
    x_unprivileged_test = tf.convert_to_tensor(x_unprivileged_test)

    m1 = multi_layer_perceptron(n_features, classes, 'softmax')
    adam = optimizers.Adam(lr=0.0001)

    m1.compile(adam, loss='categorical_crossentropy', metrics=["accuracy","mse"])
    history1 = m1.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=0)
    m1.save(os.getcwd() + '\\saved_models\\model_1', save_format="h5")

    privileged_dist = m1.predict(x_train)
    y_prediction = []
    for i in privileged_dist:
        y_prediction = np.append(y_prediction, int(np.argmax(i)))
    y_prediction = y_prediction[:, None]
    y_train_privileged = ohe.fit_transform(y_prediction)
    #y_train_privileged = tf.convert_to_tensor(y_train_privileged)

    n_features_m2m3 = (n_features - int(n_features * PERCENTAGE_PRIVILEGED / 100))
    architecture = arch_final(n_features_m2m3, classes, 'softmax')
    losses = {"m2_output": "kullback_leibler_divergence", "m3_output": "categorical_crossentropy", "m_total": "categorical_crossentropy"}
    lossWeights = {"m2_output": 1.0, "m3_output": 1.0, "m_total": 1.0}
    architecture.compile(adam, loss=losses, loss_weights= lossWeights, metrics=["accuracy", "mse"])
    H = architecture.fit(x=x_unprivileged_train, y={"m2_output": y_train_privileged, "m3_output": y_train, "m_total": y_train}, epochs=EPOCHS, validation_split=0.2, verbose=0)
    architecture.save(os.getcwd() + '\\saved_models\\architecture', save_format="h5")

    m1_accuracy = history1.history['accuracy'][EPOCHS - 1]
    m2_accuracy = H.history['m2_output_accuracy'][EPOCHS - 1]
    m3_accuracy = H.history['m3_output_accuracy'][EPOCHS - 1]
    m_total_accuracy = H.history['m_total_accuracy'][EPOCHS-1]

    m1_loss = history1.history['loss'][EPOCHS - 1]
    m2_loss = H.history['m2_output_loss'][EPOCHS - 1]
    m3_loss = H.history['m3_output_loss'][EPOCHS - 1]
    m_total_loss = H.history['m_total_loss'][EPOCHS-1]
    sum_loss = H.history['loss'][EPOCHS - 1]

    m1_mse = history1.history['mse'][EPOCHS-1]
    m2_mse = H.history['m2_output_mse'][EPOCHS-1]
    m3_mse = H.history['m3_output_mse'][EPOCHS-1]
    m_total_mse = H.history['m_total_mse'][EPOCHS-1]

    m1_val_accuracy = history1.history['val_accuracy'][EPOCHS - 1]
    m2_val_accuracy = H.history['val_m2_output_accuracy'][EPOCHS - 1]
    m3_val_accuracy = H.history['val_m3_output_accuracy'][EPOCHS - 1]
    m_total_val_accuracy = H.history['val_m_total_accuracy'][EPOCHS - 1]

    m1_val_loss = history1.history['val_loss'][EPOCHS - 1]
    m2_val_loss = H.history['val_m2_output_loss'][EPOCHS - 1]
    m3_val_loss = H.history['val_m3_output_loss'][EPOCHS - 1]
    m_total_val_loss = H.history['val_m_total_loss'][EPOCHS - 1]
    sum_val_loss = H.history['val_loss'][EPOCHS - 1]

    m1_val_mse = history1.history['val_mse'][EPOCHS - 1]
    m2_val_mse = H.history['val_m2_output_mse'][EPOCHS - 1]
    m3_val_mse = H.history['val_m3_output_mse'][EPOCHS - 1]
    m_total_val_mse = H.history['val_m_total_mse'][EPOCHS - 1]

    print('{0:.2f}'.format(m1_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m2_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m3_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m_total_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m1_loss), end=" ")
    print('{0:.2f}'.format(m2_loss), end=" ")
    print('{0:.2f}'.format(m3_loss), end=" ")
    print('{0:.2f}'.format(m_total_loss), end=" ")
    print('{0:.2f}'.format(sum_loss), end=" ")
    print('{0:.4f}'.format(m1_mse), end=" ")
    print('{0:.4f}'.format(m2_mse), end=" ")
    print('{0:.4f}'.format(m3_mse), end=" ")
    print('{0:.4f}'.format(m_total_mse), end=" ")
    print('{0:.2f}'.format(m1_val_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m2_val_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m3_val_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m_total_val_accuracy * 100.0), end=" ")
    print('{0:.2f}'.format(m1_val_loss), end=" ")
    print('{0:.2f}'.format(m2_val_loss), end=" ")
    print('{0:.2f}'.format(m3_val_loss), end=" ")
    print('{0:.2f}'.format(m_total_val_loss), end=" ")
    print('{0:.2f}'.format(sum_val_loss), end=" ")
    print('{0:.4f}'.format(m1_val_mse), end=" ")
    print('{0:.4f}'.format(m2_val_mse), end=" ")
    print('{0:.4f}'.format(m3_val_mse), end=" ")
    print('{0:.4f}'.format(m_total_val_mse), end=" ")


    m1_loss_test, m1_accuracy_test, m1_mse_test = m1.evaluate(x_test, y_test, verbose=0)
    predictions1 = m1.predict(x_test)

    y_pred1 = []
    for i in predictions1:
        y_pred1 = np.append(y_pred1, int(np.argmax(i)))
    f1 = precision_recall_fscore_support(y_test_array, y_pred1, average='macro')

    (total_loss_test, m2_loss_kld_test, m3_loss_ce_test, m_total_loss_ce_fin_test, m2_accuracy_test,
    m2_mse_test, m3_accuracy_test, m3_mse_test, m_accuracy_test, m_total_mse_test) = architecture.evaluate(x_unprivileged_test, y_test, verbose=0)
    (predictions2, predictions3, predictions_total)= architecture.predict(x_unprivileged_test)

    y_pred2 = []
    for i in predictions2:
        y_pred2 = np.append(y_pred2, int(np.argmax(i)))
    f2 = precision_recall_fscore_support(y_test_array, y_pred2, average='macro')

    y_pred3 = []
    for i in predictions3:
        y_pred3 = np.append(y_pred3, int(np.argmax(i)))
    f3 = precision_recall_fscore_support(y_test_array, y_pred3, average='macro')

    y_pred_fin = []
    for i in predictions_total:
        y_pred_fin = np.append(y_pred_fin, int(np.argmax(i)))
    f_final = precision_recall_fscore_support(y_test_array, y_pred_fin, average='macro')
    test_accuracy_final = accuracy_score(y_test_array, y_pred_fin)

    #print(set(y_test_array) - set(y_pred1))
    #print(set(y_test_array) - set(y_pred2))
    #print(set(y_test_array) - set(y_pred3))
    #print(set(y_test_array) - set(y_pred_fin))

    plot("Loss", ["loss", "m2_output_loss", "m3_output_loss", "m_total_loss"])
    plot("Accuracy", ["m2_output_accuracy", "m3_output_accuracy", "m_total_accuracy"])
    plot("MSE", ["m2_output_mse", "m3_output_mse", "m_total_mse"])
    plot("Val_Loss", ["val_loss", "val_m2_output_loss", "val_m3_output_loss", "val_m_total_loss"])
    plot("Val_Accuracy", ["val_m2_output_accuracy", "val_m3_output_accuracy", "val_m_total_accuracy"])
    plot("Val_MSE", ["val_m2_output_mse", "val_m3_output_mse", "val_m_total_mse"])

    str1 = str(f1)
    str1 = str1[1:-1].split(',')
    str2 = str(f2)
    str2 = str2[1:-1].split(',')
    str3 = str(f3)
    str3 = str3[1:-1].split(',')
    str_final = str(f_final)
    str_final = str_final[1:-1].split(',')

    print('{0:.2f}'.format(m1_accuracy_test * 100.0), end=" ")
    print('{0:.2f}'.format(m2_accuracy_test * 100.0), end=" ")
    print('{0:.2f}'.format(m3_accuracy_test * 100.0), end=" ")
    print('{0:.2f}'.format(test_accuracy_final * 100.0), end=" ")
    print('{0:.2f}'.format(m1_loss_test), end=" ")
    print('{0:.2f}'.format(m2_loss_kld_test), end=" ")
    print('{0:.2f}'.format(m3_loss_ce_test), end=" ")
    print('{0:.2f}'.format(m_total_loss_ce_fin_test), end=" ")
    print('{0:.2f}'.format(total_loss_test), end=" ")
    print('{0:.4f}'.format(m1_mse_test), end=" ")
    print('{0:.4f}'.format(m2_mse_test), end=" ")
    print('{0:.4f}'.format(m3_mse_test), end=" ")
    print('{0:.4f}'.format(m_total_mse_test), end=" ")
    print('{0:.2f}'.format(float(str1[0])), end=" ")
    print('{0:.2f}'.format(float(str2[0])), end=" ")
    print('{0:.2f}'.format(float(str3[0])), end=" ")
    print('{0:.2f}'.format(float(str_final[0])), end=" ")
    print('{0:.2f}'.format(float(str1[1])), end=" ")
    print('{0:.2f}'.format(float(str2[1])), end=" ")
    print('{0:.2f}'.format(float(str3[1])), end=" ")
    print('{0:.2f}'.format(float(str_final[1])), end=" ")
    print('{0:.2f}'.format(float(str1[2])), end=" ")
    print('{0:.2f}'.format(float(str2[2])), end=" ")
    print('{0:.2f}'.format(float(str3[2])), end=" ")
    print('{0:.2f}'.format(float(str_final[2])), end=" ")