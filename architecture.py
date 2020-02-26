import tensorflow as tf
import random
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import datasets, layers, models, optimizers
#import matplotlib.pyplot as plt

DATASET = 'COIL20.mat'
PERCENTAGE_PRIVILEGED = 20
PERCENTAGE_TRAIN = 0.3


def split_knowledge_set(X, privileged_indexes, boolean):
    """
    Get the set of the desired knowledge, privileged or unprivileged
    :param X: dataset to split into privileged and unprivileged knowledge
    :param privileged_indexes: indexes randomly chosen to become the privileged one
    :param boolean: true => select privileged information
                    false => select unprivileged information
    :return: ndarray with desired information
    """
    if (boolean):
        return X[:, privileged_indexes[0]]
    else:
        return X[:, privileged_indexes[1]]

def split_indexes(n_features):
    """
    Provide a list of two arrays, the first contains privileged indexes and the second the unprivileged ones.
    The size of the two arrays is given by the PERCENTAGE of privileged knowledge.
    :param n_features: size of the array you want to split
    :return: list [[privileged_indexes],[uprivileged_indexes]]
    """
    indexes = list(range(n_features))
    random.shuffle(indexes)
    split_index = int(float(PERCENTAGE_PRIVILEGED) / 100 * n_features)
    return [indexes[:split_index], indexes[split_index:]]


def save_list_indexes(indexes):
    """
    Saves privileged and unprivileged indexes in two different files
    :param indexes: list of indexes. In indexes[0] --> privileged indexes. In indexes[1] --> unprivileged indexes.
    """
    file = open("privileged_indexes.txt", "w")
    file.write(str(indexes[0]))
    file.close()
    file = open("unprivileged_indexes.txt", "w")
    file.write(str(indexes[1]))
    file.close()


def multi_layer_perceptron(n_features):
    """
    Dense multi layer perceptron with 4 layers (1 input, 2 hidden, 1 output)
    :param n_features: size of the argument of the first layer
    :return: multi layer perceptron
    """
    input_ = layers.Input(shape=n_features)
    model = layers.Dense(units=256)(input_)
    model = layers.Activation('relu')(model)
    model = layers.Dense(units=192)(model)
    model = layers.Activation('relu')(model)
    model = layers.Dense(units=128)(model)
    model = layers.Activation('relu')(model)
    model = layers.Dense(units=classes)(model)
    output_ = layers.Activation('softmax')(model)

    model = models.Model(input_, output_)
    #model.summary()
    return model


if __name__ == "__main__":
    mat = scipy.io.loadmat('data\\' + DATASET)

    X = mat['X']  # data
    y = mat['Y']  # label

    # one hot encoding
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(y)

    classes = len(ohe.categories_[0])

    n_samples, n_features = X.shape  # number of samples and number of features
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)  # Split data into train and test

    p_indexes = split_indexes(n_features)

    #save_list_indexes(p_indexes)

    x_unprivileged_train = split_knowledge_set(x_train, p_indexes, False)  # train set with unprivileged knowledge
    x_unprivileged_test = split_knowledge_set(x_test, p_indexes, False)  # test set with unpriviliged knowledge

    # convert numpy to tensors
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)
    x_unprivileged_train = tf.convert_to_tensor(x_unprivileged_train)
    x_unprivileged_test = tf.convert_to_tensor(x_unprivileged_test)

    y_test_array = np.argwhere(y_test != 0)[:, 1] #to use during test for model 1, 2 and 3


    ### MODEL 1
    print("\nMODEL 1")
    m1 = multi_layer_perceptron(n_features)
    adam = optimizers.Adam(lr=0.0001)
    m1.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])
    history1 = m1.fit(x_train, y_train, epochs=10, verbose=0)#, steps_per_epoch=500)

    #distribution to use in m2 to let it learn from m1
    privileged_dist = m1.predict(x_train)

    # Print Train metrics Model 1
    ce_1 = history1.history['loss'][9]
    acc_1 = history1.history['accuracy'][9]
    print('Train loss = {0:.2f}'.format(ce_1))
    print('Train accuracy = {0:.2f}%'.format(acc_1 * 100.0))

    # Calculate and print Test metrics Model 1
    print("\nTEST MODEL 1")
    test_loss1, test_accuracy1 = m1.evaluate(x_test, y_test, verbose=0)
    print('Test loss = {0:.2f}'.format(test_loss1))
    print('Test accuracy = {0:.2f}%'.format(test_accuracy1 * 100.0))
    predictions1 = m1.predict(x_test)

    y_pred1 = []
    for i in predictions1:
        y_pred1 = np.append(y_pred1, int(np.argmax(i)))
    f1 = precision_recall_fscore_support(y_test_array, y_pred1, average='macro')
    print('precision, recall, fbeta_score, support' + str(f1))


    ### MODEL 2
    print("\nMODEL 2")
    m2 = multi_layer_perceptron(n_features - int(n_features * PERCENTAGE_PRIVILEGED / 100))
    adam = optimizers.Adam(lr=0.0001)
    m2.compile(adam, loss='kullback_leibler_divergence', metrics=["accuracy"])

    history2 = m2.fit(x_unprivileged_train, privileged_dist, epochs=10, verbose=0)#, steps_per_epoch=500)

    # Print Train metrics Model 2
    #Kullbach-Leibler divergence to verify how much m2 differs from m1
    kld = history2.history['loss'][9]
    acc_2 = history2.history['accuracy'][9]
    print('Train loss = {0:.2f}'.format(kld))
    print('Train accuracy = {0:.2f}%'.format(acc_2 * 100.0))

    # Calculate and print Test metrics Model 2
    print("\nTEST MODEL 2")
    test_loss2, test_accuracy2 = m2.evaluate(x_unprivileged_test, y_test, verbose=0)
    print('Test loss = {0:.2f}'.format(test_loss2))
    print('Test accuracy = {0:.2f}%'.format(test_accuracy2 * 100.0))
    predictions2 = m2.predict(x_unprivileged_test)

    y_pred2 = []
    for i in predictions2:
        y_pred2 = np.append(y_pred2, int(np.argmax(i)))
    f2 = precision_recall_fscore_support(y_test_array, y_pred2, average='macro')
    print('\nprecision, recall, fbeta_score, support' + str(f2))

    # create a copy of m2 without last layer (activation function), just to get the features of the secondlast layer
    intermediate_layer_m2 = tf.keras.Model(inputs=m2.input, outputs=m2.layers[-2].output)
    features_2 = intermediate_layer_m2.predict(x_unprivileged_train)
    features_2_test = intermediate_layer_m2.predict(x_unprivileged_test)

    ### MODEL 3
    print('\nMODEL 3')
    m3 = multi_layer_perceptron(n_features - int(n_features * PERCENTAGE_PRIVILEGED / 100))
    adam = optimizers.Adam(lr=0.0001)
    m3.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])

    history3 = m3.fit(x_unprivileged_train, y_train, epochs=10, verbose=0)#, steps_per_epoch=500)

    # Print Train metrics Model 3
    ce_3 = history3.history['loss'][9]
    acc_3 = history3.history['accuracy'][9]
    print('Train loss = {0:.2f}'.format(ce_3))
    print('Train accuracy = {0:.2f}%'.format(acc_3 * 100.0))

    # Calculate and print Test metrics Model 3
    print("\nTEST MODEL 3")
    test_loss3, test_accuracy3 = m3.evaluate(x_unprivileged_test, y_test, verbose=0)
    print('Test loss = {0:.2f}'.format(test_loss3))
    print('Test accuracy = {0:.2f}%'.format(test_accuracy3 * 100.0))
    predictions3 = m3.predict(x_unprivileged_test)
    y_pred3 = []
    for i in predictions3:
        y_pred3 = np.append(y_pred3, int(np.argmax(i)))
    f3 = precision_recall_fscore_support(y_test_array, y_pred3, average='macro')
    print('precision, recall, fbeta_score, support' + str(f3))

    # create a copy of m3 without last layer (activation function), just to get the features of the secondlast layer
    intermediate_layer_m3 = tf.keras.Model(inputs=m3.inputs, outputs=m3.layers[-2].output)
    features_3 = intermediate_layer_m3.predict(x_unprivileged_train)
    features_3_test = intermediate_layer_m3.predict(x_unprivileged_test)

    # add features got from model 2 and features got from model 3 before to apply last layer of the net (softmax)
    # and now apply softmax, both to training and to test set
    y_train_fin_feat = layers.Activation('softmax')(features_2 + features_3)
    y_test_fin = layers.Activation('softmax')(features_2_test + features_3_test)

    # final cross entropy between y_train --> tensor of true targets
    #                       and   y_train_fin_feat --> tensor of predicted targets
    ce_final = tf.reduce_mean(tf.losses.categorical_crossentropy(y_train, y_train_fin_feat)).numpy()

    #print final loss
    print('Cross entropy final = {0:.2f}'.format(ce_final))

    # Calculate and print Test metrics architecture
    print("\nFINAL")
    y_pred_fin = []
    for i in y_test_fin:
        y_pred_fin = np.append(y_pred_fin, int(np.argmax(i)))
    f_final = precision_recall_fscore_support(y_test_array, y_pred_fin, average='macro')
    test_accuracy_final = accuracy_score(y_test_array, y_pred_fin)
    print('Test accuracy = {0:.2f}%'.format(test_accuracy_final * 100.0))
    print('precision, recall, fbeta_score, support' + str(f_final))