# import the necessary packages
import tensorflow as tf
import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt


DATASET = 'COIL20.mat'
info_privileged = 'zero_nothingelse_indices'
PERCENTAGE_TEST = 0.3
EPOCHS = 8

def read_privileged_indices(file):
    array=[]
    c=''
    with open(file) as f:
        line = f.readline()
        for el in line:
            if el.isnumeric():
                c = c + el
            if el == ',':
                array.append(int(c))
                c = ''
        array.append(int(c))
    return array


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

def split_indices(array, n_features):
    unprivileged_indices = []
    for num in range(n_features):
        if num not in array:
            unprivileged_indices.append(num)
    return unprivileged_indices


def multi_layer_perceptron(n_features, classes, finalAct):
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
    output_ = layers.Activation(finalAct)(model)

    model = models.Model(input_, output_, name = "Model_1_total_knowledge")
    #model.summary()
    return model

def mlp(inputs, classes, finalAct, output):
    """
    Dense multi layer perceptron with 4 layers (1 input, 2 hidden, 1 output)
    :param n_features: size of the argument of the first layer
    :return: multi layer perceptron
    """
    model = layers.Dense(units=256)(inputs)
    model = layers.Activation('relu')(model)
    model = layers.Dense(units=192)(model)
    model = layers.Activation('relu')(model)
    model = layers.Dense(units=128)(model)
    model = layers.Activation('relu')(model)
    last = layers.Dense(units=classes)(model)
    output = layers.Activation(finalAct, name=output)(last)

    # The build function will create the model so it is not necessary to run the following operation
    #model = models.Model(inputs, output)

    return output

def build(n_features_m2m3, classes, finalAct='softmax'):
    """
    Building

    :param n_features_m2m3: size of the argument for the first layer
    :param classes: size of the expected output
    :param finalAct: function to use on the last layer
    :return: a built net with two branches, m2 and m3
    """
    # Constructing both the "m2" and "m3" sub-networks
    inputs = layers.Input(shape=n_features_m2m3)
    m2branch = mlp(inputs, classes, finalAct=finalAct, output="m2_output")
    m3branch = mlp(inputs, classes, finalAct=finalAct, output="m3_output")

    # Creating the model using the input and two separate outputs -- one for m2 branch
    # and another for the m3 branch, respectively
    modelm2m3 = models.Model(inputs, [m2branch, m3branch], name="Model_2_3_unprivileged")

    # Return the constructed network architecture
    #modelm2m3.summary()
    return modelm2m3


if __name__ == "__main__":
    mat = scipy.io.loadmat('data\\' + DATASET)

    X = mat['X']  # data
    y = mat['Y']  # label

    # one hot encoding
    ohe = OneHotEncoder(sparse=False)
    Y = ohe.fit_transform(y)

    classes = len(ohe.categories_[0])

    n_samples, n_features = X.shape  # number of samples and number of features
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=PERCENTAGE_TEST)  # Split data into train and test

    privileged_indices = read_privileged_indices('privileged_selected_indices\\'+info_privileged +'.txt') #read privileged indices already established
    unprivileged_indices = split_indices(privileged_indices, n_features)

    p_indices = [privileged_indices, unprivileged_indices]

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

    ####################################################################################################################
    print("\n######### TRAINING ########")
    ### MODEL 1
    m1 = multi_layer_perceptron(n_features, classes, 'softmax')
    adam = optimizers.Adam(lr=0.0001)

    # Compiling model 1
    #print("\n[INFO] Compiling model 1")
    m1.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])

    # Training model 1
    #print("\n[INFO] Training model 1")
    history1 = m1.fit(x_train, y_train, epochs=10, verbose=0)

    # Saving the model to disk
    #print("\n[INFO] Serializing network...")
    m1.save(os.getcwd() + '\\run\\run1\\saved_models\\'+info_privileged+'\\model_1', save_format="h5")

    # Distribution to use in m2 to let it learn from m1
    privileged_dist = m1.predict(x_train)
    y_prediction = []
    for i in privileged_dist:
        y_prediction = np.append(y_prediction, int(np.argmax(i)))
    y_prediction = y_prediction[:, None]
    y_train_privileged = ohe.fit_transform(y_prediction)
    y_train_privileged = tf.convert_to_tensor(y_train_privileged)
    ########################################## MODEL 2 MODEL 3 ###########################################

    #PERCENTAGE_PRIVILEGED = int(len(privileged_indices) * 100 / len(x_train))

    # Initializing the multi-output network
    n_features_m2m3 = (n_features - len(privileged_indices))
    model_2_3 = build(n_features_m2m3, classes=classes, finalAct='softmax')

    # Defining two dictionaries: one that specifies the loss method for
    # each output of the network along with a second dictionary that
    # specifies the weight per loss
    losses = {
        "m2_output": "kullback_leibler_divergence", "m3_output": "categorical_crossentropy",
    }
    lossWeights = {"m2_output": 1.0, "m3_output": 1.0}

    # Initializing the optimizer and compiling the model
    #print("\n[INFO] Compiling model 2 and 3...")
    opt = optimizers.Adam(lr=0.0001)
    model_2_3.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
                      metrics=["accuracy"])

    # Training model 2 and 3
    #print("\n[INFO] Training model 2 and 3 ")
    # train the network to perform multi-output classification
    H = model_2_3.fit(x=x_unprivileged_train,
                      y={"m2_output": y_train_privileged, "m3_output": y_train},
                      epochs=EPOCHS,
                      verbose=0)

    # Saving the model to disk
    #print("\n[INFO] Serializing network...")
    model_2_3.save(os.getcwd() + '\\run\\run1\\saved_models\\'+info_privileged+'\\model_2_3', save_format="h5")

    #print("\n[INFO] Printing training values for M1 M2 M3")
    # Print Train values Model 1
    ce_1 = history1.history['loss'][9]
    acc_1 = history1.history['accuracy'][9]

    # Printing Train values Model 2 and 3
    total_loss = H.history['loss'][EPOCHS - 1]
    m2_loss = H.history['m2_output_loss'][EPOCHS - 1]
    m3_loss = H.history['m3_output_loss'][EPOCHS - 1]

    m2_accuracy = H.history['m2_output_accuracy'][EPOCHS - 1]
    m3_accuracy = H.history['m3_output_accuracy'][EPOCHS - 1]

    print('Train_m1_accuracy {0:.2f}%'.format(acc_1 * 100.0))
    print('Train_m2_accuracy {0:.2f}%'.format(m2_accuracy * 100.0))
    print('Train_m3_accuracy {0:.2f}%\n'.format(m3_accuracy * 100.0))

    print('Train_m1_loss {0:.2f}'.format(ce_1))
    print('Train_m2_loss {0:.2f}'.format(m2_loss))
    print('Train_m3_loss {0:.2f}'.format(m3_loss))

    print("\n######### TESTING ########")
    # evaluation and prediction model 1
    #print("\nTEST MODEL 1")
    test_loss1, test_accuracy1 = m1.evaluate(x_test, y_test, verbose=0)
    predictions1 = m1.predict(x_test)

    y_pred1 = []
    for i in predictions1:
        y_pred1 = np.append(y_pred1, int(np.argmax(i)))
    y_test1 = np.argwhere(y_test != 0)[:, 1]
    f1 = precision_recall_fscore_support(y_test1, y_pred1, average='macro')
    str1 = str(f1)
    str1 = str1[1:-1].split(',')

    #print("\nTEST MODEL 2 AND MODEL 3 --- UNPRIVILEGED SECTION")
    (total_output_loss, m2_output_loss_kld, m3_output_loss_ce, m2_output_accuracy,
     m3_output_accuracy) = model_2_3.evaluate(x_unprivileged_test, (y_test, y_test), verbose=0)
    (predictions2, predictions3) = model_2_3.predict(x_unprivileged_test)

    # print("\nModel 2 values")
    y_pred2 = []
    for i in predictions2:
        y_pred2 = np.append(y_pred2, int(np.argmax(i)))
    y_test2 = np.argwhere(y_test != 0)[:, 1]
    f2 = precision_recall_fscore_support(y_test2, y_pred2, average='macro')
    str2 = str(f2)
    str2 = str2[1:-1].split(',')

    #print("\nModel 3 values")
    y_pred3 = []
    for i in predictions3:
        y_pred3 = np.append(y_pred3, int(np.argmax(i)))
    y_test3 = np.argwhere(y_test != 0)[:, 1]
    f3 = precision_recall_fscore_support(y_test3, y_pred3, average='macro')
    str3 = str(f3)
    str3 = str3[1:-1].split(',')

    # plot the total loss, m2 loss, and m3 loss
    lossNames = ["loss", "m2_output_loss", "m3_output_loss"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        # plot the loss
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].legend()
    # save the losses figure
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\run\\run1\\saved_images\\'+info_privileged+'\\loss'.format("png"))
    # plt.show()
    plt.close()

    # create a new figure for the accuracies
    accuracyNames = ["m2_output_accuracy", "m3_output_accuracy"]
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
    # loop over the accuracy names
    for (i, l) in enumerate(accuracyNames):
        # plot the loss
        ax[i].set_title("Accuracy for {}".format(l))
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Accuracy")
        ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
        ax[i].legend()
    # save the accuracies figure
    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\run\\run1\\saved_images\\'+info_privileged+'\\accuracy'.format("png"))
    # plt.show()
    plt.close()

    #print("\nFINAL")
    # create a copy of m3 without last layer (activation function), just to get the features of the secondlast layer
    intermediate_layer_m2_m3 = tf.keras.Model(inputs=model_2_3.inputs, outputs=[model_2_3.get_layer("dense_7").output,
                                                                                model_2_3.get_layer("dense_11").output],
                                              name="Intermediate_M2_M3")
    # intermediate_layer_m2_m3.summary()
    features_2_3 = intermediate_layer_m2_m3.predict(x_unprivileged_train)

    sum_features = (features_2_3[0] + features_2_3[1]) / 2

    y_train_fin = layers.Activation('softmax')(sum_features)
    # final cross entropy between y_train --> tensor of true targets
    #                       and   y_train_fin --> tensor of predicted targets
    ce_final = tf.reduce_mean(tf.losses.categorical_crossentropy(y_train, y_train_fin)).numpy()


    features_2_3_test = intermediate_layer_m2_m3.predict(x_unprivileged_test)
    sum_features_test = (features_2_3_test[0] + features_2_3_test[1]) / 2

    y_test_fin = layers.Activation('softmax')(sum_features_test)

    # Calculate and print Test metrics architecture
    y_pred_fin = []
    for i in y_test_fin:
        y_pred_fin = np.append(y_pred_fin, int(np.argmax(i)))
    f_final = precision_recall_fscore_support(y_test_array, y_pred_fin, average='macro')
    test_accuracy_final = accuracy_score(y_test_array, y_pred_fin)

    str_final = str(f_final)
    str_final = str_final[1:-1].split(',')

    ############ PRINTING ###################
    print('Test_m1_accuracy {0:.2f}%'.format(test_accuracy1 * 100.0))
    print('Test_m2_accuracy {0:.2f}%'.format(m2_output_accuracy * 100.0))
    print('Test_m3_accuracy {0:.2f}%'.format(m3_output_accuracy * 100.0))
    print('Test_accuracy {0:.2f}%\n'.format(test_accuracy_final * 100.0))

    print('Test_m1_loss {0:.2f}'.format(test_loss1))
    print('Test_m2_loss {0:.2f}'.format(m2_output_loss_kld))
    print('Test_m3_loss {0:.2f}'.format(m3_output_loss_ce))
    print('Test_total_loss {0:.2f}\n'.format(total_output_loss))

    print('precision_m1 {0:.2f}'.format(float(str1[0])))
    print('precision_m2 {0:.2f}'.format(float(str2[0])))
    print('precision_m3 {0:.2f}'.format(float(str3[0])))
    print('precision_final {0:.2f}\n'.format(float(str_final[0])))

    print('recall_m1 {0:.2f}'.format(float(str1[1])))
    print('recall_m2 {0:.2f}'.format(float(str2[1])))
    print('recall_m3 {0:.2f}'.format(float(str3[1])))
    print('recall_final {0:.2f}\n'.format(float(str_final[1])))

    print('fbeta_score_m1 {0:.2f}'.format(float(str1[2])))
    print('fbeta_score_m2 {0:.2f}'.format(float(str2[2])))
    print('fbeta_score_m3 {0:.2f}'.format(float(str3[2])))
    print('fbeta_score_final {0:.2f}\n'.format(float(str_final[2])))

    # print final loss
    print('Cross_entropy_final {0:.2f}'.format(ce_final))
