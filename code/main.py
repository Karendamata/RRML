import keras
import pandas as pd
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score
import psutil
import os
import numpy as np
import random
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

########## variables ############

# epsilon range = {initial, final}
initial = 0.01
final = 0.04


train_loss_history_after = []
train_accuracy_history_after = []

val_loss_history_after = []
val_accuracy_history_after = []

test_loss_history_before = []
test_accuracy_history_before = []

epsilon_history = []
freq_history = []
fc = []
ft = []
mem = []
FT_history = []
pred_history = []
alpha_lst = []
F1score_lst = []
############################################


######## Loading the  Data ###############
print('-----------------Loading the  Data ---------------')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# divide x_train into validation and test
x_val = X_train[40000:]
x_train = X_train[:40000]

y_val = y_train[40000:]
y_train = y_train[:40000]

################ Classifier Definition#################
conv_args = dict(activation=tf.nn.leaky_relu, kernel_size=3, padding="same")
input_img = Input(shape=(32, 32, 3))
x = Conv2D(64, **conv_args)(input_img)
x = Conv2D(128, **conv_args)(x)
x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Conv2D(128,  **conv_args)(x)
x = Conv2D(256,  **conv_args)(x)
x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Conv2D(128,  **conv_args)(x)
x = Conv2D(64, **conv_args)(x)
x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='leaky_relu')(x)
classifier = Dense(11, activation='sigmoid')(x)
classifier_model = Model(inputs=input_img, outputs=classifier)

######## Training the model #######
print('-----------------Training the model---------------')
Loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
alpha = 0.001
opt = keras.optimizers.Adam(learning_rate=alpha)
classifier_model.compile(loss=Loss_object, optimizer=opt, metrics='accuracy')
initial_iter = 3
for i in range(initial_iter):

    pred_loss_test, pred_acc_test = classifier_model.evaluate(X_test, y_test, verbose=2)
    prediction = np.array([np.argmax(p) for p in classifier_model.predict(X_test)])

    F1score_lst.append(f1_score(y_test, prediction, average='weighted'))
    alpha_lst.append(alpha)
    modelFit = classifier_model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

    train_loss_history_after.append(modelFit.history['loss'][0])  # after training
    train_accuracy_history_after.append(modelFit.history['accuracy'][0])

    val_loss_history_after.append(modelFit.history['val_loss'][0])  # after training
    val_accuracy_history_after.append(modelFit.history['val_accuracy'][0])

    test_loss_history_before.append(pred_loss_test)
    test_accuracy_history_before.append(pred_acc_test)



####### Creating the Adversarial Examples#####
Testing_Data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(X_test), y_test))

def take_percentage(fgm, pgd, real, testing_df):
    global fgm_len, pgd_len, real_len, fgm_batches, pgd_batches
    fgm_perc = int(X_test.shape[0] * (fgm / 100))
    pgd_perc = int(X_test.shape[0] * (pgd / 100))
    real_perc = int(X_test.shape[0] * (real / 100))

    testing_fgm = testing_df.shuffle(X_test.shape[0]).take(fgm_perc)
    testing_pgd = testing_df.shuffle(X_test.shape[0]).take(pgd_perc)
    testing_real = testing_df.shuffle(X_test.shape[0]).take(real_perc)

    fgm_len = testing_fgm.cardinality().numpy()
    pgd_len = testing_pgd.cardinality().numpy()
    real_len = testing_real.cardinality().numpy()
    testing_fgm, testing_pgd, testing_real = testing_fgm.batch(batch_size=128), testing_pgd.batch(batch_size=128), testing_real.batch(batch_size=128)
    fgm_batches = testing_fgm.cardinality().numpy()
    pgd_batches = testing_pgd.cardinality().numpy()

    return testing_fgm, testing_pgd, testing_real


def generate_adversary(eps, testing_fgm, testing_pgd, testing_real):
    X_real = tf.zeros([0, 32, 32, 3], dtype=float)
    Y_real = tf.zeros([0, 1], dtype='uint8')

    X_fgm = tf.zeros([0, 32, 32, 3], dtype=float)
    Y_fgm = tf.convert_to_tensor(np.array([[x] for x in np.ones(fgm_len) * 10]), dtype='uint8')

    X_pgd = tf.zeros([0, 32, 32, 3], dtype=float)
    Y_pgd = tf.convert_to_tensor(np.array([[x] for x in np.ones(pgd_len) * 10]), dtype='uint8')

    real_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.array([0] * real_len), dtype='int32'))
    fgm_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.array([1] * fgm_len), dtype='int32'))
    pgd_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(np.array([2] * pgd_len), dtype='int32'))

    for x, y in testing_real:
        X_real = tf.concat([X_real, x], 0)
        Y_real = tf.concat([Y_real, y], 0)
    X_real = tf.data.Dataset.from_tensor_slices((X_real))
    Y_real=tf.data.Dataset.from_tensor_slices(Y_real)

    print("Generate Adversary images with epsilon " + str(eps)+ " using FGM:")
    progress_bar_test = tf.keras.utils.Progbar(fgm_batches)
    for x,y in testing_fgm:
        x_fgm = fast_gradient_method(classifier_model, x, eps, np.inf)
        X_fgm = tf.concat([X_fgm, x_fgm], 0)
        progress_bar_test.add(1)

    X_fgm = tf.data.Dataset.from_tensor_slices((X_fgm))
    Y_fgm = tf.data.Dataset.from_tensor_slices(Y_fgm)

    print("Generate Adversary images with epsilon " + str(eps)+ " using PGD:")
    progress_bar_test = tf.keras.utils.Progbar(pgd_batches)
    for x, y in testing_pgd:
        x_pgd = projected_gradient_descent(classifier_model, x, eps, 0.01, 40, np.inf)
        X_pgd = tf.concat([X_pgd, x_pgd], 0)
        progress_bar_test.add(1)
    X_pgd = tf.data.Dataset.from_tensor_slices((X_pgd))
    Y_pgd = tf.data.Dataset.from_tensor_slices(Y_pgd)

    real_df = tf.data.Dataset.zip((X_real, Y_real, real_label))
    fgm_df = tf.data.Dataset.zip((X_fgm, Y_fgm, fgm_label))
    pgd_df = tf.data.Dataset.zip((X_pgd, Y_pgd, pgd_label))

    test_df = fgm_df.concatenate(pgd_df)
    test_df = test_df.concatenate(real_df)

    return test_df.shuffle(10000)


# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


# decorator function
def track_mem(func):
    def wrapper(*args, **kwargs):
        global mem
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
        mem.append(mem_after - mem_before)
        return result
    return wrapper


@track_mem
def optimize():
    num_imgs = len(images)
    failure_times = np.array([])
    opt = keras.optimizers.Adam(learning_rate=alpha)
    pred_HR = []
    progress_bar_train = tf.keras.utils.Progbar(num_imgs)
    for img, img_class, img_label, time in [[images[i].reshape(1, 32, 32, 3), Class[i][0], label[i], i] for i in
                                            range(num_imgs)]:
        prediction = np.argmax(classifier_model(img))
        pred_HR.append([prediction, img_class, img_label])
        progress_bar_train.add(1)
        if (prediction != 10) and img_label:
            failure_times = np.append(failure_times, time)
            ft.append(time + 10000 * n)

    pred_history.append(pred_HR)
    FT_history.append(failure_times)

    # Before Training
    pred_loss_test, pred_acc_test = classifier_model.evaluate(images, Class, verbose=2)

    classifier_model.compile(loss=Loss_object, optimizer=opt, metrics='accuracy')
    history = classifier_model.fit(images_train, Class_train, epochs=4, validation_data=(images_val, Class_val))
    F1score_lst.append(f1_score([p[0] for p in pred_HR],[p[1] for p in pred_HR], average='weighted'))

    # After Training
    train_loss_history_after.append(history.history['loss'][0])  # after training
    train_accuracy_history_after.append(history.history['accuracy'][0])

    val_loss_history_after.append(history.history['val_loss'][0])  # after training
    val_accuracy_history_after.append(history.history['val_accuracy'][0])

    test_loss_history_before.append(pred_loss_test)
    test_accuracy_history_before.append(pred_acc_test)
    fc.append(len(failure_times))


################# Online Training ##############
print('-----------------Online Training ---------------')
loss_metric = tf.keras.metrics.Mean(name='train_loss')
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
for n in range(30):
    if (n % 5) == 0:
        alpha /= 10
    print('--------------------Iteration: {}--------------'.format(n+1))
    alpha_lst.append(alpha)
    epsilon = round(random.uniform(initial,final),4)
    epsilon_history.append(epsilon)

    real_per = 50
    fgm_per = random.randint(1,50)
    pgd_per = 50-fgm_per

    Test_fgm, Test_pgd, Test_real = take_percentage(fgm_per, pgd_per, real_per, Testing_Data)
    combined_test = generate_adversary(epsilon, Test_fgm, Test_pgd, Test_real)
    freq_history.append([fgm_per, pgd_per])
    images, Class, label = tuple(zip(*combined_test))

    # dividing combined_test into training, validation and test
    images_train = np.array(images[:5000])
    Class_train = np.array(Class[:5000])
    label_train = np.array(label[:5000])

    images_val = np.array(images[5000:])
    Class_val = np.array(Class[5000:])
    label_val = np.array(label[5000:])

    print("FGM: {} , PGD : {} , Real: {}".format(fgm_per , pgd_per , real_per))
    images = np.array(images)
    label = np.array(label)
    Class = np.array(Class)
    #############
    loss_metric.reset_states()
    accuracy_metric.reset_states()
    optimize()


############# Exporting #################
Fc = list(np.zeros(initial_iter))
Ft = list(np.zeros(initial_iter))
Epsilon = list(np.zeros(initial_iter))
Fgsm = list(np.zeros(initial_iter))
Pgd = list(np.zeros(initial_iter))
Memory = list(np.zeros(initial_iter))

Dict = {"T": range(1,len(train_accuracy_history_after)+1),
        "FC": Fc+fc,
        "Alpha": alpha_lst,
        "F1_Score": F1score_lst,
        "Epsilon": Epsilon+epsilon_history,
        "FGSM": Fgsm+list(np.array(freq_history)[:,0]),
        "PGD": Pgd+list(np.array(freq_history)[:,1]),
        "Train_Accuracy": train_accuracy_history_after,
        "Train_Loss": train_loss_history_after,
        "Val_Accuracy": val_accuracy_history_after,
        "Val_Loss": val_loss_history_after,
        "Test_Accuracy": test_accuracy_history_before,
        "Test_Loss": test_loss_history_before,
        "Memory": Memory+[m / 1024**2 for m in mem]}


pwd = os.getcwd()

if not os.path.exists(pwd+'/Data'):
    os.mkdir(pwd+'/Data')

# failure count data + covariates
FC_DF = pd.DataFrame.from_dict(data=Dict).set_index("T")
FC_DF.to_csv(pwd+'/Data/'+str(initial)+'-'+str(final)+'-FC-Results.csv')

# failure time data
Ft_DF = pd.DataFrame(data=Ft+ft)
Ft_DF = Ft_DF.reset_index()
Ft_DF.columns = ["FN", "FT"]
Ft_DF = Ft_DF.set_index("FN")
Ft_DF.to_csv(pwd+'/Data/'+str(initial)+'-'+str(final)+'-FT-Results.csv')
