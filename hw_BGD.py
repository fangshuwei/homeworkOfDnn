from keras.utils import np_utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(10)
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def view_img(x_image, y_id):
    """view the data"""
    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(10):
        img = x_image[y_id == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(
        nrows=5,
        ncols=5,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    for i in range(25):
        img = x_image[y_id == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()



# 取数据
x_train, y_train = load_mnist('./', kind='train')
print(f'Rows: {x_train.shape[0]}, Columns: {x_train.shape[1]}')
x_test, y_test = load_mnist('./', kind='t10k')
print(f'Rows: {x_test.shape[0]}, Columns: {x_test.shape[1]}')



# 转换数据
x_train_v = x_train.reshape(60000,784).astype('float32')
x_test_v = x_test.reshape(10000,784).astype('float32')

#标准化
x_train_norm = x_train_v/255
x_test_norm = x_test_v/255

y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)


# 占位符 placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

learning_rate = 0.01
training_epochs = 2
# for BGD
batch_size = 100
total_batch = int(60000 / batch_size)
display_step = 1

#NetWork parameters
n_hidden_1 = 256#1st layer num features
n_hidden_2 = 256#2nd layer num features
n_input = 784
n_classses = 10

def multilayer_perceptron(_X,_weights,_biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,_weights['h2']),_biases['b2']))
    return tf.matmul(layer2,_weights['out'])+_biases['out']

weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classses]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classses]))
}
pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

save_path = 'model/'
saver = tf.train.Saver()

loss = np.zeros([training_epochs])
pred_n = np.zeros([training_epochs])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x: x_train_norm, y: y_train_onehot})
        # for BGD
        avg_cost = 0
        for i in range(total_batch):
            batch_xs = x_train_norm[i*batch_size:(i+1)*batch_size,]
            batch_ys = y_train_onehot[i*batch_size: (i+1)*batch_size,]
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs, y:batch_ys})
            avg_cost += c / total_batch
        # for SGD
        np.random.shuffle(x_train_norm)

        if (epoch + 1) % display_step == 0:
            print('epoch= ', epoch+1, ' cost= ', c)

        loss[epoch] = avg_cost
    print('finished')

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('accuracy: ', accuracy.eval({x:x_test_norm, y:y_test_onehot}))

    save = saver.save(sess, save_path=save_path+'mnist.cpkt')

file=open('data.txt', 'w+')
st = np.array2string(loss)
file.writelines(st)
file.close()


plt.plot(loss, 'b', label='loss')
plt.title('Training and validation accuracy')
plt.legend()  # 绘制图plt.show()例，默认在右上角
plt.savefig('./BGD.png')
plt.show()