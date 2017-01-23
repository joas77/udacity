from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

import numpy as np

# Pad images with 0s
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

import random
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])
#plt.show()

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten


def LeNet(x):
    # x is our input image 32x32x1-  32 height, 32 width, 1 channel
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    print("1a: LaNet Net Input tensor:", x , "well it is an image 32x32 pixels and 1 channel")

    """
    filter shape is the patch we move through image - in this section our goal is to setup filter size that will
    generate the output tensor to be = 28x28x6 where 6 is number of filters (it will be blur, edge detector, dilate or
    similar convolutional filters

    Don't think to much now. Just simply try to apply proper filter for tensor that after using tf.nn.conv2d will
    give you proper output size

    According to documentation:
    For the 'VALID' padding, the output height and width are computed as:

    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

    padding pixels are computed. For the 'SAME' padding, the output height and width are computed as:

    out_height = ceil(float(in_height) / float(strides[1]))
    out_width  = ceil(float(in_width) / float(strides[2]))


    As we see SAME depends only from stride size so it will be hard to get 28 from 32 dividing only by int numbers ;)
    So now we know we will use 'VALID' padding
    """

    filter_shape = [5, 5, 1, 6]
    strides = [1, 1, 1, 1]
    print("1b: filter shape ", filter_shape)
    """
    in filter shape according to documentation filter shape structrure is:
    [filter_height, filter_width, in_channels, out_channels]
    we want to have 6 channels on out so last number is for sure 6
    our input number of channels is 1 so 3rd number will be 1
    first two numbers is size of filter calculate from 'VALID' equation above
    """
    print("1c: strades array ", strides)




    """
    Because we want that NN compute for us new values of filters we need tu put it inside tf.Variable with shape we
    defined before

    """

    layer1_W = tf.Variable(tf.truncated_normal(shape=filter_shape))
    print("1d:  size of the filter as tensor ", layer1_W)

    layer1_bias = tf.Variable(tf.zeros(6))
    print("1e bias size should mathch  tensor with lenght size equal to number of new layers(deep) - 6", layer1_bias)


    layer1 = tf.nn.conv2d(x,layer1_W,strides=strides,padding='VALID') + layer1_bias

    print("1f if everything was setup properly our empty tensor should look something like shape=(?, 28, 28, 6)",layer1)


    """
    we only need to wrap our current network in relu tensor
    """
    layer1 = tf.nn.relu(layer1)
    print("1g our CNN tensor wraped in relu: ", layer1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    """
    save memory. Use Pooling, Don't try to understand why! NN are a little like magic and no one knows why some tricks
    works and some not, just refer to LaNET experience and implement it, this task is about it!

    In this point we want to resize our NN output to size 14x14x6, so deep is still the same (6) and it's quite easy
    to see that input is TWO times bigger than output so output should be divded by 2 in Pooling operation.

    According to documentation: tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    output[i] = reduce(value[strides * i:strides * i + ksize])

    The tf.nn.max_pool() function performs max pooling with the ksize parameter as the size of the filter and the
    strides parameter as the length of the stride. 2x2 filters with a stride of 2x2 are common in practice.

    The ksize and strides parameters are structured as 4-element lists, with each element corresponding to a dimension
    of the input tensor ([batch, height, width, channels]). For both ksize and strides, the batch and channel dimensions
    are typically set to 1


    value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32. - this is place for our previous
    output (Layer1)



    """
    ksize = [1,2,2,1]
    strides = [1,2,2,1]
    pool = tf.nn.max_pool(layer1, ksize=ksize, strides=strides, padding='VALID')
    print("layer after Pool", pool)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    """
    We already did it once but now input and output size of NN is different
    from 14x14x6 to 10x10x16

    """
    filter_shape = [5, 5, 6 , 16] #[filter_height, filter_width, in_channels, out_channels]
    strides = [1, 1, 1, 1]

    layer2_W = tf.Variable(tf.truncated_normal(shape=filter_shape))
    print("2a:  size of the filter as tensor ", layer2_W)

    layer2_bias = tf.Variable(tf.zeros(16))
    print("2b: bias size should mathch  tensor with lenght size equal to number of new layers(deep) - 16", layer2_bias)
    layer2 = tf.nn.conv2d(pool, layer2_W, strides=strides, padding='VALID') + layer2_bias



    # TODO: Activation.
    layer2 = tf.nn.relu(layer2)
    print("layer2 after conv2d",layer2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    #exactly the same but on deeper layer
    ksize = [1,2,2,1]
    strides = [1,2,2,1]
    pool2 = tf.nn.max_pool(layer2, ksize=ksize, strides=strides, padding='VALID')
    print("layer after Pool2", pool2)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    """
    Now we  want( sorry, not we, but- LaNet algorithm) to use classic NN to do so, we need flatten it
    There is one tensorflow algorithm to do this  - flatten, fortunatly we don't have to reinwent the wheel
    from tensorflow.contrib.layers import flatten
    """
    flatten1 = flatten(pool2)
    print("Tenssor after flatten",flatten1)



    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    """
    Going back to classic again use regullar tensor people from LaNet has find out that it will work cool
    if we have 120 tensors on the next layer. Black magic!
    """
    number_of_input_tensors = 400
    number_of_output_tensors = 120
    layer3_weight = tf.Variable(tf.truncated_normal(shape=(number_of_input_tensors,number_of_output_tensors)))
    layer3_bias = tf.Variable(tf.zeros(number_of_output_tensors))

    layer3 = tf.matmul(flatten1,layer3_weight) + layer3_bias
    print("layer3:" ,layer3)




    # TODO: Activation.
    layer3 = tf.nn.relu(layer3)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    # There is a magic creature that tells me that it will be great if we create next layers with 84 neurons on the end
    number_of_input_tensors = 120
    number_of_output_tensors = 84
    layer4_weight = tf.Variable(tf.truncated_normal(shape=(number_of_input_tensors, number_of_output_tensors)))
    layer4_bias = tf.Variable(tf.zeros(number_of_output_tensors))

    layer4 = tf.matmul(layer3, layer4_weight) + layer4_bias



    # TODO: Activation.

    layer4 = tf.nn.relu(layer4)

    print("layer4:", layer4)


    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    #FINALLY There are 10 digits so we need 10 classes on the end to do proper training
    number_of_input_tensors = 84
    number_of_output_tensors = 10
    logits_weight = tf.Variable(tf.truncated_normal(shape=(number_of_input_tensors, number_of_output_tensors)))
    logits_bias = tf.Variable(tf.zeros(number_of_output_tensors))

    logits = tf.matmul(layer4,logits_weight) + logits_bias
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    import time
    tic = time.clock()
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    print("Session time {}".format(time.clock() - tic))

    saver.save(sess, 'lenet')
    print("Model saved")
