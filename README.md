#LeNet

Handcraft convolutional NN with LeNet. Although there are several innovative tricks and methods which are proposed after LeNet, but it is still classic for every ML/DL engineers. I'll try to implement the original algorithm with LeNet which is proposed by Yann LeCun on 1998. [The Paper of LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

#The detail of LeNet:

Data Pre-processing:
1. the size of input image = 28x28
2. padding zeros with image = 32x32
3. normalize the value of input image to -0.1~1.175

Architecture of CNN:
1. conv5x5 with 6 filters, activation = 1.7159 * tanh(2/3 * x)
2. average pooling, applied weights and biases on each channel, activation = 1.7159 * tanh(2/3 * x)
3. conv5x5 with 16 filters, activation = 1.7159 * tanh(2/3 * x), but the inputs are combination of the previous layer's outputs, as below:

1 X - - - X X X - - X X X X - X X

2 X X - - - X X X - - X X X X - X

3 X X X - - - X X X - - X - X X X

4 - X X X - - X X X X - - X - X X

5 - - X X X - - X X X X - X X - X

6 - - - X X X - - X X X X - X X X

4. average pooling, applied weights and biases on each channel, activation = 1.7159 * tanh(2/3 * x)
5. conv5x5 with 120 filters, activation = 1.7159 * tanh(2/3 * x)
6. fully with 84 filters, activation = 1.7159 * tanh(2/3 * x)
7. RBF layer (loss = 0.5 * (labels_P - bitmap[labels_GT])^2

Conv, weights initialize with (-2.4/Fi ~ 2.4/Fi), Fi is the number of neurons in previous layer in a neuron of next layer.
Ex. conv5x5 with 6 filters, input's channel = 3, Fi = 5x5x3.
Fully, weights initialize with (normal distribution * sqrt(2/Fi)). Refer to Xavier initializer.

20 epochs, learning rate are [0.0005]*2  [0.0002]*3  [0.0001]*3  [0.00005]*4  [0.00001]*8,
but before training in each epochs, we will utilize Stochastic Diagonal Levenberg-Marquaedt with 500 training data to refine the learning rate in each layer.

#Result

<img src="https://github.com/AlgorithmicIntelligence/LeNet/blob/master/Accuracy.png" width="900">

##Reference

1. [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
2. [mattwang44 repo](https://github.com/mattwang44/LeNet-from-Scratch)
