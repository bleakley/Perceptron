# Perceptron
Artificial Neural Network class with Back Error Propagation learning method for MATLAB.
Sigmoid transfer function.

This implementation is simple and effective. Unlike many other excessive implementations with a separate class for a single neuron this code is based on matrix algebra. Neuron layers are essentially vectors and links' weights are nothing else but matrices.
Matrix representation of an artificial neural network makes all methods and calculations elegant and efficient at lower computational cost. The absense of excessive parameters also offers you a user friendly experience.

Methods

1. PERCEPTRON(layers_vector) - creates an instance of PERCEPTRON with specified number of neurons. The layers vector may look like the following [10,12,12,12,5]. A network with the above layers vector would have 10 input sensor neurons, 3 hidden layers of 12 neurons each and 5 output neurons.

2. forward(input_col_vector) - single sample forward calculation method from input to output

3. backprop(input, desired_output, eta) - single sample back error propagation method. The parameter 'eta' controls convergence speed, normally 0 < eta < 1. Typical value eta = 0.001.
