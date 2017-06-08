class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        # def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        # self.activation_function = sigmoid
        self.derivative_function = lambda x: self.activation_function(x) * (1 - self.activation_function(x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            whoc = self.weights_hidden_to_output[:, 0]
            yc = np.asscalar(y)

            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)  # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.dot(hidden_outputs, whoc)  # signals into final output layer
            final_outputs = self.activation_function(final_inputs)  # signals from final output layer

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = yc - final_outputs  # Output layer error is the difference between desired target and actual output.

            output_error_term = error * self.derivative_function(final_outputs)

            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(output_error_term, whoc)

            # TODO: Backpropagated error terms - Replace these values with your calculations.
            hidden_error_term = hidden_error * self.derivative_function(hidden_outputs)

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:, None]

            # Weight step (hidden to output)
            print("shape of output_error_term: ", output_error_term.shape, " value: ", output_error_term)
            print(type(output_error_term))
            print("shape of hidden_outputs: ", hidden_outputs.shape, " value: ", hidden_outputs)

            delta_weights_h_o += output_error_term[:, None] * hidden_outputs

            print("delta_weights_h_o: ", delta_weights_h_o.shape, " value: ", delta_weights_h_o)

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.weights_hidden_to_output  # self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.weights_input_to_hidden  # self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = self.activation_function(final_inputs)  # signals from final output layer

        return final_outputs