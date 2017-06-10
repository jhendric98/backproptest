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
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(
            hidden_inputs)  # 1/(1+np.exp(-hidden_inputs))# signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs  # Output layer error is the difference between desired target and actual output.

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(error, self.weights_hidden_to_output.T)

        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += learning_rate * (
        delta_weights_h_o / n_records)  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += learning_rate * (
        delta_weights_i_h / n_records)  # update input-to-hidden weights with gradient descent step

        print(self.weights_hidden_to_output)
        print(self.weights_input_to_hidden)
