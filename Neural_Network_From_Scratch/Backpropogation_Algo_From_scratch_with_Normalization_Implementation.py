import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_layers, neurons_per_layer, output_size, activation):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + neurons_per_layer + [output_size]
        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.random.randn(layer_sizes[i+1]))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def activate(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        else:
            raise ValueError("Activation function not supported.")

    def activate_derivative(self, x):
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'tanh':
            return self.tanh_derivative(x)
        else:
            raise ValueError("Activation function not supported.")

    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.activate(z))
        return self.activations[-1]

    def backward_propagation(self, X, y, learning_rate, momentum):
        output = self.forward_propagation(X)
        error = y - output
        deltas = [error * self.activate_derivative(output)]

        for i in range(len(self.weights)-1, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            deltas.append(error * self.activate_derivative(self.activations[i]))

        deltas = deltas[::-1]

        for i in range(len(self.weights)):
            delta_adjusted = deltas[i]  # Adjust delta dimensions
            if delta_adjusted.shape[1] != self.weights[i].shape[1]:
                delta_adjusted = delta_adjusted[:, :self.weights[i].shape[1]]
            self.weights[i] += learning_rate * self.activations[i].T.dot(delta_adjusted) + momentum * self.weights[i]
            self.biases[i] += learning_rate * np.sum(delta_adjusted, axis=0) + momentum * self.biases[i]
            # Clip gradients to avoid exploding gradients
            self.weights[i] = np.clip(self.weights[i], -5, 5)
            self.biases[i] = np.clip(self.biases[i], -5, 5)


    def train(self, X_train, y_train, X_test, y_test, epochs, learning_rate, momentum, batch_size):
        train_errors = []
        test_errors = []
        
        # Splitting the dataset into train and test sets
        train_size = int(0.8 * len(X_train))
        X_train_data, X_val_data = X_train[:train_size], X_train[train_size:]
        y_train_data, y_val_data = y_train[:train_size], y_train[train_size:]
        
        for epoch in range(epochs):
            for i in range(0, len(X_train_data), batch_size):
                X_batch = X_train_data[i:i+batch_size]
                y_batch = y_train_data[i:i+batch_size]
                self.backward_propagation(X_batch, y_batch, learning_rate, momentum)
            
            # Calculate training loss for the epoch
            train_loss = np.mean(np.square(y_train_data - self.forward_propagation(X_train_data)))
            train_errors.append(train_loss)
            
            # Calculate validation loss for the epoch
            val_loss = np.mean(np.square(y_val_data - self.forward_propagation(X_val_data)))
            test_errors.append(val_loss)
            
            # Print epoch and loss
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {val_loss}")

        # Calculate test loss
        test_loss = np.mean(np.square(y_test - self.forward_propagation(X_test)))
        print(f"Test Accuracy: {1 - test_loss}")

        return train_errors, test_errors


    def evaluate(self, X_test, y_test):
        if X_test is not None and y_test is not None:
            test_loss = np.mean(np.square(y_test - self.forward_propagation(X_test)))
            return test_loss
        else:
            print("Test data is not provided.")
            return None

    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)

    @staticmethod
    def plot_errors(train_errors, test_errors=None):
        epochs = len(train_errors)
        plt.plot(range(epochs), train_errors, label='Train Error')
        if test_errors is not None:
            plt.plot(range(epochs), test_errors, label='Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Training and Test Error')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot training loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(range(epochs), train_errors, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


def load_datasets(dataset_name):
    if dataset_name == 'iris':
        data = np.genfromtxt('iris.csv', delimiter=',', dtype=str)
        X = data[:, :-1].astype(float)
        y = data[:, -1]
        # Normalize input features
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        # One-hot encode the labels
        classes = np.unique(y)
        y_one_hot = np.eye(len(classes))[np.searchsorted(classes, y)]
        
        # Split the data manually
        train_ratio = 0.8  # 80% of data for training
        train_size = int(train_ratio * len(X))
        X_train, X_test = X_normalized[:train_size], X_normalized[train_size:]
        y_train, y_test = y_one_hot[:train_size], y_one_hot[train_size:]
        
        return X_train, y_train, X_test, y_test
        
    elif dataset_name == 'wine':
        # Load wine dataset
        wine_data = np.genfromtxt('wine.csv', delimiter=',', skip_header=1)
        X = wine_data[:, 1:]
        y = wine_data[:, 0]
        # Normalize input features
        X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        # One-hot encode the labels
        classes = np.unique(y)
        y_one_hot = np.eye(len(classes))[np.searchsorted(classes, y)]
        
        # Split the data manually
        train_ratio = 0.8  # 80% of data for training
        train_size = int(train_ratio * len(X))
        X_train, X_test = X_normalized[:train_size], X_normalized[train_size:]
        y_train, y_test = y_one_hot[:train_size], y_one_hot[train_size:]
        
        return X_train, y_train, X_test, y_test
        
    else:
        raise ValueError("Invalid dataset name. Please choose 'iris' or 'wine'.")



# User inputs
input_size = int(input("Enter the input size: "))
hidden_layers = int(input("Enter the number of hidden layers: "))
neurons_per_layer = []
for i in range(hidden_layers):
    neurons = int(input(f"Enter the number of neurons in hidden layer {i+1}: "))
    neurons_per_layer.append(neurons)
output_size = int(input("Enter the output size: "))
activation = input("Enter the activation function (sigmoid or tanh): ")
batch_size = int(input("Enter the batch size: "))
epochs = int(input("Enter the number of epochs: "))
learning_rate = float(input("Enter the learning rate: "))
momentum = float(input("Enter the momentum: "))
dataset_name = input("Enter the dataset name (iris or wine): ")

# Load datasets
X_train, y_train, X_test, y_test = load_datasets(dataset_name)
print("Dataset loaded successfully.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Train the model
model = MLP(input_size=input_size, hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer, output_size=output_size, activation=activation)
train_errors, test_errors = model.train(X_train, y_train, X_test, y_test, epochs=epochs, learning_rate=learning_rate, momentum=momentum, batch_size=batch_size)

# Plot the errors
MLP.plot_errors(train_errors, test_errors)

