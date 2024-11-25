import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / (hidden_dim + output_dim))
        self.b2 = np.zeros((1, output_dim))

    # define activation functions and their derivatives
    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation(self.z2)  # For binary classification, use sigmoid
        return self.a2

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        dz2 = self.a2 - y  # Binary cross-entropy derivative
        dW2 = np.dot(self.a1.T, dz2) / X.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

        # TODO: update weights with gradient descent
        dz1 = np.dot(dz2, self.W2.T) * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / X.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

        # TODO: store gradients for visualization
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.gradients = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    # **Hidden Space Visualization:**
    xx, yy = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Generate grid for hidden activations
    hidden_x_range = np.linspace(np.min(mlp.a1[:, 0]) - 0.5, np.max(mlp.a1[:, 0]) + 0.5, 50)
    hidden_y_range = np.linspace(np.min(mlp.a1[:, 1]) - 0.5, np.max(mlp.a1[:, 1]) + 0.5, 50)
    xx, yy = np.meshgrid(hidden_x_range, hidden_y_range)
    grid_hidden = np.c_[xx.ravel(), yy.ravel()]

    # Compute activation surface using hidden layer activations and output weights
    activation_surface = mlp.activation(np.dot(grid_hidden, mlp.W2[:2, :]) + mlp.b2).reshape(xx.shape)

    # Plot the activation surface
    ax_hidden.plot_surface(xx, yy, activation_surface, alpha=0.4, color='green')

    # Plot the hyperplane of the first hidden neuron
    # Plot hyperplane for the first hidden neuron
    weights = mlp.W1[:, 0]  # Weights of the first hidden neuron (2D)
    bias = mlp.b1[0, 0]     # Bias of the first hidden neuron
    zz = -(weights[0] * xx + weights[1] * yy + bias)  # 2D hyperplane equation
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='gold')

    # Scatter plot of hidden layer activations
    ax_hidden.scatter(
        mlp.a1[:, 0], mlp.a1[:, 1], mlp.a1[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.8
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame}")
    ax_hidden.set_xlabel('Hidden Neuron 1')
    ax_hidden.set_ylabel('Hidden Neuron 2')
    ax_hidden.set_zlabel('Hidden Neuron 3')

    # Dynamically adjust axis limits
    xlim = (np.min(mlp.a1[:, 0]) - 0.5, np.max(mlp.a1[:, 0]) + 0.5)
    ylim = (np.min(mlp.a1[:, 1]) - 0.5, np.max(mlp.a1[:, 1]) + 0.5)
    zlim = (np.min(mlp.a1[:, 2]) - 0.5, np.max(mlp.a1[:, 2]) + 0.5)

    ax_hidden.set_xlim(*xlim)
    ax_hidden.set_ylim(*ylim)
    ax_hidden.set_zlim(*zlim)

    # Add grid and legend for better clarity
    ax_hidden.grid(True)
    ax_hidden.legend()

    # Dynamically adjust axis limits
    ax_hidden.set_xlim(np.min(mlp.a1[:, 0]) - 0.5, np.max(mlp.a1[:, 0]) + 0.5)
    ax_hidden.set_ylim(np.min(mlp.a1[:, 1]) - 0.5, np.max(mlp.a1[:, 1]) + 0.5)
    ax_hidden.set_zlim(np.min(mlp.a1[:, 2]) - 0.5, np.max(mlp.a1[:, 2]) + 0.5)

    # Input space visualization
    x1, x2 = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    grid = np.c_[x1.ravel(), x2.ravel()]
    output = mlp.forward(grid).reshape(x1.shape)
    ax_input.contourf(x1, x2, output, levels=50, cmap='bwr', alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', edgecolor='k', alpha=0.7)
    ax_input.set_title("Decision Boundary")

    # Gradient visualization
    nodes = {'x1': (-1, 1), 'x2': (-1, -1), 'h1': (0, 1), 'h2': (0, 0), 'h3': (0, -1), 'y': (1, 0)}
    norm_input_hidden = (np.abs(mlp.W1) - np.min(np.abs(mlp.W1))) / (
        np.max(np.abs(mlp.W1)) - np.min(np.abs(mlp.W1))
    )
    norm_input_hidden = 0.1 + norm_input_hidden * 2.0  # Scale between [0.1, 2.0]

    norm_hidden_output = (np.abs(mlp.W2) - np.min(np.abs(mlp.W2))) / (
        np.max(np.abs(mlp.W2)) - np.min(np.abs(mlp.W2))
    )
    norm_hidden_output = 0.1 + norm_hidden_output * 2.0

    for i, (x_name, x_coord) in enumerate(list(nodes.items())[:2]):
        for j, (h_name, h_coord) in enumerate(list(nodes.items())[2:5]):
            thickness = norm_input_hidden[i, j]
            ax_gradient.plot(
                [x_coord[0], h_coord[0]], [x_coord[1], h_coord[1]],
                linewidth=thickness, color='purple', alpha=0.5
            )

    for i, (h_name, h_coord) in enumerate(list(nodes.items())[2:5]):
        thickness = norm_hidden_output[i]
        ax_gradient.plot(
            [h_coord[0], nodes['y'][0]], [h_coord[1], nodes['y'][1]],
            linewidth=thickness, color='blue', alpha=0.5
        )

    for name, coord in nodes.items():
        ax_gradient.scatter(*coord, s=300, color='blue')
        ax_gradient.text(*coord, name, fontsize=12, ha='center', va='center', color='white')

    ax_gradient.set_title(f"Gradients at Step {frame}")
    # The edge thickness visually represents the magnitude of the gradient


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)