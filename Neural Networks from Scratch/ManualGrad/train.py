import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    
class GradTensor:

    def __init__(self, params):
        self.params = params
        self.grad = None
    
    def _zero_grad(self):
        self.grad = None
    
class Linear(Layer):
    
    """
    Basic Implementation of the Linear Layer following nn.Linear
    y = xW^T + b
    """

    def __init__(self, in_features, out_features, bias=True):
    
        self.in_features = in_features
        self.out_features = out_features

        ### Initialization to Match nn.Linear ###
        k = 1 / self.in_features

        self.weight = GradTensor(
            np.random.uniform(
                low=-math.sqrt(k),
                high=math.sqrt(k), 
                size=(in_features, out_features)
            )
        )

        self.bias = None
        if bias is not None:
            self.bias = GradTensor(
                np.random.uniform(
                    low=-math.sqrt(k), 
                    high=math.sqrt(k), 
                    size=(1,out_features)
                )
            )

    def forward(self, x):

        # For backprop, dL/dW will need X^t, so save X for future use 
        self.x = x

        # X has shape (B x in_features), w has shape (in_feature x out_features)
        x = x @ self.weight.params

        if self.bias is not None:
            x = x + self.bias.params

        return x
    
    def backward(self, output_grad):

        ### Deriviate w.r.t. W: X^t @ output_grad 
        self.weight.grad = self.x.T @ output_grad

        ### Derivative of Bias is jsut the output grad summed along batch 
        self.bias.grad = output_grad.sum(axis=0, keepdims=True)

        ### We need derivative w.r.t for the next step 
        input_grad = output_grad @ self.weight.params.T

        return input_grad

class MSELoss:

    """
    L = E[(y-y_hat)^2]
    """
    def forward(self, y_pred, y_true):
    
        self.y_pred = y_pred
    
        self.y_true = y_true
    
        return np.mean((y_true - y_pred)**2)
    
    def backward(self):

        batch_size = self.y_pred.shape[0]

        grad = -(2/batch_size) * (self.y_true - self.y_pred)

        return grad
    
class Sigmoid:
    def forward(self, x):

        self.x = x

        return 1 / (1 + np.exp(-x))
    
    def backward(self, output_grad):

        sigmoid_x = self.forward(self.x)

        sigmoid_grad = sigmoid_x * (1 - sigmoid_x)

        input_grad = sigmoid_grad * output_grad
        
        return input_grad

    
class NeuralNetwork:
    
    """
    The most basic Neural Network Ever!
    """
    def __init__(self):

        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)

    def parameters(self):
        parameters = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                parameters.append(layer.weight)
                if layer.bias is not None:
                    parameters.append(layer.bias)
        return parameters
    
class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for param in self.parameters:
            assert param.grad.shape == param.params.shape, "Something has gone horribly wrong"
            param.params -= param.grad * self.lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    
x_train = np.arange(-10,10,0.1).reshape(-1,1)
y_train = 3*x_train**2 + 2 + np.random.normal(scale=5, size=(x_train.shape[0], x_train.shape[1]))

network = NeuralNetwork()
network.add(Linear(1,32))
network.add(Sigmoid())
network.add(Linear(32,64))
network.add(Sigmoid())
network.add(Linear(64,64))
network.add(Sigmoid())
network.add(Linear(64,32))
network.add(Sigmoid())
network.add(Linear(32,1))

loss_func = MSELoss()
optimizer = SGD(network.parameters(), 1e-3)

losses = []
for idx in tqdm(range(25000)):
    output = network(x_train)

    loss = loss_func.forward(output, y_train)
    loss_grad = loss_func.backward()
    network.backward(loss_grad)
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss)

    if idx % 250 == 0:
        print("Loss:", loss)
pred = network(x_train)

plt.plot(y_train, label="truth")
plt.plot(pred, label="pred")
plt.legend()
plt.show()

plt.plot(losses)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm


# class Layer:
#     def forward(self, x):
#         raise NotImplementedError
    
#     def backward(self, grad):
#         raise NotImplementedError

# import math
# class Linear(Layer):
#     def __init__(self, in_features, out_features, bias=True):
#         self.w = np.random.uniform(
#                 low=-math.sqrt(1/in_features),
#                 high=math.sqrt(1/in_features), 
#                 size=(in_features, out_features)
#             )
#         self.w_grad = np.zeros_like(self.w)
#         self.bias = bias
#         if bias:
#             self.b =  np.random.uniform(
#                 low=-math.sqrt(1/in_features),
#                 high=math.sqrt(1/in_features), 
#                 size=(1, out_features)
#             )
#             self.b_grad = np.zeros_like(self.b)
#         else:
#             self.b = np.zeros((1, out_features))
#             self.b_grad = None  # No gradient if bias is disabled

#     def forward(self, x):
#         self.x = x  # Save input for backprop
#         output = x @ self.w + self.b
#         return output

#     def backward(self, output_grad):
#         self.w_grad = self.x.T @ output_grad
#         if self.bias:
#             self.b_grad = output_grad.sum(axis=0, keepdims=True)
#         return output_grad @ self.w.T  # Input gradient


# class Sigmoid(Layer):
#     def forward(self, x):
#         self.out = 1 / (1 + np.exp(-x))
#         return self.out
    
#     def backward(self, output_grad):
#         return output_grad * self.out * (1 - self.out)


# class MSELoss:
#     def forward(self, y_pred, y_true):
#         self.y_pred = y_pred
#         self.y_true = y_true
#         return np.mean((y_true - y_pred) ** 2)
    
#     def backward(self):
#         return -2 * (self.y_true - self.y_pred) / self.y_pred.shape[0]


# class NeuralNetwork:
#     def __init__(self):
#         self.layers = []
    
#     def add(self, layer):
#         self.layers.append(layer)
    
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer.forward(x)
#         return x
    
#     def backward(self, grad):
#         for layer in reversed(self.layers):
#             grad = layer.backward(grad)
    
#     def parameters(self):
#         params = []
#         for layer in self.layers:
#             if isinstance(layer, Linear):
#                 params.append((layer.w, layer.w_grad))
#                 if layer.bias:
#                     params.append((layer.b, layer.b_grad))
#         return params


# class SGD:
#     def __init__(self, network, lr=0.001):
#         self.network = network
#         self.lr = lr
    
#     def step(self):
#         for param, grad in self.network.parameters():
#             param -= self.lr * grad
    
#     def zero_grad(self):
#         for layer in self.network.layers:
#             if isinstance(layer, Linear):
#                 layer.w_grad.fill(0)
#                 if layer.bias:
#                     layer.b_grad.fill(0)


# class LinearScheduler:
#     def __init__(self, optimizer, total_steps, warmup_steps):
#         self.optimizer = optimizer
#         max_lr = optimizer.lr
#         self.schedule = np.concatenate([
#             np.linspace(0, max_lr, warmup_steps),
#             np.linspace(max_lr, 0, total_steps - warmup_steps)
#         ])
#         self.step_count = 0
    
#     def step(self):
#         self.optimizer.lr = self.schedule[self.step_count]
#         self.step_count += 1


# def train(network, x_train, y_train, loss_fn, optimizer, scheduler=None, epochs=25000):
#     losses = []
#     for epoch in tqdm(range(epochs)):
#         # Forward pass
#         y_pred = network.forward(x_train)
#         loss = loss_fn.forward(y_pred, y_train)
        
#         # Backward pass
#         loss_grad = loss_fn.backward()
#         network.backward(loss_grad)
        
#         # Optimization step
#         optimizer.step()
#         if scheduler:
#             scheduler.step()
#         optimizer.zero_grad()
        
#         losses.append(loss)
#         if epoch % 250 == 0:
#             print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
#     return losses


# if __name__ == "__main__":
#     # Data
#     x_train = np.arange(-10, 10, 0.1).reshape(-1, 1)
#     y_train = 3 * x_train**2 + 2 + np.random.normal(scale=5, size=x_train.shape)

#     # Model
#     net = NeuralNetwork()
#     net.add(Linear(1, 32))
#     net.add(Sigmoid())
#     net.add(Linear(32, 64))
#     net.add(Sigmoid())
#     net.add(Linear(64, 64))
#     net.add(Sigmoid())
#     net.add(Linear(64, 32))
#     net.add(Sigmoid())
#     net.add(Linear(32, 1))

#     # Training setup
#     loss_fn = MSELoss()
#     optimizer = SGD(net, lr=1e-3)
#     scheduler = LinearScheduler(optimizer, total_steps=25000, warmup_steps=2500)

#     # Train
#     losses = train(net, x_train, y_train, loss_fn, optimizer, scheduler)

#     # Visualize
#     y_pred = net.forward(x_train)
#     plt.plot(y_train, label="Truth")
#     plt.plot(y_pred, label="Prediction")
#     plt.legend()
#     plt.show()

#     plt.plot(losses, label="Loss")
#     plt.legend()
#     plt.show()