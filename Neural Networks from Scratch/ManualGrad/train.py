import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

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

class SoftMax:

    def forward(self, x):
        
        self.x = x

        shifted_x = x - np.max(x, axis=-1, keepdims=True)

        exp_x = np.exp(shifted_x)

        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return probs

    def backward(self, output_grad):
        
        probs = self.forward(self.x)

        gradient = np.zeros_like(probs)

        batch_size = len(self.x)

        for i in range(batch_size):

            sample_probs = probs[i]
            
            j = -sample_probs.reshape(-1,1) * sample_probs.reshape(1,-1)
            j[np.diag_indices(j.shape[0])] = sample_probs * (1 - sample_probs)

            a = output_grad[i] @ j
            
            gradient[i] = a
        
        return gradient

class CrossEntropyLoss:

    def forward(self, y_true, y_pred):

        self.y_true = y_true
        self.y_pred = y_pred

        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        
        return loss
        
    def backward(self):
        
        grad = - self.y_true / self.y_pred

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

class ReLU:
    def forward(self, x):
        self.x = x
        return np.clip(x, a_min=0, a_max=None)
    
    def backward(self, output_grad):
        
        grad = np.zeros_like(self.x)
        grad[self.x > 0] = 1
        
        return grad * output_grad
    
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
    
    def __repr__(self):
        # Start by printing the class name for the whole model
        model_repr = "NeuralNetwork(\n"
        
        # Print each layer's class name and parameters
        for layer in self.layers:
            if isinstance(layer, Linear):
                model_repr += f"  Linear(in_features={layer.in_features}, out_features={layer.out_features}, bias={layer.bias is not None})\n"
            elif isinstance(layer, Sigmoid):
                model_repr += "  Sigmoid()\n"
            elif isinstance(layer, ReLU):
                model_repr += "  ReLU()\n"
            elif isinstance(layer, SoftMax):
                model_repr += "  SoftMax()\n"
        
        # Close the model string representation
        model_repr += ")"
        
        return model_repr
    
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

if __name__ == "__main__":

    network = NeuralNetwork()

    network.add(Linear(784,512))
    network.add(ReLU())
    network.add(Linear(512,256))
    network.add(ReLU())
    network.add(Linear(256,128))
    network.add(ReLU())
    network.add(Linear(128,10))
    network.add(SoftMax())

    print(network)

    ### Prep Dataset ###
    train_dataset = MNIST("../../data", train=True, download=True)
    test_dataset = MNIST("../../data", train=True, download=True)

    def collate_fn(batch):
 
        ### Prep and Scale Images ###
        images = np.concatenate([np.array(i[0]).reshape(1,784)for i in batch]) / 255

        ### One Hot Encode Label (MNIST only has 10 classes) ###
        labels = [i[1] for i in batch]
        labels = np.eye(10)[labels]

        return images, labels
    
    trainloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
    testloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    loss_func = CrossEntropyLoss()
    optimizer = SGD(network.parameters(), lr=1e-3)

    training_iterations = 2500

    all_train_losses = []
    all_train_accs = []
    all_test_losses = []
    all_test_accs = []

    num_iters = 0
    train = True
    pbar = tqdm(range(training_iterations))
    
    while train:

        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for images, labels in trainloader:
            
            ### Get Outputs from Model ###
            output = network(images)
            loss = loss_func.forward(y_pred=output, y_true=labels)
            loss_grad = loss_func.backward()

            ### Compute Gradients in Model ###
            network.backward(loss_grad)

            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad()

            ### Compute Accuracy ###
            preds = output.argmax(axis=-1)
            labels = labels.argmax(axis=-1)
            acc = (preds == labels).sum() / len(preds)

            ### Store Loss for Plotting ###
            train_losses.append(loss)
            train_accs.append(acc)
            
            ### Eval Loop ###
            if num_iters % 250 == 0:

                for images, labels in testloader:

                    ### Get Outputs from Model ###
                    output = network(images)
                    loss = loss_func.forward(y_pred=output, y_true=labels)

                    ### Compute Accuracy ###
                    preds = output.argmax(axis=-1)
                    labels = labels.argmax(axis=-1)
                    acc = (preds == labels).sum() / len(preds)

                    ### Store Loss for Plotting ###
                    test_losses.append(loss)
                    test_accs.append(acc)

                ### Average Up Performance and Store ###
                train_losses = np.mean(train_losses)
                train_accs = np.mean(train_accs)
                test_losses = np.mean(test_losses)
                test_accs = np.mean(test_accs)

                all_train_losses.append(train_losses)
                all_train_accs.append(train_accs)
                all_test_losses.append(test_losses)
                all_test_accs.append(test_accs)

                print("Training Loss:", train_losses)
                print("Training Acc:", train_accs)
                print("Testing Loss:", test_losses)
                print("Testing Acc:", test_accs)

                ### Reset Lists ###
                train_losses = []
                train_accs = []
                test_losses = []
                test_accs = []

            
            num_iters += 1
            pbar.update(1)

            if num_iters >= training_iterations:
                print("Completed Training")
                train = False
                break
