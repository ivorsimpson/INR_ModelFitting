import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
dev = torch.cpu 
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    dev = mps_device
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")


class INR(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, ip_scale_factor=4.0):
        super(INR, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        torch.nn.init.uniform_(self.hidden_layers[0].weight, a=-np.sqrt(6/input_size), b=np.sqrt(6/input_size))
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            torch.nn.init.uniform_(self.hidden_layers[-1].weight, a=-np.sqrt(6/hidden_sizes[i]), b=np.sqrt(6/hidden_sizes[i]))

        for layer in self.hidden_layers:
            torch.nn.init.zeros_(layer.bias)

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.ip_log_scale_factor = torch.nn.Parameter(torch.tensor(np.log(ip_scale_factor), dtype=torch.float32), requires_grad=True)
        

    def forward(self, input):
        x = input * torch.exp(self.ip_log_scale_factor)
        for layer in self.hidden_layers:
            x = torch.sin(layer(x))
        x = self.output_layer(x)
        output = torch.clamp(x, 0, 1)
        return output
    
class ModelFitter(nn.Module):
    def __init__(self, model, loss_fn, optimizer, noise_std=0.05):
        super(ModelFitter, self).__init__()
        self.model = model.to(dev)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.noise_std = noise_std
        

    def forward(self, input):
        y_pred = self.model(input)
        return y_pred

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        x += torch.randn_like(x) * (1.0/32.)*self.noise_std
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_step(self, x, y):
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            y_pred = self.forward(x)
        return y_pred

# Example usage
input_size = 2
output_size = 3




def CIFAR10_loader():

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    return unpickle('/Users/is321/Downloads/cifar-10-batches-py/data_batch_1')[b'data']

images = CIFAR10_loader()





def super_resolve_image(image, new_side_length):
    # Define the number of iterations and the interval for printing the loss
    num_iterations = 3000
    ip_scale_factor = 5.0
    print_interval = 10

    # Reshape the image to its original dimensions
    image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))

    # Display the image using matplotlib
    #plt.imshow(image)
    #plt.axis('off')
    #plt.show()


    def create_square_grid(s):
        # Create a 1D tensor of values from 0 to 31
        x = torch.linspace(0, s-1, s)
        # Create a 2D grid of pixel coordinates
        grid_x, grid_y = torch.meshgrid(x, x)

        # Scale the coordinates between -1 and 1
        grid_x = (grid_x / (s-1)) * 2 - 1
        grid_y = (grid_y / (s-1)) * 2 - 1

        # Flatten the coordinates into a list of pixel coordinates
        grid = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).to(dev)

        return grid

    grid = create_square_grid(32)
    grid_high_res = create_square_grid(new_side_length)

    image_t = torch.tensor(image.reshape(32*32, 3), device=dev, dtype=torch.float32) / 255.0


    

    hidden_sizes = [64 for x in range(4)]

    model = INR(input_size, hidden_sizes, output_size, ip_scale_factor=ip_scale_factor)
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4)
    mf = ModelFitter(model, nn.MSELoss(), optimizer=optimiser, noise_std=0.0)

    # Loop for training the model
    for i in range(num_iterations):
        # Perform a training step
        mf.train_step(grid, image_t)
        
        # Print the loss every 10 iterations
        if (i + 1) % print_interval == 0:
            y_pred = mf.predict(grid)
            loss = mf.loss_fn(y_pred, image_t)
            print(f"Iteration: {i+1}, Loss: {loss.item()}, Scale: {torch.exp(mf.model.ip_log_scale_factor).item()}")

    reconstructed_image = torch.floor(mf.predict(grid)*255.0).reshape(32, 32, 3).cpu().detach().numpy().astype(np.uint8)
    super_res_image = torch.floor(mf.predict(grid_high_res)*255.0).reshape(new_side_length, new_side_length, 3).cpu().detach().numpy().astype(np.uint8)
    return reconstructed_image, super_res_image

import matplotlib.pyplot as plt
no_examples = 4
no_cols = 4
for i in range(no_examples):
    image = images[i]
    recon, super_res = super_resolve_image(image, 128)
    plt.subplot(no_examples,no_cols,i*no_cols+1)
    plt.imshow(np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0)), interpolation='bilinear')
    plt.axis('off')
    if i == 0:
        plt.title("Orig-Bilin")    
    plt.subplot(no_examples,no_cols,i*no_cols+2)
    plt.imshow(np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0)), interpolation='lanczos')
    plt.axis('off')
    if i == 0:
        plt.title("Orig-Lanczos")
    plt.subplot(no_examples,no_cols,i*no_cols+no_cols-1)
    plt.imshow(recon, interpolation='lanczos')
    if i == 0:
        plt.title("Recon-Lanczos")
    plt.subplot(no_examples,no_cols,i*no_cols+no_cols)
    plt.imshow(super_res, interpolation='lanczos')
    if i == 0:
        plt.title("SuperRes-Lanczos")
plt.show()


    
