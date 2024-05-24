# Fruits classificator Model using Convolutional Neural Netoworks

For this model I used a convolutional neural network model that classifies fruits with the following classification :

model=nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=10, kernel_size=10),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=10),
    nn.Flatten(),
    nn.Linear(810, 1024),
    nn.ReLU(),
    nn.Linear(1024, 6),
    nn.Softmax(dim=1)
)
