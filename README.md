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
Fruit Classification Model Using CNN

I implemented a fruit classification model using a Convolutional Neural Network (CNN) in Python. The goal was to classify fruit and vegetable images into 36 different classes. Here are the key details:

    Dataset:
        I trained the CNN model on a dataset containing over 2800 images of various fruits.
        The dataset included a variety of fruit types, such as apples, bananas, oranges, and more.

    Model Architecture:
        The CNN architecture consisted of convolutional layers, pooling layers, and fully connected layers.
        I used Keras to build the model, which allowed for easy experimentation and customization.

    Training and Validation:
        During training, I split the dataset into training and validation sets.
        The model achieved an impressive accuracy of 90% on the validation set.

    Data Augmentation:
        To improve generalization, I applied data augmentation techniques such as rotation, zoom, and horizontal flips.
        Data augmentation helped the model learn robust features from the limited training data.

    Resume Highlights:
        Developed a CNN-based fruit classification model using Python and Keras.
        Achieved 90% accuracy on a diverse fruit dataset.
        Demonstrated proficiency in deep learning and computer vision.
