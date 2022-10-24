import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import itertools # To join a list of lists
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

FOLDER_NAME = 'photos_colored_strings_cropped_augmented'
#FOLDER_NAME = 'photos_colored_strings'
# Switch Device
device = 'cuda'
#device = 'cpu'
kernel_size = 30
# Switch to false if there is a model already made and you don't want to continue training
train_model = True

print(os.getcwd() + '\\' + FOLDER_NAME + '.pth')

"""
Potentially keep aspect ratio just decrease size

Need to look into other torch transform options
"""
transform = transforms.Compose([
        transforms.Resize([280,280]), # Resizing the image
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0.5 due to how torchvision data is in range [0,1]
    ])

# Specify the batch size for SGD

#TODO: could experiment with batch size ... check size if marks dataset
batch_size = 64

# Use the ImageFolder class, which is a generic dataloder useful when images stored in labelled directories
# Apply transform to images specified above, importantly converting "ToTensor"
dataset = torchvision.datasets.ImageFolder(root=FOLDER_NAME, transform=transform)
# Specify classes for convenience in printing
classes = ( 'no_string', 'string1', 'string2', 'string3', 'string4', 'string5', 'string6')
print("Dataset has N = {} samples".format(len(dataset)))

# List of targets/labels for entire dataset
targets = dataset.targets

# Use sklearn function to split the data into training-test sets (80-20 split)
# Stratified ensures that same percentage of samples of each target class form a complete set
train_idx, test_idx= train_test_split(
    np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)

# Printing the unique class labels in each split set and the number of occurrences per class
print("Printing the unique class labels in each split set and the number of occurrences per class")
print(np.unique(np.array(targets)[train_idx], return_counts=True))
print(np.unique(np.array(targets)[test_idx], return_counts=True))

# Create training and test subsets based on split indices
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

# Define iterable for our newly created dataset and shuffle samples
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

### Plotting stuff for visualization ###
fig = plt.figure(figsize=(20,20))
# Utility function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    img_np = img.numpy()
    # Todo: check order of channels and compare with livestream
    plt.imshow(np.transpose(img_np, (1, 2, 0)))


###Used for printing examples
# Get some random training dataset images
dataiter = iter(train_dataloader)
# Extract a batch
images, labels = dataiter.next()
"""
print(labels)
# Show 8 images for display
num_display = 8
imshow(torchvision.utils.make_grid(images[:num_display]))
# Print labels as a concatenated string
print("\nLabels corresponding to randomly drawn images from the training set:")
print(" ".join(f'{classes[labels[j]]:5s}\t' for j in range(num_display)))
"""


# Note that PyTorch uses NCHW (samples, channels, height, width) image format convention
class ConvNet(nn.Module):
    def __init__(self, in_channels, num_filters, out_classes, kernel_size):
        super().__init__()
        # Conv2D layer with 'same' padding so image retains shape
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, padding='same')
        self.drop = nn.Dropout(p=0.2)
        # self.dropout =
        # MaxPooling layer with 2x2 kernel size and stride 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size, padding='same')
        self.fc1 = nn.Linear(num_filters * 2 * 70 * 70, num_filters)
        self.fc2 = nn.Linear(num_filters, out_classes)

    def forward(self, x):
        # Non-linear ReLU activations between convolutional layers
        # Conv->ReLU->Pooling
        # 280x280 image -> 140x140 after pooling
        x = self.pool(self.drop(F.relu(self.conv1(x))))

        # 140x140 feature map -> 70x70 after pooling
        x = self.pool(self.drop(F.relu(self.conv2(x))))

        # Flatten all dimensions except batch (start_dim=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


input_channels = images.shape[1]

model = ConvNet(input_channels, num_filters=16, out_classes=len(classes), kernel_size=kernel_size).to(device)
# Dummy inputs so we can plot a summary of the neural network's architecture and no. of parameters
summary(model, input_size=(input_channels, 280, 280))

# Double check if model exists
if os.path.exists(os.getcwd() + '\\' + FOLDER_NAME + '_kernalsize_' + str(kernel_size) + '.pth'):
    model.load_state_dict(torch.load(os.getcwd() + '\\' + FOLDER_NAME + '_kernalsize_' + str(kernel_size) + '.pth'))
    print("Loaded model from disk!")



##########################
###### Training
##########################

def model_train_loader(model, dataloader, criterion, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        predictions = model(X.to(device))
        loss = criterion(predictions, y.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report loss every 10 batches
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def model_test_loader(model, dataloader, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Tracking test loss (cross-entropy) and correct classification rate (accuracy)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            predictions = model(X)
            test_loss += criterion(predictions, y)
            correct += (predictions.argmax(1) == y).type(torch.float).sum()

    test_loss /= num_batches
    correct /= size
    print(f"\nTest Error\n\tAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to output
criterion = nn.CrossEntropyLoss().to(device)

num_epochs = 5
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
if train_model:
    for t in range(num_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        model_train_loader(model, train_dataloader, criterion, optimizer)
        model_test_loader(model, test_dataloader, criterion)

    # Saving the model file 'cnn_fmd.pth' to my current working directory (cwd)
    print("Saving model to disk!")
    torch.save(model.state_dict(), os.getcwd() + '\\' + FOLDER_NAME + '_kernalsize_' + str(kernel_size) + '.pth')



"""
Bring back to cpu
"""
model = model.to('cpu')

correct = 0

predicted_list = []
labels_list = []

# Since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    i = 0
    for data in test_dataloader:
        i += 1
        images, labels = data

        print(images.shape)

        # Forward pass through network
        outputs = model(images)
        # Take most probable class
        _, predicted = torch.max(outputs.data, 1)
        predicted_list.append(predicted.detach().numpy())
        labels_list.append(labels.detach().numpy())

# Hacky conversions from a list of lists of arrays into joined numpy arrays
predicted_list = np.array(list(itertools.chain.from_iterable(predicted_list)))
labels_list = np.array(list(itertools.chain.from_iterable(labels_list)))
print("true labels: ", labels_list)
# Count up number of labels and correct predictions
correct = sum(predicted_list == labels_list)

print(f"Accuracy of the network on the test set images: {100 * correct // len(test_dataloader.dataset)} %")

print("Confusion Matrix (columns: True class, rows: Predicted class):")
conf_mat = confusion_matrix(labels_list, predicted_list);
conf_display = ConfusionMatrixDisplay.from_predictions(labels_list, predicted_list, colorbar=False);
plt.xlabel("True Labels");
plt.ylabel("Predicted Labels");
plt.show()