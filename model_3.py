
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# class Net(nn.Module):
#     #This defines the structure of the NN.
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm2d(16)

#         self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm2d(16)

#         self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn3   = nn.BatchNorm2d(32)

#         self.conv4 = nn.Conv2d(32, 16, kernel_size=1)
#         self.bn4   = nn.BatchNorm2d(16)

#         self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn5   = nn.BatchNorm2d(32)

#         self.conv6 = nn.Conv2d(32, 16, kernel_size=1)
#         self.bn6   = nn.BatchNorm2d(16)

#         self.conv7 = nn.Conv2d(16, 16, kernel_size=3)
#         self.bn7   = nn.BatchNorm2d(16)

#         # Fully connected layer
#         self.fc1 = nn.Linear(400, 10)

#         # Dropout layers
#         self.dropout1 = nn.Dropout(0.05)   # 5% dropout


#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.dropout1(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.dropout1(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.dropout1(x)
#         x = F.max_pool2d(x, 2)   # pooling after activation

#         x = F.relu(self.conv4(x))
#         x = F.relu(self.bn5(self.conv5(x)))
#         x = self.dropout1(x)
#         x = F.max_pool2d(x, 2)

#         x = F.relu(self.bn6(self.conv6(x)))

#         x = F.relu(self.bn7(self.conv7(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 400)   # flatten

#         x = self.dropout1(x)
#         x = self.fc1(x)

#         return F.log_softmax(x, dim=1)


import torch.nn.functional as F
dropout_value = 0.02
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

#!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.05),
                        transforms.Resize((28, 28)),
                        transforms.RandomRotation((-8., 8.), fill=1),
                        #transforms.RandomAffine(0, translate=(0.1, 0.1)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

images, labels = next(iter(train_loader))
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

print(images.shape)
print(labels.shape)

# Let's visualize some of the images
# %matplotlib inline
import matplotlib.pyplot as plt

plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

print(len(test_loader))
print(len(train_loader))

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0


    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += GetCorrectPredCount(output, target)
        processed += len(data)
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_accuracy

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.05)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
criterion = nn.CrossEntropyLoss()

EPOCH=15
for epoch in range(EPOCH):
    print(f'Model 3 Epoch {epoch}')
    train(model, device, train_loader, optimizer, epoch, criterion)
    test(model, device, test_loader,criterion)
    scheduler.step()