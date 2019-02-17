"""
Made by: 
Vincent Morissette-Thomas
Fanny Salvail-Bérard
Vilayphone Vilaysouk

Inspired by:
https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/401_CNN.py
"""
# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import accumulate

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()


def plot_training_history(train_loss_hist, valid_loss_hist, valid_accu_hist):
  # Loss Curves
  plt.figure(figsize=[8,6])
  plt.plot(train_loss_hist,'r',linewidth=2.0)
  plt.plot(valid_loss_hist,'b',linewidth=2.0)
  plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss',fontsize=16)
  plt.title('Loss Curves',fontsize=20, fontweight='bold')
  plt.savefig("Loss.png")

  # Accuracy Curves
  plt.figure(figsize=[8,6])
  plt.plot(valid_accu_hist,'b',linewidth=2.0)
  plt.legend(['Validation Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Accuracy',fontsize=16)
  plt.title('Accuracy Curves',fontsize=20, fontweight='bold')
  plt.savefig("Accuracy.png")

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    print("Length :", lengths)
    indices = torch.randperm(len(dataset))
    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(accumulate(lengths), lengths)]


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 30              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.1              # learning rate
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

train_dataset, valid_dataset = random_split(train_data, [50000, 10000])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:10000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
train_accu_hist = []
valid_accu_hist = []
train_loss_hist = []
valid_loss_hist = []

# training and testing
for epoch in range(EPOCH):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    accuracy = 0.0

    ###################
    # train the model #
    ###################
    cnn.train()

    print('------ train ------')

    for data, target in train_loader:


        # move tensors to GPU if CUDA is available
        # target = target.Long()

        # if train_on_gpu:
        #     data, target = data.cuda(), target.cuda()

        output = cnn(data)[0]               # cnn output
        # target = output.type_as(target)
        # print(type(data[0]), type(target[0]), type(output[0]))
        loss = loss_func(output, target)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # update training loss
        train_loss += loss.item()

    
    ######################
    # validate the model #
    ######################
    cnn.eval()

    print('------ valid ------')

    for data, target in valid_loader:

        # move tensors to GPU if CUDA is available
        # if train_on_gpu:
        #     data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model


        output = cnn(data)[0]               # cnn output
        loss = loss_func(output, target)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients


        # update average validation loss
        valid_loss += loss.item()
        # pred_y = torch.max(output, 1)[1].data.numpy()
        pred_y = torch.max(output, 1)[1].data
        equals = (pred_y == target)
        
        accuracy += float(torch.sum(equals))



    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    accuracy = accuracy / len(valid_loader.dataset)

    train_loss_hist.append(train_loss)
    valid_loss_hist.append(valid_loss)
    valid_accu_hist.append(accuracy)

    # print training/validation statistics
    print('Epoch: {} \tTrain. Loss: {:.6f} \tValid. Loss: {:.6f}  \tValid. Accuracy: {:.6f}  '.format(
        epoch, train_loss, valid_loss, accuracy))


# print 10 predictions from test data

test_output, _ = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data
equals = (pred_y == test_y)
accuracy = float(torch.sum(equals))/len(equals)
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')

plot_training_history(train_loss_hist, valid_loss_hist, valid_accu_hist)

pytorch_total_params = sum(p.numel() for p in cnn.parameters())
print("numb. parameters ", pytorch_total_params)
