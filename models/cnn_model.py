import torch
import torch.nn as nn

class conv_net(nn.Module):
    def __init__(self, num_filters, hidden):
        super(conv_net, self).__init__()
        C_in = 1
        # parametry dla conv
        conv_kernel_size = 3
        conv_stride = 1
        conv_padding = (conv_kernel_size - 1) // 2 # dzięki czemu w conv h i w nie zmienią się

        # parametry dla max_pool
        max_pool_kernel_size = 2
        max_pool_stride = 2

        self.hidden = hidden
        num_classes = 10

        # h i w dla danych MNIST
        h = 28
        w = 28

        self.conv1 = nn.Conv2d(C_in, num_filters[0], conv_kernel_size, stride=conv_stride, padding=conv_padding)
        # h i w po conv
        h = ((h - conv_kernel_size + 2 * conv_padding) / conv_stride) + 1
        w = ((w - conv_kernel_size + 2 * conv_padding) / conv_stride) + 1

        self.batch_norm1 = nn.BatchNorm2d(num_filters[0])
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(max_pool_kernel_size, stride = max_pool_stride)
        # h i w po max_pool
        h = ((h - max_pool_kernel_size) / max_pool_stride) + 1
        w = ((w - max_pool_kernel_size) / max_pool_stride) + 1

        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], conv_kernel_size, stride=conv_stride, padding=conv_padding)
        h = ((h - conv_kernel_size + 2 * conv_padding) / conv_stride) + 1
        w = ((w - conv_kernel_size + 2 * conv_padding) / conv_stride) + 1
        self.batch_norm2 = nn.BatchNorm2d(num_filters[1])
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(max_pool_kernel_size, stride = max_pool_stride)
        h = int(((h - max_pool_kernel_size) / max_pool_stride) + 1)
        w = int(((w - max_pool_kernel_size) / max_pool_stride) + 1)

        # Warstwa liniowa
        self.fc1 = nn.Linear(num_filters[1] * h * w, self.hidden)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, num_classes)


    def forward(self, x):
        # expekt image torch.Size([64, 1, 28, 28])
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

    @staticmethod
    def save(model, path="C:/Users/malel/Desktop/python/mnist_project/data/models/cnn_model.pth"):
        checkpoint = {
            'state_dict': model.state_dict(),  # Save model parameters
        }
        torch.save(checkpoint, path)
        print(f"Saved model to {path}")

    @staticmethod
    def load(model, path="C:/Users/malel/Desktop/python/mnist_project/data/models/cnn_model.pth"):
        checkpoint = torch.load(path, weights_only=True)  # Load the checkpoint
        model.load_state_dict(checkpoint['state_dict'])  # Load parameters into the model
        print(f"Loaded model from {path}")