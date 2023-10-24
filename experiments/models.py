from torch import nn
from torch.functional import F

class NN(nn.Module):
    def __init__(self, num_classes):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x
    


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out




class SmallLeNet(nn.Module):
    def __init__(self):
        super(SmallLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(10, 10)
        # self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)          
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = F.max_pool2d(x, 2)
        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)

        # output = F.log_softmax(x, dim=1)
        return x