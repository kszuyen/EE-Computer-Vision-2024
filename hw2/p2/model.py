import torch
import torch.nn as nn
import torchvision.models as models


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.1))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 3), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout(0.1))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 3), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout(0.1))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Sequential(nn.Conv2d(512, 1024, 3), nn.BatchNorm2d(1024), nn.ReLU(), nn.Dropout(0.1))
        self.fc1 = nn.Sequential(nn.Linear(1024 * 2 * 2, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = self.conv7(x)
        x = x.view(-1, 1024 * 2 * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)

        self.resnet.maxpool = Identity()
        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. #
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################

    def forward(self, x):
        return self.resnet(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    # from torchsummary import summary

    model = ResNet18()
    # summary(model, (3, 32, 32))
    print(model)

    # x = torch.rand(4, 3, 32, 32)
    # # for i, k in enumerate(list(models.densenet121().children())):
    # #     print(i, ': ', k)
    # model = MyNet()
    # y = model(x)
    # print(y.shape)
