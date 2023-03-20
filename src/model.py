import torch
from torchsummary import summary
from config import cfg
from torchvision.models import resnet50


class VGGnet(torch.nn.Module):
    '''
    this class implement the model VGGnet

    args:
        None

    return:
        None
    '''
    def __init__(self):
        super(VGGnet, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=3, padding='same').cuda()
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, padding='same').cuda()

        self.conv3 = torch.nn.Conv2d(4, 8, kernel_size=3, padding='same').cuda()
        self.conv4 = torch.nn.Conv2d(8, 8, kernel_size=3, padding='same').cuda()

        self.conv5 = torch.nn.Conv2d(8, 16, kernel_size=3, padding='same').cuda()
        self.conv6 = torch.nn.Conv2d(16, 16, kernel_size=3, padding='same').cuda()

        self.conv7 = torch.nn.Conv2d(16, 32, kernel_size=3, padding='same').cuda()
        self.conv8 = torch.nn.Conv2d(32, 32, kernel_size=3, padding='same').cuda()

        self.conv9 = torch.nn.Conv2d(32, 64, kernel_size=3, padding='same').cuda()
        self.conv10 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same').cuda()

        self.conv11 = torch.nn.Conv2d(64, 128, kernel_size=3, padding='same').cuda()
        self.conv12 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same').cuda()

        self.dense1 = torch.nn.Linear(6144, 512, bias=True).cuda()
        self.dense2 = torch.nn.Linear(512, 128, bias=True).cuda()
        self.dense3 = torch.nn.Linear(128, 6, bias=True).cuda()

        self.maxpooling = torch.nn.MaxPool2d(2, stride=2).cuda()
        self.relu = torch.nn.ReLU().cuda()

        self.dropout = torch.nn.Dropout(0.1)

    '''
    this function is made to compute prediction using the given batch

    args:
        x: torch tensor representing one batch of data
    
    return:
        x: torch tensor which contains a batch of prediction
    '''
    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.dense3(x)

        return x

'''
this function is made to build one VGGnet model

args:
    pretrained_weights: path of the checkpoint file you want to use to load your model's weights

return:
    model: one instance of VGGnet model
''' 
def get_vgg(pretrained_weights=None):
    torch.manual_seed(0)
    model = VGGnet()

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    if torch.cuda.is_available():
        model.cuda()

    return model

class MobileNet(torch.nn.Module):

    '''
    this class implement the model mobilenet

    args:
        None

    return:
        None
    '''
    def __init__(self):
        super(MobileNet, self).__init__()

        self.deepwise_conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding='same', groups=3).cuda()
        self.pointwise_conv2 = torch.nn.Conv2d(3, 64, kernel_size=1, padding='same').cuda()
        self.bn1 = torch.nn.BatchNorm2d(3).cuda()
        self.bn2 = torch.nn.BatchNorm2d(64).cuda()

        self.deepwise_conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same', groups=64).cuda()
        self.pointwise_conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same').cuda()
        self.bn3 = torch.nn.BatchNorm2d(64).cuda()
        self.bn4 = torch.nn.BatchNorm2d(128).cuda()

        self.deepwise_conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=128).cuda()
        self.pointwise_conv6 = torch.nn.Conv2d(128, 256, kernel_size=1, padding='same').cuda()
        self.bn5 = torch.nn.BatchNorm2d(128).cuda()
        self.bn6 = torch.nn.BatchNorm2d(256).cuda()

        self.deepwise_conv7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256).cuda()
        self.pointwise_conv8 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same').cuda()
        self.bn7 = torch.nn.BatchNorm2d(256).cuda()
        self.bn8 = torch.nn.BatchNorm2d(256).cuda()

        self.deepwise_conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256).cuda()
        self.pointwise_conv10 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same').cuda()
        self.bn9 = torch.nn.BatchNorm2d(256).cuda()
        self.bn10 = torch.nn.BatchNorm2d(256).cuda()

        self.deepwise_conv11 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256).cuda()
        self.pointwise_conv12 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same').cuda()
        self.bn11 = torch.nn.BatchNorm2d(256).cuda()
        self.bn12 = torch.nn.BatchNorm2d(256).cuda()

        # self.dense1 = torch.nn.Linear(18432, 512, bias=True).cuda()
        self.dense2 = torch.nn.Linear(4096, 128, bias=True).cuda()
        self.dense3 = torch.nn.Linear(128, 6, bias=True).cuda()

        self.maxpooling = torch.nn.MaxPool2d(2, stride=2).cuda()
        self.relu = torch.nn.ReLU().cuda()

        self.dropout = torch.nn.Dropout(0.1)

    '''
    this function is made to compute prediction using the given batch

    args:
        x: torch tensor representing one batch of data
    
    return:
        x: torch tensor which contains a batch of prediction
    '''
    def forward(self, x):

        # x.shape == (512, 384)
        x = self.deepwise_conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pointwise_conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (256, 192)
        x = self.deepwise_conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.pointwise_conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (128, 96)
        x = self.deepwise_conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.pointwise_conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (64, 48)
        x = self.deepwise_conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.pointwise_conv8(x)
        x = self.relu(x)
        x = self.bn8(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (32, 24)
        x = self.deepwise_conv9(x)
        x = self.relu(x)
        x = self.bn9(x)
        x = self.pointwise_conv10(x)
        x = self.relu(x)
        x = self.bn10(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (16, 12)
        x = self.deepwise_conv11(x)
        x = self.relu(x)
        x = self.bn11(x)
        x = self.pointwise_conv12(x)
        x = self.relu(x)
        x = self.bn12(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (8, 6)
        x = torch.flatten(x, start_dim=1)

        # x = self.dense1(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.dense3(x)

        return x

'''
this function is made to build one VGGnet model

args:
    pretrained_weights: path of the checkpoint file you want to use to load your model's weights

return:
    model: one instance of mobilenet model
'''  
def get_mobilenet(pretrained_weights=None):
    torch.manual_seed(0)
    model = MobileNet()

    summary(model, (3, cfg.TRAIN.IMAGE_SHAPE[0], cfg.TRAIN.IMAGE_SHAPE[1]))

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    if torch.cuda.is_available():
        model.cuda()

    return model


class resnet(torch.nn.Module):

    '''
    this class implement the model resnet

    args:
        None

    return:
        None
    '''
    def __init__(self):
        super(resnet, self).__init__()

        self.pointwise0 = torch.nn.Conv2d(3, 4, kernel_size=1, padding='same').cuda()
        self.conv1 = torch.nn.Conv2d(4, 4, kernel_size=3, padding='same').cuda()
        self.conv2 = torch.nn.Conv2d(4, 4, kernel_size=3, padding='same').cuda()
        self.bn1 = torch.nn.BatchNorm2d(4).cuda()
        self.bn2 = torch.nn.BatchNorm2d(4).cuda()

        self.pointwise1 = torch.nn.Conv2d(4, 8, kernel_size=1, padding='same').cuda()
        self.conv3 = torch.nn.Conv2d(8, 8, kernel_size=3, padding='same').cuda()
        self.conv4 = torch.nn.Conv2d(8, 8, kernel_size=3, padding='same').cuda()
        self.bn3 = torch.nn.BatchNorm2d(8).cuda()
        self.bn4 = torch.nn.BatchNorm2d(8).cuda()

        self.pointwise2 = torch.nn.Conv2d(8, 16, kernel_size=1, padding='same').cuda()
        self.conv5 = torch.nn.Conv2d(16, 16, kernel_size=3, padding='same').cuda()
        self.conv6 = torch.nn.Conv2d(16, 16, kernel_size=3, padding='same').cuda()
        self.bn5 = torch.nn.BatchNorm2d(16).cuda()
        self.bn6 = torch.nn.BatchNorm2d(16).cuda()

        self.pointwise3 = torch.nn.Conv2d(16, 32, kernel_size=1, padding='same').cuda()
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=3, padding='same').cuda()
        self.conv8 = torch.nn.Conv2d(32, 32, kernel_size=3, padding='same').cuda()
        self.bn7 = torch.nn.BatchNorm2d(32).cuda()
        self.bn8 = torch.nn.BatchNorm2d(32).cuda()

        self.pointwise4 = torch.nn.Conv2d(32, 64, kernel_size=1, padding='same').cuda()
        self.conv9 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same').cuda()
        self.conv10 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same').cuda()
        self.bn9 = torch.nn.BatchNorm2d(64).cuda()
        self.bn10 = torch.nn.BatchNorm2d(64).cuda()

        self.pointwise5 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same').cuda()
        self.conv11 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same').cuda()
        self.conv12 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same').cuda()
        self.bn11 = torch.nn.BatchNorm2d(128).cuda()
        self.bn12 = torch.nn.BatchNorm2d(128).cuda()

        self.dense1 = torch.nn.Linear(128, 64, bias=True).cuda()
        self.dense2 = torch.nn.Linear(64, 32, bias=True).cuda()
        self.dense3 = torch.nn.Linear(32, 6, bias=True).cuda()

        self.relu = torch.nn.ReLU().cuda()
        self.avgpool = torch.nn.AvgPool2d((int(384),int(512)))

        # self.softmax = torch.nn.Softmax(dim=-1)

        # self.dropout = torch.nn.Dropout(0.1)

    '''
    this function is made to compute prediction using the given batch

    args:
        x: torch tensor representing one batch of data
    
    return:
        x: torch tensor which contains a batch of prediction
    '''
    def forward(self, x):

        x=self.pointwise0(x)
        # x.shape == (512, 384)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)

        x1prime = x + x1
        x1prime = self.relu(x1prime)
        # x2 = self.dropout(x2)

        x1prime=self.pointwise1(x1prime)
        # x.shape == (256, 192)
        x2 = self.conv3(x1prime)
        x2 = self.bn3(x2)
        x2 = self.relu(x2)
        x2 = self.conv4(x2)
        x2 = self.bn4(x2)

        x2prime = x1prime + x2
        x2prime = self.relu(x2prime)
        # x4 = self.dropout(x4)

        x2prime=self.pointwise2(x2prime)
        # x.shape == (128, 96)
        x3 = self.conv5(x2prime)
        x3 = self.bn5(x3)
        x3 = self.relu(x3)
        x3 = self.conv6(x3)
        x3 = self.bn6(x3)

        x3prime = x2prime + x3
        x3prime = self.relu(x3prime)
        # x6 = self.dropout(x6)

        x3prime=self.pointwise3(x3prime)
        # x.shape == (64, 48)
        x4 = self.conv7(x3prime)
        x4 = self.bn7(x4)
        x4 = self.relu(x4)
        x4 = self.conv8(x4)
        x4 = self.bn8(x4)
        x4prime = x3prime + x4
        x4prime = self.relu(x4prime)
        # x8 = self.dropout(x8)

        x4prime=self.pointwise4(x4prime)
        # x.shape == (32, 24)
        x5 = self.conv9(x4prime)
        x5 = self.bn9(x5)
        x5 = self.relu(x5)
        x5 = self.conv10(x5)
        x5 = self.bn10(x5)
        x5prime = x4prime + x5
        x5prime = self.relu(x5prime)
        # x = self.dropout(x)

        x5prime=self.pointwise5(x5prime)
        # x.shape == (16, 12)
        x6 = self.conv11(x5prime)
        x6 = self.bn11(x6)
        x6 = self.relu(x6)
        x6 = self.conv12(x6)
        x6 = self.bn12(x6)
        x6prime = x5prime + x6
        x6prime = self.relu(x6prime)
        # x = self.dropout(x)

        x12 = self.avgpool(x6prime)

        # x.shape == (8, 6)
        x12 = torch.flatten(x12, start_dim=1)

        x12 = self.dense1(x12)
        x12 = self.relu(x12)
        # x12 = self.dropout(x12)

        x12 = self.dense2(x12)
        x12 = self.relu(x12)

        x12 = self.dense3(x12)
        # x12 = self.softmax(x12)

        return x12
    
'''
this function is made to build one VGGnet model

args:
    pretrained_weights: path of the checkpoint file you want to use to load your model's weights

return:
    model: one instance of resnet model
''' 
def get_resnet(pretrained_weights=None):
    torch.manual_seed(0)
    model = resnet()

    summary(model, (3, 384, 512))

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    if torch.cuda.is_available():
        model.cuda()

    return model


class resnet_pretrained(torch.nn.Module):
    '''
    this class implement the model resnet with pretrained weight

    args:
        None

    return:
        None
    '''
    def __init__(self):
        super(resnet_pretrained, self).__init__()

        self.resnet = resnet50(weights=cfg.TRAIN.RESENET50_WEIGHTS).cuda()
        self.resnet.fc = torch.nn.Identity()

        self.dense1 = torch.nn.Linear(2048, 512, bias=True).cuda()
        self.dense2 = torch.nn.Linear(512, 128, bias=True).cuda()
        self.dense3 = torch.nn.Linear(128, 6, bias=True).cuda()

        self.relu = torch.nn.ReLU().cuda()

        self.dropout = torch.nn.Dropout(0.1)

    '''
    this function is made to compute prediction using the given batch

    args:
        x: torch tensor representing one batch of data
    
    return:
        x: torch tensor which contains a batch of prediction
    '''
    def forward(self, x):

        x = self.resnet(x)
        x = self.dropout(x)

        # x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.dense3(x)

        return x

'''
this function is made to build one resnet_pretrained model

args:
    pretrained_weights: path of the checkpoint file you want to use to load your model's weights

return:
    model: one instance of VGGnet model
''' 
def get_resnet_pretrained(pretrained_weights=None):
    torch.manual_seed(0)
    model = resnet_pretrained()

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    if torch.cuda.is_available():
        model.cuda()

    return model