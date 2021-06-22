"""
ResNet based AutoEncoder 

"""
# from torchsummary import summary
# summary(model.cuda(), (3,224,224))


from torchvision import models as models
import torch.nn as nn


model = models.resnet50(pretrained= True)
model = nn.Sequential(*list(model.children())[:-1])


class ResNetAutoEncoder(nn.Module):
   
    def __init__(self):
        super(ResNetAutoEncoder, self).__init__()
        model = models.resnet50(pretrained= True)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.FC1 = nn.Linear(2048,512)
        self.FC2 = nn.Linear(512,2048)
        

    def forward(self, x):
        x = self.model(x)
        x = self.FC1(x)
        
        