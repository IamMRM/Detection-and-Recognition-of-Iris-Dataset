import torch
from torchvision import transforms, datasets, models
import torch.nn as nn

def crop_img(tensor,target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = int(delta / 2)
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]


# In[ ]:


def double_conv2D(in_channel,out_channel):
    conv= nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel,out_channel,kernel_size=3),
        nn.ReLU(inplace=True))
    return conv


# In[ ]:


class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.initial_pad = nn.ZeroPad2d(92)
        self.conv1= double_conv2D(1,64)      
        self.maxpool2D = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2= double_conv2D(64,128)
        self.conv3= double_conv2D(128,256)
        self.conv4= double_conv2D(256,512)
        self.conv5= double_conv2D(512,1024)
        
        self.newup_1 = nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1)
        self.upconv1 = double_conv2D(1024,512)
        self.newup_2 = nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1)
        self.upconv2 = double_conv2D(512,256)
        self.newup_3 = nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1)
        self.upconv3 = double_conv2D(256,128)
        self.newup_4 = nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1)
        self.upconv4 = double_conv2D(128,64)
        self.newup_5 = nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)
        self.upconv5 = double_conv2D(64,32)
        
        self.out = nn.Conv2d(64,1,kernel_size=1)#change for output channels
        
    def forward(self,img):
        #encorder
        x0 = self.initial_pad(img)
        x1 = self.conv1(x0)
        x2 = self.maxpool2D(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool2D(x3)
        x5 = self.conv3(x4)
        x6 = self.maxpool2D(x5)
        x7 = self.conv4(x6)
        x8 = self.maxpool2D(x7)
        x9 = self.conv5(x8)
        
        #decorder
        x_1 = self.newup_1(x9)
        x7 = crop_img(x7,x_1)
        x_2 = self.upconv1(torch.cat([x_1,x7], 1))
        x_3 = self.newup_2(x_2)
        x5 = crop_img(x5,x_3)
        x_4 = self.upconv2(torch.cat([x_3,x5], 1))
        x_5 = self.newup_3(x_4)
        x3 = crop_img(x3,x_5)
        x_6 = self.upconv3(torch.cat([x_5,x3], 1))
        x_7 = self.newup_4(x_6)
        x1 = crop_img(x1,x_7)
        x_8 = self.upconv4(torch.cat([x_7,x1], 1))
        x = self.out(x_8)
        return x

#for testing
"""img = torch.rand((4,1,212,212))
model = Unet()
#   print(model)
print(model(img).shape)"""