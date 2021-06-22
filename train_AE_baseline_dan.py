
import os
import atexit

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

if not os.path.exists('./img_viz'):
    os.mkdir('./img_viz')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(-1, 1)
    x = x.view(x.size(0), 3, SIZE_Y, SIZE_X)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# SIZE_X, SIZE_Y = 126, 256
SIZE_X, SIZE_Y = 64, 126
# SIZE_X, SIZE_Y = 32, 64

img_transform = transforms.Compose([
    transforms.Resize((SIZE_Y, SIZE_X)),
    # transforms.Resize((112,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ImageDataset(Dataset):
    def __init__(self, flist, transform=None):
        self.imlist = flist        
        self.transform = transform

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = Image.open(os.path.join(impath)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img#[img, impath]

    def __len__(self):
        return len(self.imlist)

# dataset = MNIST('./data', transform=img_transform)
file_list = [f'./DATASET/{f}' for f in os.listdir('./DATASET') if ".png" in f]
dataset = ImageDataset(file_list, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(SIZE_Y * SIZE_X * 3, 11200),
            nn.ReLU(True),
            nn.Linear(11200, 2048),
            nn.ReLU(True), 
            nn.Linear(2048, 256), 
            nn.ReLU(True), 
            nn.Linear(256, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 11200), 
            nn.ReLU(True), 
            nn.Linear(11200, SIZE_Y * SIZE_X * 3), 
            nn.Tanh()
            )
    def forward(self, x):
        x_enc = self.encoder(x)
        x_rec = self.decoder(x_enc)
        return x_enc, x_rec


model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
writer = SummaryWriter(log_dir="/home/druta/ui/fasterrcnn/img_viz")

data_i = 0
for epoch in range(num_epochs):
    for data in dataloader:

        print(f'\r{data_i%len(dataloader)*batch_size}/{len(dataloader)*batch_size}', end="", flush=True)
        data_i += 1

        img = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        #writer.add_scalar("loss", loss.item(), data_i)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('\nEpoch [{}/{}]\t Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    #writer.add_scalar("epoch_loss", loss.item(), epoch)

    pic = to_img(output.cpu().data)
    save_image(pic, './img_viz/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './autoencoder.pth')

#def exit_handler (self):
#    print("\nExiting script...")
#    writer.flush()
#    writer.close()
#atexit.register(exit_handler)