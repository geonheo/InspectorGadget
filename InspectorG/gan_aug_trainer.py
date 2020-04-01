import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import shutil
import numpy as np
import pickle
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn.utils.spectral_norm as spectral_norm
import argparse
from matplotlib import pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--vector_size", type=int, default=100, help="size of input random noise")
parser.add_argument("--lr_G", type=float, default=1e-4, help="learning rate of Generator")
parser.add_argument("--lr_D", type=float, default=1e-4, help="learning rate of Discriminator")
parser.add_argument("--d_image", type=int, default=3, help="dimension of image (grayscale : 1, RGB : 3)")
parser.add_argument("--resize", type=int, default=0, help="resize image size")
parser.add_argument("--path", type=str, default="", help="image folder path. ex) folder1/folder2 (images in this folder)")
parser.add_argument("--c_period", type=int, default=10, help="how often do you want to check the training? (if c_period == 10, draw the output of G every 10 epochs")
opt = parser.parse_args()
print(opt)

save_folder = 'Save_folder/'
g_save_dir = os.path.join(save_folder, 'G_Model_Save/')
d_save_dir = os.path.join(save_folder, 'D_Model_Save/')
output_save_dir = os.path.join(save_folder, 'Output_Save/')

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
if not os.path.isdir(g_save_dir):
    os.mkdir(g_save_dir)
if not os.path.isdir(d_save_dir):
    os.mkdir(d_save_dir)
if not os.path.isdir(output_save_dir):
    os.mkdir(output_save_dir)
    
size_x = []
size_y = []    
dir_subfolder = os.path.join(opt.path, 'subfolder/')
if os.path.isdir(dir_subfolder):
    image_list = os.listdir(dir_subfolder)
    for image_name in image_list :
        image = Image.open(dir_subfolder + image_name)
        shape = np.shape(image)
        size_x.append(shape[0])
        size_y.append(shape[1])        
    
else :
    os.mkdir(dir_subfolder)
    image_list = os.listdir(opt.path)
    for image_name in image_list :
        if image_name != 'subfolder' :
            image = Image.open(opt.path + image_name)
            shape = np.shape(image)
            size_x.append(shape[0])
            size_y.append(shape[1])
            shutil.move(opt.path + image_name, dir_subfolder + image_name)

np.save(save_folder + 'original_image_size_x_list', size_x)
np.save(save_folder + 'original_image_size_y_list', size_y)


# GPU number used to execute code
os.environ["CUDA_VISIBLE_DEVICES"]="1"
if torch.cuda.is_available():
    use_gpu = True

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=opt.vector_size, out_channels=128, 
                               kernel_size=5, stride=1, padding=0, 
                               bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, 
                               kernel_size=3, stride=2, padding=0, 
                               bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                               kernel_size=4, stride=2, padding=0, 
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                               kernel_size=3, stride=2, padding=0, 
                               bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, 
                               kernel_size=4, stride=2, padding=0, 
                               bias=False),
            nn.Tanh()
        )
    def forward(self, inputs):
        out = self.main(inputs.view(-1, opt.vector_size, 1, 1))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=opt.d_image, out_channels=128, 
                      kernel_size=4, stride=2, padding=0, 
                      bias=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=128, 
                      kernel_size=3, stride=2, padding=0, 
                      bias=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, 
                      kernel_size=4, stride=2, padding=0, 
                      bias=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=32, 
                      kernel_size=3, stride=2, padding=0, 
                      bias=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=1, 
                      kernel_size=5, stride=1, padding=0, 
                      bias=False))
        )
    def forward(self, inputs):
        out = self.main(inputs)
        return out.view(-1, 1)

# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

G = Generator()
D = Discriminator()

G.apply(weights_init)
D.apply(weights_init)

if use_gpu:
    G.cuda()
    D.cuda()

G_optimizer = Adam(G.parameters(), lr=opt.lr_G, betas=(0.5, 0.999))
D_optimizer = Adam(D.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5,), std=(0.5,)) 
])

transform_color = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 
])

if opt.resize > 0 :
    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((opt.resize, opt.resize)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5,), std=(0.5,)) 
    ])
    transform_color = transforms.Compose([
            transforms.Resize((opt.resize, opt.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

# Training dataset loader
if opt.d_image == 1:
    dataset = dset.ImageFolder(root=opt.path, transform=transform)
else :
    dataset = dset.ImageFolder(root=opt.path, transform=transform_color)
                         
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True)

# fixed vector for check the training process
with torch.no_grad():
    z_fixed = Variable(torch.randn((1, opt.vector_size)))
    
if use_gpu:
    z_fixed = z_fixed.cuda()

# loss function type
BCE_stable = torch.nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

if use_gpu :
    criterion = criterion.cuda()
    BCE_stable = BCE_stable.cuda()
    

# target values for training
# target_real : indicate real
# target_fake : indicate fake
target_real = Variable(torch.ones(opt.batch_size, 1))
target_fake = Variable(torch.zeros(opt.batch_size, 1))

if use_gpu:
    target_real, target_fake = target_real.cuda(), target_fake.cuda()
        
for epoch in range(opt.n_epochs):
    D_loss_list = []
    G_loss_list = []
    for real_data, _ in dataloader:
        
        real_data = Variable(real_data)
        # input random noise vector
        z = Variable(torch.randn((opt.batch_size, opt.vector_size)))
        
        if use_gpu:
            real_data = real_data.cuda()
            z = z.cuda()

        # Discriminator training
        D.zero_grad()
        d_real = D(real_data)
        fake_data = G(z)
        d_fake = D(fake_data.detach())

        d_loss = (BCE_stable(d_real - torch.mean(d_fake), target_real) + BCE_stable(d_fake - torch.mean(d_real), target_fake))/2
        d_loss.backward()
        D_optimizer.step()
        D_loss_list.append(d_loss.cpu().data.numpy())
        
        # Generator training
        G.zero_grad()
        d_fake = D(fake_data)
        d_real = D(real_data)
        r_to_f = d_real - torch.mean(d_fake)
        f_to_r = d_fake - torch.mean(d_real)

        g_loss = (BCE_stable(r_to_f, target_fake) + BCE_stable(f_to_r, target_real))/2
        g_loss.backward()
        G_optimizer.step()
        G_loss_list.append(g_loss.cpu().data.numpy())

    print("[%d/%d] D_loss : %f\t G_loss : %f" %(epoch+1, opt.n_epochs, np.mean(D_loss_list), np.mean(G_loss_list)))
    if (epoch+1) % opt.c_period == 0 :
        # To verify the training precess by watching the generator output
        output = G(z_fixed)
        image_size = real_data.size(2)
        if opt.d_image == 1 :
            outputs = output.view(image_size, image_size).cpu().data.numpy()
            denorm_image = interval_mapping(outputs, -1.0, 1.0, 0, 255).astype('uint8')
            plt.imshow(denorm_image, cmap = 'gray', interpolation='nearest')
        else :
            outputs = output.view(opt.d_image, image_size, image_size).cpu().data.numpy()
            denorm_image = interval_mapping(outputs, -1.0, 1.0, 0, 255).astype('uint8')
            denorm_image = np.transpose(denorm_image, (1,2,0))
            plt.imshow(denorm_image, interpolation='nearest')        
            
        plt.savefig(output_save_dir + str(epoch+1) + '_epoch.png')
        # For backup Generator and Discriminator model
        torch.save(G.state_dict(), g_save_dir + "Generator_" + str(epoch+1) + "_epoch.pkl")
        torch.save(D.state_dict(), d_save_dir + "Discriminator_" + str(epoch+1) + "_epoch.pkl")

# Model Save
torch.save(G.state_dict(), g_save_dir + "Generator.pkl")
torch.save(D.state_dict(), d_save_dir + "Discriminator.pkl")

image_list = os.listdir(dir_subfolder)
for image_name in image_list :
    shutil.move(dir_subfolder + image_name, opt.path + image_name)
os.rmdir(dir_subfolder)