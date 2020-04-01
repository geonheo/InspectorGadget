import torch
import os
import cv2
import numpy as np
import pickle
import torchvision
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--vector_size", type=int, default=100, help="size of input random noise")
parser.add_argument("--d_image", type=int, default=3, help="dimension of image (grayscale : 1, RGB : 3)")
parser.add_argument("--image_size", type=int, default=100, help="image size")
parser.add_argument("--save_path", type=str, default="Generated_image/", help="image save folder")
parser.add_argument("--save_folder_path", type=str, default="Save_folder/", help="path of save_folder (the folder of package)")
parser.add_argument("--model_path", type=str, default="Save_folder/G_Model_Save/Generator.pkl", help="path of generator model (pkl file)")
parser.add_argument("--g_num", type=int, default=10, help="How many images do you want to generated?")
opt = parser.parse_args()
print(opt)

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

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

G = Generator()
G.load_state_dict(torch.load(opt.model_path))


rand_seed = np.random.choice(60000, opt.g_num)
size_x = np.load(opt.save_folder_path + 'original_image_size_x_list.npy')
size_y = np.load(opt.save_folder_path + 'original_image_size_y_list.npy')

path = os.path.join(opt.save_folder_path, opt.save_path)
if not os.path.isdir(path):
    os.mkdir(path)
    
for i in range(opt.g_num) :
    random_seed = rand_seed[i]
    torch.manual_seed(random_seed)
    
    input_vector = torch.randn((1, opt.vector_size))
    output = G(input_vector)
    if opt.d_image == 1 :
        outputs = output.view(opt.image_size, opt.image_size).cpu().data.numpy()
        denorm_image = interval_mapping(outputs, -1.0, 1.0, 0, 255).astype('uint8')
    else :
        outputs = output.view(opt.d_image, opt.image_size, opt.image_size).cpu().data.numpy()
        denorm_image = interval_mapping(outputs, -1.0, 1.0, 0, 255).astype('uint8')
        denorm_image = np.transpose(denorm_image, (1,2,0))
        
    num_choice = len(size_x)
    size_pick = int(np.random.choice(num_choice, 1))
    a = size_x[size_pick]
    b = size_y[size_pick]

    generated_image = cv2.resize(denorm_image, (b,a), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(opt.save_folder_path + opt.save_path + str(i) + ".png", generated_image)
    print("%dth image is generated at %s" %(i+1, opt.save_folder_path + opt.save_path))

print("finish")