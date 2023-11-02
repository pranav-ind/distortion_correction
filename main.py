from data_loader import slicing
import torch
import matplotlib.pyplot as plt
import kornia as K
import torchfields
from tqdm import tqdm
import torchvision.transforms.functional as F1
from torchvision.transforms import v2

from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils import viewer_3d,show_mv,plt_images,complex_pyramid,complex_NLL,complex_total_variation
from data_loader import slicing


f_pos = torch.load('/Users/pi58/Library/CloudStorage/Box-Box/PhD/Low Field Data-1/20230213 - Chloe for Itamar/TSEV3_1_highGrad/1/k_space.t')
f_inv = torch.load("/Users/pi58/Library/CloudStorage/Box-Box/PhD/Low Field Data-1/20230213 - Chloe for Itamar/TSEV3_1_highGrad_invPol/1/k_space.t")
# img_inv = torch.flip(img_inv,[0])

img_pos = torch.fft.ifftshift(torch.fft.ifftn(f_pos))  # Inverse FFT
img_inv = torch.fft.ifftshift(torch.fft.ifftn(f_inv))

f_pos, f_inv, img_pos, img_inv = slicing(img_pos, img_inv)


slices = [15, 16, 17, 18, 19]
z = [torch.abs(img_pos), torch.abs(img_inv)]
plt_images(slices,z,"Row-1 : I+, Row-2 : I-Inv",0 , 0.002)
