import torch
import pandas as pd
import numpy as np
import torchvision.transforms as Tr
import torch.nn.functional as F


def csv_to_df(file):
    """
    :param file: location of the csv file which is expected to have data complex data
    :return: a dataframe of size 220 columns(describing the complex pairs)
    """
    # Create a df and read entire data
    # df = pd.DataFrame(columns=list(range(1, 441)))
    df = pd.read_csv(file, header=None)

    # Create a new data frame for 220 columns
    new_df = pd.DataFrame(columns=list(range(1, 221)))
    # new_df[str(0)] = [1,2,3,4,5]
    new_df.columns = new_df.columns.astype(str)
    # new_df

    # Combine subsequent rows to make 440 columns to 220 columns
    df.columns = df.columns.astype(str)
    i = 1
    for k in range(0, len(df.keys()), 2):
        # print(k)
        df[str(k)] = pd.to_numeric(df[str(k)])  # Converting string to float
        df[str(k + 1)] = pd.to_numeric(df[str(k + 1)])
        new_df[str(i)] = df[str(k)] + 1j * df[str(k + 1)]
        i += 1
        i += 1

    return new_df


def df_to_tensor(new_df):
    """
    :param new_df: Takes the new_df from csv_to_df() and converts into a 3d tensor
    :return: Returns 3d tensor of shape (100,110,220) indicating 100 slices of shape (110,220)
    """
    img = torch.tensor(new_df.values)
    img = torch.reshape(img, (11000, 220))
    img = torch.reshape(img, (100, 110, 220))
    # print(img.shape)
    return img


def mag_img_2d(path1, path2, slice_num=46, img_size=110):
    """
    Takes the location of img+ and img_inv csv files and returns magnitude+ and magnitude- (2D k-space data) of a slice(2D)

    :param path1: Path to k-space data csv acquired in positive-encode direction
    :param path2: Path to k-space data csv acquired in negative-encode direction
    :param slice_num: The slice number that's to be worked on (In case of 2D); Default = 47th
    :param img_size: Size of the image for center crop; Default = 110
    :return: mag+ and mag-
    """
    img_ks = df_to_tensor(csv_to_df(path1))  # We have k-space data in 3D now
    img_inv_ks = df_to_tensor(csv_to_df(path2))
    img_inv_ks = torch.flip(img_inv_ks, [1, 2])

    mag_pos = (np.abs(img_ks))
    mag_inv = (np.abs(img_inv_ks))

    #Converting 3D to 2D
    mag_pos = mag_pos[slice_num, ]  # Taking slice_num+1 th slice
    mag_inv = mag_inv[slice_num, ]  # Taking slice_num+1 th slice

    # Transforming the (110,220) k-space data into (110,110) k-space data
    transform = Tr.CenterCrop(img_size)
    mag_pos = transform(mag_pos.float())
    mag_inv = transform(mag_inv.float())


    return mag_pos, mag_inv

def to_2d(k_space,slice_num=46):

    """
    Converts 3D k-space data to 2D image space by taking a slice, then cropping it to a square shape,
    and padding it and taking. Also converts 3D K-space data to 2D K-space

    :param k_space: 3D K-space data
    :param slice_num: the ID of the slice (i+1 th. For ex : 46 will return 47th slice)
    :return: 2D k-space data, 2D Image data (Magnitude only)
    """


    img = torch.fft.ifftn(k_space) #Inverse FFT
    img = torch.fft.ifftshift(img)
    img = np.abs(img)
    img = img[46,] #Taking a slice (converting 3d data to 2d)
    transform = Tr.CenterCrop(160)
    img = transform(img.float())
    pad = (10,10,10,10)
    img = F.pad(img,pad,"constant",0)

    f = torch.fft.fftn(img) #Convert this 2d slice to k-space

    return f, img


def kspace_image(k_space):

    # Takes 3D k-space data and returns the 3D FT and 3D Image
    img = torch.fft.ifftn(k_space)  # Inverse FFT
    img = torch.fft.ifftshift(img)
    # img = np.abs(img)
    transform = Tr.CenterCrop(220)
    img = transform(img.float())
    # pad = (10,10,10,10)
    # img = F.pad(img,pad,"constant",0)

    # f = torch.fft.fftn(img)  # Convert the image to k-space

    return img



def slicing(img_pos,img_inv):
    """
    This function is for Leiden Data with (100 slices)
    Takes Image+ and Image-1, slices the required slices and returns the sliced images along with their k-space data
    :param img_pos: Image in Phase-Encode direction
    :param img_inv: Image in Phase-Reversal direction
    :return: K-space (+ve), K-space (inv), Image(+), Image(inv)
    """
    img_pos = img_pos[25:60]  # Extracting slices 25-60 as remaining are the background slices and do not contain much information
    img_inv = torch.flip(img_inv, [0])  # Reordering the slices for corresponding images
    img_inv = img_inv[24:59, :, :] #Extracting required 35 slices only (-1 to match the slices with img_pos
    img_inv = torch.flip(img_inv, [1, 2]) #Rotating
    f_pos = torch.fft.fftn(img_pos) #Converting to k-space
    f_inv = torch.fft.fftn(img_inv)

    return f_pos,f_inv, img_pos,img_inv





def save_tensor(tensor,tensor_name):
    location = "/Users/pi58/Library/CloudStorage/Box-Box/PhD/Optimization/Non-Linear"
    torch.save(tensor,location+tensor_name+".t")

def load_tensor(tensor_name,location):
    x = torch.load(location+tensor_name+".t")
    return x


