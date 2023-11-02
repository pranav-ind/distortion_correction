import torch
import napari
import kornia as K
from tkinter import filedialog as fd
import kornia as K
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib.ticker import FormatStrFormatter




def show_result_from_file():
  temp = "a"
  while (temp != 'q'):
    tensor_name = fd.askopenfilename()
    tensor = torch.load(tensor_name)
    viewer_3d([tensor], [tensor_name[-5:]])
    temp = input("Press any key to continue, q to exit")






def plt_images(slices_list, z,title,minmin,maxmax):
  """

  :param slices_list: Ex : [10,11,12,13,14,15]
  :param z: The list that contains all tensors. Ex : To plot (img+,img-) -> z = [img_pos,img_inv]
  :return:
  """

  # minmin = torch.min(torch.min(z[0]), torch.min(z[-1]))
  # maxmax = torch.max(torch.max(z[0]), torch.max(z[-1]))
  slices = slices_list
  i = 0
  rows = len(z)
  cols = len(slices)
  f, ax = plt.subplots(rows, cols)
  f.set_figheight(8)
  f.set_figwidth(16)
  f.suptitle(title,fontsize=18)


  for r in range(rows):
    for c in range(cols):
      temp = z[r]
      slice_num = slices[i]
      temp = ax[r, c].imshow(temp[slice_num], cmap='gray',vmin=minmin,vmax=maxmax)
      ax[r, c].set_title("Slice = " + str(slice_num))
      ax[r, c].axis('off')

      cbar = f.colorbar(temp,ticklocation="bottom")
      cbar.ax.tick_params(labelsize=7)
      # cbar.set_ticks(ticks=[minmin,maxmax], labels = [float("{:.f}".format(minmin)),float("{:.4f}".format(maxmax))])

      i += 1
    i = 0
  plt.show()




def complex_pyramid(input, n=4):
  """
  Builds a Gaussian Image pyramid for complex torch tensors
  :param input: Complex Torch Tensor
  :param n: No. of resolution layers
  :return: List containing pyramids (dtype = complex64)
  """
  output = []
  real_input = torch.real(input)
  imag_input = torch.imag(input)
  real_output = K.geometry.transform.build_pyramid(real_input, n)
  imag_output = K.geometry.transform.build_pyramid(imag_input, n)

  for i in range((n)):
    output.append(torch.complex(real_output[i], imag_output[i]))

  return output



def viewer_3d(img_list,title_list):
  """
  Parse the images as a list and image titles as a list of strings
  :param img_list: List of images to be viewed
  :param title_list: List of titles to be displayed
  :return: all the images in napari viewer
  """

  temp_img_list = img_list.copy()
  viewer_list = img_list.copy()

  for i in range(len(img_list)):
    temp_img_list[i] = img_list[i].clone()
    viewer_list[i] = napari.view_image(temp_img_list[i].detach().numpy(), name=title_list[i])
  napari.run()




def complex_NLL(input,model_variable):
  """
  Complex Utility Function for Gaussian Negative Log Likelihood function
  :param input:
  :param model_variable:
  :return:
  """
  loss = torch.nn.GaussianNLLLoss()

  real_input = torch.real(input)
  imag_input = torch.imag(input)

  mv_real = torch.real(model_variable)
  mv_imag = torch.imag(model_variable)


  real_output = loss(real_input, mv_real, torch.ones_like(mv_real)*0.1)
  imag_output = loss(imag_input, mv_imag, torch.ones_like(mv_imag)*0.1)

  print(real_output.shape)
  output = torch.abs(torch.complex(real_output, imag_output))
  

  return output



def complex_MSE(input,model_variable):
  """
  Complex Utility Function for Gaussian Negative Log Likelihood function
  :param input:
  :param model_variable:
  :return:
  """
  loss = torch.nn.MSELoss()

  real_input = torch.real(input)
  imag_input = torch.imag(input)

  mv_real = torch.real(model_variable)
  mv_imag = torch.imag(model_variable)


  real_output = loss(real_input, mv_real)
  imag_output = loss(imag_input, mv_imag)


  output = torch.abs(torch.complex(real_output, imag_output))

  return output





def complex_total_variation(input):
  """
  Complex Utility Function for Gaussian Negative Log Likelihood function
  :param input:
  :param model_variable:
  :return:
  """
  loss = K.losses.TotalVariation()

  real_output = loss(torch.real(input))
  imag_output = loss(torch.imag(input))

  output = torch.abs(torch.complex(real_output, imag_output))

  return output.mean()

def show_mv(mv1, mv2):
  f, ax = plt.subplots(2, 2)
  f.set_figheight(9)
  f.set_figwidth(16)
  mv1 = torch.fft.ifftn(mv1)
  mv2 = torch.fft.ifftn(mv2)
  mv1 = torch.abs(mv1.reshape(mv1.shape[-3], mv1.shape[-1], mv1.shape[-1]).detach())
  mv2 = torch.abs(mv2.reshape(mv2.shape[-3], mv2.shape[-1], mv2.shape[-1]).detach())
  ax[0, 0].imshow(mv1[17], cmap='gray')
  ax[0, 1].imshow(mv1[20], cmap='gray')
  ax[1, 0].imshow(mv2[17], cmap='gray')
  ax[1, 1].imshow(mv2[20], cmap='gray')
  plt.show()
  # print('abc')




def forward_model_multi_resolution(w,template_img):
  #W = Transformation matrix (Deformation matrix) (Warping matrix) (theta)
  img = template_img.reshape(1,template_img.shape[-3],template_img.shape[-1],template_img.shape[-1])

  #Warping
  img = w(img)
  #ift = torch.matmul(K,img())
  img = torch.fft.fftn(img)
  return img


def optimize_multi_res(epochs, learning_rate, w, template_img, f_pos_pyramid, f_inv_pyramid):

  template_image_list = []
  w_list = []
  losses = []
  loss = torch.nn.GaussianNLLLoss()
  total_variation = K.losses.TotalVariation()
  # print(template_img.shape,template_img.is_leaf)

  for i in reversed(range(4)):
    # Initialize optimizer
    # tv_denoiser = TVDenoise(template_img)

    optimizer = torch.optim.LBFGS([w, template_img], lr=learning_rate, max_iter=20)
    for t in tqdm(range(epochs)):
      # w = f1(g)
      def closure():
        optimizer.zero_grad()
        model_variable_1 = forward_model_multi_resolution(w, template_img)  # 1
        model_variable_2 = forward_model_multi_resolution(-w, template_img)  # 1
        l1 = complex_NLL(f_pos_pyramid[i], model_variable_1)  # 2.1
        l2 = complex_NLL(f_inv_pyramid[i], model_variable_2)  # 2.2
        l3 = complex_total_variation(model_variable_1) + complex_total_variation(model_variable_2)

        NLL = l1 + l2 + (5e-1 * l3)

        losses.append(NLL.item())  # To plot the loss graph
        NLL.backward()
        return NLL

      optimizer.step(closure)

    template_image_list.append(template_img)
    w_list.append(w)

    # loss_list.append(losses)

    if (i != 0):
      upsampler = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      template_img = upsampler(template_img)
      w = w.up()
      # f1 = f1.up()
      # g = g.up()
    # T =  (T * torch.rand_like(T,requires_grad=True))

    if (i == 3):  # Because one of the images is shaped 27 therefore padding it by 1
      pad = (1, 0, 1, 0)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
      template_img = F.pad(template_img, pad, "constant", 0)
      w = F.pad(w, pad, "constant", 0)
      # f1 = F.pad(f1, pad, "constant", 0)
      # g = F.pad(g, pad, "constant", 0)
    # print(template_img.shape)
    template_img.retain_grad()
    w.retain_grad()
    # f1.retain_grad()
    # g.retain_grad()
  return template_image_list, w_list, losses




def plot_distortion_field(w):
    temp = w.reshape(2,w.shape[-2],w.shape[-1]).detach()
    # temp = w_list[-2].reshape(2,116,116).detach()
    # cmap = plt.get_cmap("Oranges")
    # colors = cmap([0.2,0.4,0.6,0.8])
    # u,v = np.meshgrid(temp[0],temp[1])
    dx = temp[0]
    dy = temp[1]
    q = plt.quiver(dx,dy,cmap='gray')
    plt.colorbar(q)
    plt.show()
