import torch
from tensorboardX import SummaryWriter

writer = SummaryWriter()

##add scalar
for n_iter in range(10):
	scalar = torch.rand(1)
	writer.add_scalar('data/scalar', scalar[0], n_iter)


##add scalars
for n_iter in range(10):
	x = torch.Tensor([n_iter])
	writer.add_scalars('data/scalar_group', {'sinx': torch.sin(x)[0],'cosx':torch.cos(x)[0],'atanx': torch.atan(x)[0]}, n_iter)


##add image

from PIL import Image
import torchvision.transforms.functional as TF


demopic = Image.open("demopic.jpg")
image_tensor = TF.to_tensor(demopic)
writer.add_image('Image',image_tensor)


##add images
image_tensors = torch.Tensor([image_tensor.numpy(),image_tensor.numpy()])
writer.add_image('Images',image_tensors)



writer.close()