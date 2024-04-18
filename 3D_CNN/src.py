
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm




def CNN_loss(recon_x, x, mu, logvar, KL_weight = 1.0):
    #BCE_shape = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #BCE_color = 
    #SUM = BCE_shape + BCE_color
    #return SUM
    return None 


def CNN_train(model, dataloader, criterion, optimizer, epochs = 5, device = 'cpu'):

    pbar_epoch = tqdm(range(epochs), desc='Epochs')

    for epoch in pbar_epoch:
        running_loss = 0.0
        running_KL = 0.0
        running_recon = 0.0


        for images, labels in dataloader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            recon_x = outputs[0]
            mu = outputs[1]
            logvar = outputs[2]
            loss  = criterion(recon_x=recon_x, x = images, mu = mu, logvar = logvar)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)



        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_KL = running_KL / len(dataloader.dataset)
        epoch_recon = running_recon / len(dataloader.dataset)
        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.5f}, KL: {epoch_KL:.5f}, Recon: {epoch_recon:.5f}")    
        #set postfix with .5f to limit the number of decimals
        pbar_epoch.set_postfix({'Loss': f'{epoch_loss:.5f}', 'KL': f'{epoch_KL:.5f}', 'Recon': f'{epoch_recon:.5f}'})


class FiguresDataset3D(torch.utils.data.Dataset):
    def __init__(self, num_samples=10000, image_size=(28, 28, 28)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, label = self.generate_image()
        image_tensor = torch.from_numpy(image).permute(3, 0, 1, 2).float() / 255.0
        label_tensor = torch.tensor(label)
        return image_tensor, label_tensor

    def generate_image(self):
        image = np.ones((self.image_size[0], self.image_size[1], self.image_size[2], 3), dtype=np.uint8) * 255  # White background

        shape = np.random.choice(['cube','sphere',  'tetrahedron'])
        color = self.choose_color()
        size_x = np.random.randint(3, self.image_size[0] // 2)
        size_y = np.random.randint(3, self.image_size[1] // 2)
        size_z = np.random.randint(3, self.image_size[2] // 2)
        pos_x = np.random.randint(0, self.image_size[1] - size_x)
        pos_y = np.random.randint(0, self.image_size[0] - size_y)
        pos_z = np.random.randint(0, self.image_size[2] - size_z)

        if shape == 'cube':
            image = self.draw_cube(image, x=pos_x, y=pos_y, z=pos_z, size_x=size_x, size_y=size_y, size_z=size_z, color=color)
        elif shape == 'sphere':
            image = self.draw_sphere(image, x=pos_x, y=pos_y, z=pos_z, radius=size_x // 2, color=color)
        else:
            image = self.draw_tetrahedron(image, x=pos_x, y=pos_y, z=pos_z, size=size_x, color=color)

        # label is the shape, color, size, and position of the shape
        shape_dict = {'cube': 0, 'sphere': 1,'tetrahedron': 2}
        label = np.array([shape_dict[shape], size_x, size_y, size_z, pos_x, pos_y, pos_z, color[0], color[1], color[2]])

        return image, label

    def choose_color(self):
        color_r = np.array([255, 0, 0])
        color_g = np.array([0, 255, 0])
        color_b = np.array([0, 0, 255])
        rand_col = np.random.randint(0, 3)
        if rand_col == 0:
            return color_r
        elif rand_col == 1:
            return color_g
        else:
            return color_b

    def draw_cube(self, image, x, y, z, size_x, size_y, size_z, color):
        image[z:z+size_z, y:y+size_y, x:x+size_x] = color
        return image

    def draw_sphere(self, image, x, y, z, radius, color):
        yy, xx, zz = np.ogrid[-y:image.shape[0]-y, -x:image.shape[1]-x, -z:image.shape[2]-z]
        mask = xx*xx + yy*yy + zz*zz <= radius*radius
        image[mask] = color
        return image

    # def draw_triangle(self, image, x, y, size, color):
    #     image[y:y+size, x:x+size] = color

    #     # Determine whether to remove upper or lower half of the square
    #     if np.random.rand() < 0.5:
    #         # Remove upper half of the square
    #         for i in range(size):
    #             for j in range(size):
    #                 if i > j:
    #                     image[y+i, x+j] = 255  # Background color
    #     else:
    #         # Remove lower half of the square
    #         for i in range(size):
    #             for j in range(size):
    #                 if i < j:
    #                     image[y+i, x+j] = 255  # Background color
    #     return image

    def draw_tetrahedron(self, image, x, y, z, size, color):
        image[z:z+size, y:y+size, x:x+size] = color
        return image


def visualize_batch(images_batch):
    num_images = len(images_batch)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    for i in range(num_images):
        image = images_batch[i].permute(1, 2, 3, 0).numpy()  # Convert from tensor to numpy array and rearrange dimensions

        # Plot image
        axes[i].imshow(image)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# class FiguresDataset3D_tmp(torch.utils.data.Dataset):
#     def __init__(self, num_samples=10000, image_size=(28, 28, 28)):
#         self.num_samples = num_samples
#         self.image_size = image_size


#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         image, label = self.generate_image()
#         image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
#         label_tensor = torch.tensor(label)
#         return image_tensor, label_tensor

#     def generate_image(self):
#         image = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 255  # White background

#         shape = np.random.choice(['square', 'circle', 'triangle'])
#         color = self.choose_color()
#         size = np.random.randint(3, self.image_size[0] // 2)
#         pos_x = np.random.randint(0, self.image_size[1] - size)
#         pos_y = np.random.randint(0, self.image_size[0] - size)

#         if shape == 'square':
#             image = self.draw_square(image, x = pos_x, y = pos_y,  size=size, color=color)
#         elif shape=='circle':
#             image = self.draw_circle(image, x = pos_x, y = pos_y, radius=size//2, color=color)
#         else:
#             image = self.draw_triangle(image, x = pos_x, y = pos_y, size=size, color=color)

#         #label is the shape, color, size and position of the shape
#         shape_dict = {'square': 0, 'circle': 1, 'triangle': 2}
#         label = np.array([shape_dict[shape],  size, pos_x, pos_y, color[0], color[1], color[2]])


#         return image, label

#     def choose_color(self):
#         color_r = np.array([255, 0, 0])
#         color_g = np.array([0, 255, 0])
#         color_b = np.array([0, 0, 255])
#         rand_col = np.random.randint(0, 3)
#         if rand_col == 0:
#             return color_r
#         elif rand_col == 1:
#             return color_g
#         else:
#             return color_b


#     def draw_square(self, image, x, y, size, color):
#         image[y:y+size, x:x+size] = color
#         return image

#     def draw_triangle(self, image, x, y, size, color):
#         image[y:y+size, x:x+size] = color

#         # Determine whether to remove upper or lower half of the square
#         if np.random.rand() < 0.5:
#             # Remove upper half of the square
#             for i in range(size):
#                 for j in range(size):
#                     if i > j:
#                         image[y+i, x+j] = 255  # Background color
#         else:
#             # Remove lower half of the square
#             for i in range(size):
#                 for j in range(size):
#                     if i < j:
#                         image[y+i, x+j] = 255  # Background color
#         return image

#     def draw_circle(self, image, x, y, radius, color):

#         yy, xx = np.ogrid[-y:image.shape[0]-y, -x:image.shape[1]-x]
#         mask = xx*xx + yy*yy <= radius*radius
#         image[mask] = color
#         return image




