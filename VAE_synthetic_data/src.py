
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm




def VAE_loss(recon_x, x, mu, logvar, KL_weight = 1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # BCE = F.mse_loss(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KL_weight*KLD, BCE, KLD

def VAE_train(model, dataloader, criterion, optimizer, epochs = 5, KL_weight = 1.0, device = 'cpu'):

    for epoch in tqdm(range(epochs), desc='Epochs'):
        running_loss = 0.0
        running_KL = 0.0
        running_recon = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Training')
        for i, (images, _) in pbar:
        #for images, _ in dataloader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            recon_x = outputs[0]
            mu = outputs[1]
            logvar = outputs[2]
            loss, KL, RECON = criterion(recon_x=recon_x, x = images, mu = mu, logvar = logvar, KL_weight = KL_weight)

            loss.backward()
            optimizer.step()

            running_KL += KL.item() * images.size(0)
            running_recon += RECON.item() * images.size(0)
            running_loss += loss.item() * images.size(0)

            pbar.set_postfix({'Loss': running_loss / ((i+1) * dataloader.batch_size),
                              'KL': running_KL / ((i+1) * dataloader.batch_size),
                              'Recon': running_recon / ((i+1) * dataloader.batch_size)}, refresh=True)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_KL = running_KL / len(dataloader.dataset)
        epoch_recon = running_recon / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.5f}, KL: {epoch_KL:.5f}, Recon: {epoch_recon:.5f}")    


class GeometricFiguresDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=60000, image_size=(28, 28), random_color=True):
        self.num_samples = num_samples
        self.image_size = image_size
        self.random_color = random_color

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, label = self.generate_image()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.tensor(label)
        return image_tensor, label_tensor

    def generate_image(self):
        image = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 255  # White background

        shape = np.random.choice(['square', 'circle', 'triangle'])
        color = self.choose_color()
        size = np.random.randint(3, self.image_size[0] // 2)
        pos_x = np.random.randint(0, self.image_size[1] - size)
        pos_y = np.random.randint(0, self.image_size[0] - size)

        if shape == 'square':
            image = self.draw_square(image, x = pos_x, y = pos_y,  size=size, color=color)
        elif shape=='circle':
            image = self.draw_circle(image, x = pos_x, y = pos_y, radius=size//2, color=color)
        else:
            image = self.draw_triangle(image, x = pos_x, y = pos_y, size=size, color=color)

        #label is the shape, color, size and position of the shape
        shape_dict = {'square': 0, 'circle': 1, 'triangle': 2}
        label = np.array([shape_dict[shape],  size, pos_x, pos_y, color[0], color[1], color[2]])


        return image, label

    def choose_color(self):
        if self.random_color:
            return np.random.randint(0, 256, size=3)  # Random RGB color
        else:
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


    def draw_square(self, image, x, y, size, color):
        image[y:y+size, x:x+size] = color
        return image

    def draw_triangle(self, image, x, y, size, color):
        image[y:y+size, x:x+size] = color

        # Determine whether to remove upper or lower half of the square
        if np.random.rand() < 0.5:
            # Remove upper half of the square
            for i in range(size):
                for j in range(size):
                    if i > j:
                        image[y+i, x+j] = 255  # Background color
        else:
            # Remove lower half of the square
            for i in range(size):
                for j in range(size):
                    if i < j:
                        image[y+i, x+j] = 255  # Background color
        return image

    def draw_circle(self, image, x, y, radius, color):

        yy, xx = np.ogrid[-y:image.shape[0]-y, -x:image.shape[1]-x]
        mask = xx*xx + yy*yy <= radius*radius
        image[mask] = color
        return image


class GeometricFiguresDatasetBlackWhite(GeometricFiguresDataset):
    def __init__(self, num_samples=60000, image_size=(28, 28)):
        super().__init__(num_samples, image_size, random_color=False)

    def choose_color(self):
        return np.array([0, 0, 0])  # Black color

    def __getitem__(self, idx):
        img, label =  super().__getitem__(idx)
        img = img[0]  # Take only one channel
        img = img/255.0  # Normalize
        return img, label





def visualise_dataset(dataloader, model = None, device = 'cpu'):

    if dataloader == 'latent_sample':
        assert model is not None, "Model must be provided to sample from latent space"
        latent_dim = model.latent_dim
        random_sample = torch.randn(10, latent_dim).to(device)
        output = model.decode(random_sample)

        plt.figure(figsize=(10, 5))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.imshow(output[i].permute(1, 2, 0).cpu().detach().numpy())
            plt.xticks([])
            plt.yticks([])
            if i == 0 or i == 5:
                plt.ylabel('Sampled from latent space')
        
        plt.show()
        return None


    
    random_batch = next(iter(dataloader))
    random_batch = random_batch[0].to(device)
    output = model(random_batch)[0] if model is not None else None

    if not model:
        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(random_batch[i].permute(1, 2, 0).cpu().detach().numpy())
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('Input')
        plt.show()
        return None

    if model:

        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.imshow(random_batch[i].permute(1, 2, 0).cpu().detach().numpy())
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('Input')

            plt.subplot(2, 5, i+6)
            plt.imshow(output[i].permute(1, 2, 0).cpu().detach().numpy())
            plt.xticks([])
            plt.yticks([])
            #plt.ylabel('Output') if model is not None else plt.ylabel('Input (repeat)')
            if i == 0:
                plt.ylabel('Model')

        plt.show()
        return None
