import os
import torchvision.io as io
from torch.utils.data import Dataset


class GANDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        super(GANDataset, self).__init__()
        self.root_folder = root_folder
        self.x_files = sorted(os.listdir(root_folder + '/x/'))
        self.y_files = sorted(os.listdir(root_folder + '/y/'))
        self.transform = transform

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        x_image = io.read_image(self.root_folder + '/x/' + self.x_files[idx])
        y_image = io.read_image(self.root_folder + '/y/' + self.y_files[idx])

        if self.transform:
            x_image = self.transform(x_image)
            y_image = self.transform(y_image)

        x_image = x_image / 127.5 - 1.0
        y_image = y_image / 127.5 - 1.0

        return x_image, y_image
