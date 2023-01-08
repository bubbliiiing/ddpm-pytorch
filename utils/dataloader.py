import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class DiffusionDataset(Dataset):
    def __init__(self, annotation_lines, input_shape):
        super(DiffusionDataset, self).__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image   = Image.open(self.annotation_lines[index].split()[0])
        image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)
        
        image   = np.array(image, dtype=np.float32)
        image   = np.transpose(preprocess_input(image), (2, 0, 1))
        return image

def Diffusion_dataset_collate(batch):
    images = []
    for image in batch:
        images.append(image)
    images = torch.from_numpy(np.array(images, np.float32))
    return images
