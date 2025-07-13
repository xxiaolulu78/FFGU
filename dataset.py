# dataset.py
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import os
import torch
from PIL import Image, UnidentifiedImageError


class VGGFaceDataset(Dataset):
    def __init__(self, root_dir, client_id, transform=None):
        self.root_dir = root_dir
        self.client_id = f"{client_id:03d}"
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if
                            img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Unable to open image {img_path}. Error: {e}")
            return None, self.client_id
        if self.transform:
            image = self.transform(image)
        return image, self.client_id


def get_client_dataloader(client_id,
                          root='./Deep3DFaceRecon/checkpoints/model_name/casia',
                          batch_size=8,
                          num_samples=40):
   
    client_folder = os.path.join(root, f"{client_id:03d}") 
    print(f'dataset.py中的cid :{client_id:03d}')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),  
    ])

    client_dataset = VGGFaceDataset(client_folder, client_id, transform=transform)
    indices = torch.randperm(len(client_dataset)).tolist()[:num_samples]
    sampler = SubsetRandomSampler(indices)
    dataloader = DataLoader(client_dataset, batch_size=batch_size, sampler=sampler)

    return dataloader, len(indices)

def get_client_dataloader_init(client_id,
                          root='./Deep3DFaceRecon/checkpoints/model_name/casia',
                          batch_size=4):
  
    client_folder = os.path.join(root, f"{client_id:03d}")  
    print(f'dataset cid :{client_id:03d}')
 
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),  
    ])

    client_dataset = VGGFaceDataset(client_folder, client_id, transform=transform)
    dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

    return dataloader, len(client_dataset)

