from torch.utils.data import Dataset
from PIL import Image
import os

class CelebAHQ(Dataset):
    #根据data_path读取数据。所以我需要再调用之前就把客户端编号写入该变量
    def __init__(self, filter: str, data_path, remove_img_names=None, transform=None):
        self.data_path = data_path
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        #self.image_files = ['10000.jpg', '10001.jpg', '10002.jpg']
        self.transform = transform

        match filter:
            case "all":
                pass
            case "deletion":
                if remove_img_names is None:
                    raise ValueError('Deletion filter requires removal class to be specified.')
                    # 打印所有需要移除的图像文件路径
                    print(f"打印所有需要移除的图像文件路径: {remove_img_names}")

                    # 假设 remove_img_names 是文件名列表，生成每个文件的完整路径
                    remove_paths = [os.path.join(self.data_path, img_name) for img_name in remove_img_names]
                    print(f"假设 remove_img_names 是文件名列表，生成每个文件的完整路径: {remove_paths}")
                self.image_files = remove_img_names
            case "nondeletion":
                #加载不在remove img names中的图片
                if remove_img_names is None:
                    raise ValueError('Nondeletion filter requires removal class to be specified.')
                self.image_files = [f for f in os.listdir(data_path) if f not in remove_img_names]
            case _:
                raise ValueError('Invalid filter.') 
            
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_path, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image