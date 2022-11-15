import cv2
import imgaug.augmenters as iaa
from torchvision import transforms
from torch.utils.data import Dataset
from utils import pad_to_square


class FoodDataset(Dataset):
    def __init__(self, img_paths, label_to_idx, img_shape, apply_aug, **kwargs):
        self.img_paths = img_paths
        self.labels = [label_to_idx[p.split('/')[-2]] for p in img_paths]
        
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        
        self.img_shape = tuple(img_shape)
        self.apply_aug = apply_aug
        
        self.augmentor = iaa.RandAugment(n=(0, 5), m=(0, 9))
        self.pre_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        org_img = pad_to_square(cv2.imread(self.img_paths[idx]))
        org_img = cv2.resize(org_img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        
        label_idx = self.labels[idx]
        label_name = self.idx_to_label[label_idx]
        
        # Apply augmentation
        if self.apply_aug:
            aug_img = self.augmentor.augment_image(org_img)
        else:
            aug_img = org_img
        
        input_tensor = self.pre_transform(aug_img)
        
        return {'org_img': org_img, 'aug_img': aug_img, 'input': input_tensor, 'label': label_idx, 'label_name': label_name}