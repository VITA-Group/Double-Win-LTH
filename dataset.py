import os 
import torch 
import shutil
import numpy as np 
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader, Subset, Dataset

# standard dataloader
def cifar10_dataloaders(batch_size=128, data_dir = 'datasets/cifar10', data_rate=1, add_normalize=True):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616])

    if add_normalize:
        print('with normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print('without normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    train_image_number = int(45000*data_rate)

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(train_image_number)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar10_dataloaders_generate_adv(batch_size=128, data_dir = 'datasets/cifar10', mark_normalize = False):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616])

    if mark_normalize:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(10000)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

    return train_loader



def cifar10_dataloaders_2(batch_size=128, data_dir = 'datasets/cifar10', data_rate=1, add_normalize=True):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616])

    if add_normalize:
        print('with normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print('without normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    train_image_number = int(45000*data_rate)

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(train_image_number)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar10_dataloaders_3(batch_size=128, data_dir = 'datasets/cifar10', data_rate=1, add_normalize=True):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616])

    if add_normalize:
        print('with normalize')
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        print('without normalize')
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_image_number = int(45000*data_rate)

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(train_image_number)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader




def cifar100_dataloaders(batch_size=128, data_dir = 'datasets/cifar100', data_rate=1, add_normalize=True):

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                    std=[0.2675, 0.2565, 0.2761])
    if add_normalize:
        print('with normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print('without normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    train_image_number = int(45000*data_rate)

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(train_image_number)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def svhn_dataloaders(batch_size=128, data_dir = 'datasets/svhn', data_rate=1, add_normalize=True):

    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], 
                                    std=[0.1201, 0.1231, 0.1052])

    if add_normalize:
        print('with normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print('without normalize')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    train_image_number = int(68257*data_rate)


    train_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True),list(range(train_image_number)))
    val_set = Subset(SVHN(data_dir, split='train', transform=test_transform, download=True),list(range(68257,73257)))
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)
            
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# CIFAR-C Dataset
class CIFAR_C(Dataset):

    def __init__(self, root, transform=None, severity=5, attack_type=''):

        dataPath = os.path.join(root, '{}.npy'.format(attack_type))
        labelPath = os.path.join(root, 'labels.npy')

        self.data = np.load(dataPath)[(severity - 1) * 10000: severity * 10000]
        self.label = np.load(labelPath).astype(np.long)[(severity - 1) * 10000: severity * 10000]
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]

def cifar10_c_dataloaders(batch_size, data_dir, severity, attack_type, add_normalize=True):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616])

    if add_normalize:
        print('with normalize')
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print('without normalize')
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    test_set = CIFAR_C(data_dir, transform=test_transform, severity=severity, attack_type=attack_type)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return test_loader




# train data without augmentation
def pure_cifar10_dataloaders(batch_size=128, data_dir = 'datasets/cifar10', add_normalize=True):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2470, 0.2435, 0.2616])

    if add_normalize:
        print('with normalize')
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    else:
        print('without normalize')
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])


    train_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader





class VisDA17(Dataset):

    def __init__(self, txt_file, root_dir, transform=transforms.ToTensor(), label_one_hot=False, portion=1.0):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.lines = open(txt_file, 'r').readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.label_one_hot = label_one_hot
        self.portion = portion
        self.number_classes = 12
        assert portion != 0
        if self.portion > 0:
            self.lines = self.lines[:round(self.portion * len(self.lines))]
        else:
            self.lines = self.lines[round(self.portion * len(self.lines)):]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = str.split(self.lines[idx])
        path_img = os.path.join(self.root_dir, line[0])
        image = Image.open(path_img)
        image = image.convert('RGB')
        if self.label_one_hot:
            label = np.zeros(12, np.float32)
            label[np.asarray(line[1], dtype=np.int)] = 1
        else:
            label = np.asarray(line[1], dtype=np.int)
        label = torch.from_numpy(label)
        if self.transform:
            image = self.transform(image)
        return image, label

def visda_loader(batch_size, data_dir, add_normalize=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if add_normalize:
        print('with normalize')
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        print('without normalize')
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        val_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])        

    train_dataset = VisDA17(txt_file=os.path.join(data_dir, "train/image_list.txt"), 
                            root_dir=os.path.join(data_dir, "train"), transform=train_trans)
    val_dataset = VisDA17(txt_file=os.path.join(data_dir, "validation/image_list.txt"), 
                            root_dir=os.path.join(data_dir, "validation"), transform=val_trans)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, val_loader





class Advdataset(Dataset):

    def __init__(self, root, transform=None):

        alldata = torch.load(root)
        self.data = alldata['data']
        self.label = alldata['label']
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]

