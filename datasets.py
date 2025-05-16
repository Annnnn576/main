import numpy as np
import torch, os
from torchvision import transforms
import copy
from torch.utils.data import Dataset
from PIL import Image

def get_data(args, phase):
    dataset = Data(args, phase)
    return dataset

def read_data(base_path, phase):
    images, label, label_obs = load_data(base_path, phase)

    # split_idx_train, split_idx_val = generate_split(len(images), 0.2, np.random.RandomState(1))
    if phase == 'train':
        # non_zero_indices = [i for i in range(label.shape[0]) if not np.all(label[i, :] == 0)]
        # images = images[non_zero_indices]
        # label_obs = label_obs[non_zero_indices]
        # label = label[non_zero_indices]

        split_idx_train, split_idx_val = generate_split(len(images), 0.2, np.random.RandomState(1))
        index = split_idx_train
        label = label[index]
        images = images[index]
        label_obs = label_obs[index]
    # else:
    #     index = split_idx_val
    #     label = label[index]
    #     images = images[index]
    #     label_obs = label_obs[index]

    return images, label, label_obs

class Data(torch.utils.data.Dataset):
    def __init__(self, args, phase):
        meta = get_metadata(args.datasets)
        base_path = meta['path_to_dataset']
        images, label, label_obs = read_data(base_path, phase)
        self.dir = meta['path_to_images']
        self.phase = phase
        self.train_data = images
        self.train_labels = label
        self.label_obs = label_obs
        self.num_classes = args.number_class
        self.count = 0
        # self.soft_labels = np.zeros((len(self.train_data), self.num_classes), dtype=np.float32)
        self.prediction = np.zeros((len(self.train_data), 5, self.num_classes), dtype=np.float32)
        self.prediction_aug = np.zeros((len(self.train_data), 5, self.num_classes), dtype=np.float32)
        self.pseudo_label = np.zeros((len(self.train_data), 5, self.num_classes), dtype=np.float32)
        self.pseudo_label_aug = np.zeros((len(self.train_data), 5, self.num_classes), dtype=np.float32)

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),  # 448
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        self.transform_augment = transforms.Compose([
            transforms.RandomResizedCrop((448, 448)),  # 448
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        self.pseudo_label = np.zeros((len(self.train_data), self.num_classes), dtype=np.float32)


    def label_update(self, results, aug):
        self.count += 1
        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 5
        self.prediction[:, idx] = results
        # self.pseudo_label = results
        # self.prediction[:, idx] = results
        label_combine = self.prediction.mean(axis=1)
        self.pseudo_label = label_combine
        if aug is True:
            self.prediction_aug[:, idx] = results
            label_combine_aug = self.prediction_aug.mean(axis=1)
            self.pseudo_label_aug = label_combine_aug

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # dir = "/home/liujiayi/idea/multi-label learning/VLPL-main/data/pascal/VOCdevkit/VOC2012/JPEGImages/"
        # meta = get_metadata(args.datasets)
        # dir = meta['path_to_images']
        images, label, label_obs, p_label = self.train_data[index], self.train_labels[index], self.label_obs[index, :], self.pseudo_label[index, :]
        p_label_aug = self.pseudo_label_aug[index]
        image_path = os.path.join(self.dir, images)
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(image)
        if self.transform_augment is not None:
            img2 = self.transform_augment(image)
        return img1, img2, label, label_obs, p_label, index


    def __len__(self):
        if self.phase == 'train':
            return len(self.label_obs)
        else:
            return len(self.label_obs)



def load_data(base_path, phase):
    label = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase))).astype(float)
    label_obs = np.load(os.path.join(base_path, 'formatted_{}_labels_obs.npy'.format(phase))).astype(float)
    images = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return images, label, label_obs

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are
    returned because this function is for splitting, not shuffling.
    '''

    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2

    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])

    return idx_1, idx_2



def get_metadata(dataset_name):
    if dataset_name == 'pascal':
        meta = {
            'num_classes': 20,
            'path_to_dataset': '/home/liujiayi/idea/multi-label learning/VLPL-main/data/pascal',
            'path_to_images': '/home/liujiayi/idea/multi-label learning/VLPL-main/data/pascal/VOCdevkit/VOC2012/JPEGImages'
        }
    elif dataset_name == 'coco':
        meta = {
            'num_classes': 80,
            'path_to_dataset': './cole_2021_multi_label/data/coco',
            'path_to_images': './data/coco'
        }
    elif dataset_name == 'nuswide':
        meta = {
            'num_classes': 81,
            'path_to_dataset': './cole_2021_multi_label/data/nuswide',
            'path_to_images': './data/nuswide/Flickr'
        }
    elif dataset_name == 'cub':
        meta = {
            'num_classes': 312,
            'path_to_dataset': './cole_2021_multi_label/data/cub',
            'path_to_images': './data/cub/CUB_200_2011/images'
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta


def get_transforms():
    '''
    Returns image transforms.
    '''

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    tx = {}
    tx['train'] = transforms.Compose([
        transforms.Resize((448, 448)), #448
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((448, 448)), #448
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['test'] = transforms.Compose([
        transforms.Resize((448, 448)), #448
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return tx