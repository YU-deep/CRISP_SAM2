import math
import numpy as np
import torch
from monai import data, transforms
import itertools
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset
import os
import ast
from scipy import sparse
import random
import json
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    AddChanneld,
    CropForegroundd,
    SpatialPadd,
    Resized,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"{e}")
    return None


class UnionDataset(Dataset):
    def __init__(self, concat_dataset, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = torch.cumsum(torch.tensor([0] + self.lengths), dim=0)
        self.concat_dataset = concat_dataset

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]


class UniversalDataset(Dataset):
    def __init__(self, data, transform, test_mode, organ_list):
        self.data = data
        self.transform = transform
        # one pos point is base set
        self.num_positive_extra_max = 10
        self.num_negative_extra_max = 10
        self.test_mode = test_mode
        self.bbox_shift = 10 if test_mode else 0
        print(organ_list)
        organ_list.remove('background')
        self.target_list = organ_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get path
        item_dict = self.data[idx]
        ct_path, gt_path = item_dict['image'], item_dict['label']
        gt_shape = ast.literal_eval(gt_path.split('.')[-2].split('_')[-1])

        # load data
        ct_array = np.load(ct_path)[0]
        allmatrix_sp = sparse.load_npz(gt_path)
        gt_array = allmatrix_sp.toarray().reshape(gt_shape)

        # transform
        if self.test_mode:
            item_ori = {
                'image': ct_array,
                'label': gt_array,
            }
        else:
            item_ori = {
                'image': ct_array,
                'label': gt_array,
            }
        if self.transform is not None:
            item = self.transform(item_ori)

        if type(item) == list:
            assert len(item) == 1
            item = item[0]

        assert type(item) != list
        item['organ_name_list'] = self.target_list
        item['post_label'] = item['label']
        post_item = self.std_keys(item)
        return post_item

    def std_keys(self, post_item):
        keys_to_remain = ['image', 'post_label', 'organ_name_list']
        keys_to_remove = post_item.keys() - keys_to_remain
        for key in keys_to_remove:
            del post_item[key]
        return post_item


def query_descriptions(input_name, data, min_count=4, max_count=6):
    input_name = input_name.strip().lower()
    matching_keys = [k for k in data.keys() if k.lower() == input_name]

    if not matching_keys:
        return []

    descriptions = data[matching_keys[0]]
    count = len(descriptions)

    select_count = min(max_count, count)
    select_count = max(min_count, select_count)

    selected = descriptions[:select_count]

    return selected



class BatchedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, shuffle, batch_size, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.batch_size = batch_size

    def __iter__(self):
        print('run BatchedDistributedSampler iter')
        indices = list(range(len(self.dataset)))
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        indices = [indices[i:i + l] for i, l in zip(self.dataset.offsets[:-1], self.dataset.lengths)]

        if self.shuffle:
            for idx, subset_indices in enumerate(indices):
                random.shuffle(indices[idx])

        # drop subset last
        for idx, subset_indices in enumerate(indices):
            r = len(subset_indices) % self.batch_size
            if r > 0:
                indices[idx] = indices[idx][:-r]
        indices = list(itertools.chain(*indices))
        indices = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(indices)

        batch_num = len(indices)
        replicas_size = batch_num // self.num_replicas
        start = self.rank * replicas_size
        end = start + replicas_size if self.rank != self.num_replicas - 1 else batch_num
        batched_indices = list(itertools.chain(*(indices[start:end])))
        ##
        indices = list(itertools.chain(*indices))
        self.total_size = len(indices)
        self.num_samples = self.total_size // self.num_replicas
        ##
        return iter(batched_indices)


def collate_fn(batch, dictionary_path):
    term_dictionary = read_json_file(dictionary_path)
    images = []
    organ_name_list = None
    post_labels = []
    organ_description_list = []
    for sample in batch:
        images.append(sample['image'])
        assert organ_name_list is None or organ_name_list == sample['organ_name_list']
        organ_name_list = sample['organ_name_list']
        for organ in organ_name_list:
            organ_description_list.append(query_descriptions(organ, term_dictionary, 4, 6))
        post_labels.append(sample['post_label'])
    return {
        'image': torch.stack(images, dim=0),
        'organ_name_list': organ_name_list,
        'post_label': torch.stack(post_labels, dim=0),
        'text': organ_description_list
    }


class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d


class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d


def build_concat_dataset(root_path, dataset_codes, transform):
    concat_dataset = []
    CombinationDataset_len = 0
    for dataset_code in dataset_codes:
        datalist_json = os.path.join(root_path, dataset_code, f'{dataset_code}.json')
        with open(datalist_json, 'r') as f:
            dataset_dict = json.load(f)
        datalist = dataset_dict['train']
        universal_ds = UniversalDataset(data=datalist, transform=transform, test_mode=False,
                                        organ_list=list(dataset_dict['labels'].values()))
        concat_dataset.append(universal_ds)
        CombinationDataset_len += len(universal_ds)
    print(f'CombinationDataset loaded, dataset size: {CombinationDataset_len}')
    return UnionDataset(ConcatDataset(concat_dataset), concat_dataset)


def get_loader(args):
    if args.mode == 'train':
        train_transform = Compose(
            [
                AddChanneld(keys=["image"]),
                DimTranspose(keys=["image", "label"]),
                MinMaxNormalization(),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=args.spatial_size,
                            mode='constant'),
                transforms.OneOf(transforms=[
                    Resized(keys=["image", "label"], spatial_size=args.spatial_size),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=args.spatial_size,
                        pos=2,
                        neg=1,
                        num_samples=1,
                        image_key="image",
                        image_threshold=0,
                    ),
                ],
                    weights=[1, 1]
                ),
                RandFlipd(keys=["image", "label"], prob=args.rand_flipped_prob, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=args.rand_flipped_prob, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=args.rand_flipped_prob, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=args.rand_scale_intensityd_prob),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=args.rand_shift_intensityd_prob),
                Resized(keys=["image", "label"], spatial_size=args.spatial_size),
                ToTensord(keys=["image", "label"]),
            ]
        )

        print(f'----- train on combination dataset -----')
        combination_train_ds = build_concat_dataset(root_path=args.data_dir, dataset_codes=args.dataset_codes,
                                                    transform=train_transform)
        train_sampler = BatchedDistributedSampler(combination_train_ds, shuffle=True,
                                                  batch_size=args.batch_size) if args.dist else None
        train_loader = DataLoader(
            combination_train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return train_loader
    elif args.mode == 'test':
        test_transform = Compose(
            [
                AddChanneld(keys=["image"]),
                DimTranspose(keys=["image", "label"]),
                MinMaxNormalization(),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=args.spatial_size,
                            mode='constant'),
                Resized(keys=["image", "label"], spatial_size=args.spatial_size),
                ToTensord(keys=["image", "label"]),
            ]
        )

        print(f'----- test on combination dataset -----')
        combination_test_ds = build_concat_dataset(root_path=args.data_dir, dataset_codes=args.dataset_codes,
                                                   transform=test_transform)
        test_sampler = BatchedDistributedSampler(combination_test_ds, shuffle=False,
                                                 batch_size=args.batch_size) if args.dist else None
        test_loader = DataLoader(
            combination_test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return test_loader
    else:
        raise ValueError("mode should be either 'train' or 'test'")