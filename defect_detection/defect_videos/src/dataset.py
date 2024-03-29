import os
import time
import argparse
import json
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.transforms import (
    RandomResizedCrop,
    Normalize,
    Permute,
    UniformTemporalSubsample
)
from torchvision.transforms import (
    Lambda,
    RandomGrayscale,
    ColorJitter,
    Compose,
    RandomHorizontalFlip,
)
import matplotlib.pyplot as plt
from tqdm import tqdm


class TemporalSubsample(torch.nn.Module):

    def __init__(self, num_samples: Union[int, List[int]], temporal_dim: int = 1, mode: str = "uniform"):
        super().__init__()
        if isinstance(num_samples, int):
            self.num_samples = [num_samples]
        else:
            self.num_samples = num_samples
        self.mode = mode
        self.temporal_dim = temporal_dim
        assert mode in ["uniform", "random"], "mode must be either 'uniform' or 'random'"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[self.temporal_dim]
        ret = []
        for t in self.num_samples:
            if self.mode == 'uniform':
                indices = torch.linspace(0, T - 1, t)
                indices = torch.clamp(indices, 0, T - 1).long()
                samples =  torch.index_select(x, self.temporal_dim, indices)
                ret.append(samples)
            elif self.mode == 'random':
                if T < t:
                    indices = torch.linspace(0, T - 1, t)
                    indices = torch.clamp(indices, 0, T - 1).long()
                else:
                    indices = torch.randperm(T)[:t].sort().values
                samples = torch.index_select(x, self.temporal_dim, indices)
                ret.append(samples)
            else:
                raise NotImplementedError("mode must be either 'uniform' or 'random'")
        return ret


def make_transforms(args, train=True):
    
        
        
    if train:
        transform=Compose(
            [
                Permute((1, 0, 2, 3)),  # (T, C, H, W) -> (C, T, H, W)
                # RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.4),  # slow
                # RandomGrayscale(p=0.2),  # slow
                # Permute((1, 0, 2, 3)),  # (C, T, H, W)
                RandomHorizontalFlip(p=0.5),
                Normalize(0, 255.0),
                #Lambda(tmp_func),
                Normalize(args.video_means, args.video_stds),
                RandomResizedCrop(args.train_spatial_size, args.train_spatial_size, (0.9, 1), (1, 1)),
            ]
        )
    else:
        transform=Compose(
            [
                Permute((1, 0, 2, 3)),  # (T, C, H, W) -> (C, T, H, W)
                #Lambda(tmp_func),
                Normalize(0, 255.0),
                Normalize(args.video_means, args.video_stds),
                RandomResizedCrop(args.test_spatial_size, args.test_spatial_size, (1, 1), (1, 1)),
            ]
        )
    return transform


class FrameVideoDataset(Dataset):

    def __init__(self,
            video_label_df: pd.DataFrame,
            dataset_path: str,
            transforms: nn.Module = None,
            clip_stride: int = 0,
            num_frames_per_clip: int = 0,
            num_samples_per_clip: int = 64,
            temporal_size: int = 64
        ):
        super().__init__()
        self.df = video_label_df
        self.video_files_list = self.df[0].to_list()
        self.dataset_path = dataset_path
        self.annotations_folder = os.path.join(dataset_path, 'annotations')

        self.label_to_idx = {'nominal': [0,0], 'open': [1, 0], 'deconstruction': [0,1]}


        self.transforms = transforms

        MAX_SIZE = 100000000
        self.clip_stride = clip_stride if clip_stride else MAX_SIZE
        self.num_frames_per_clip = num_frames_per_clip if num_frames_per_clip else MAX_SIZE
        self.num_samples_per_clip = num_samples_per_clip
        self.temporal_size = temporal_size
        self.temporal_subsample = TemporalSubsample(temporal_size, temporal_dim=1)

    def __len__(self):
        return self.df.shape[0]

    def sample_clip_frame_indices(self, total_frames):
        sampled_indices = []
        for i in range(0, total_frames, self.clip_stride):
            end_frame = min(total_frames, i + self.num_frames_per_clip) - 1
            indices = torch.round(torch.linspace(i, end_frame, self.num_samples_per_clip)).to(torch.int64)
            sampled_indices.append(indices)
        return sampled_indices

    def __getitem__(self, idx) -> Dict:
        entry = str(self.video_files_list[idx])

        annotation_json = os.path.join(self.annotations_folder, entry + '.json')

        if os.path.exists(annotation_json):
            # read annotation file
            jf = open(annotation_json)
            data = json.load(jf)

            # get the id
            id = data['id']
            video_name = id + '.mp4'
            frame_folder = os.path.join(self.dataset_path, 'cropped_frames', id)
            frame_files = sorted(os.listdir(frame_folder))
            item = dict()
            item['video_name'] = video_name
            item['frame_folder'] = frame_folder

            label_idx = self.label_to_idx[data['label']]
            
            item['label'] = np.array(label_idx, dtype=np.int64)

            if len(frame_files) == 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
                print(item['video_name'])
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")

            sampled_indices = self.sample_clip_frame_indices(len(frame_files))
            unique_sampled_indices = torch.unique(torch.cat(sampled_indices))
            index_mapping = torch.zeros(len(frame_files), dtype=torch.int64)
            index_mapping[unique_sampled_indices] = torch.arange(len(unique_sampled_indices))

            frames = []
            for idx in unique_sampled_indices:
                frames.append(read_image(os.path.join(frame_folder, frame_files[idx])))
            frames = torch.stack(frames).float()

            if self.transforms:
                frames = self.transforms(frames)

            item["clips"] = []
            for indices in sampled_indices:
                sampled_frames = frames[:, index_mapping[indices], ...]
                sampled_frames = self.temporal_subsample(sampled_frames)
                item['clips'].append(sampled_frames)
            return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--train_spatial_size", default=240, type=int)
    parser.add_argument("--clip_stride", default=0, type=int, help="stride of each clip")
    parser.add_argument("--num_frames_per_clip", default=0, type=int, help="0 means using all the frames")
    parser.add_argument("--num_samples_per_clip", default=64, type=int)
    parser.add_argument("--temporal_size", default=[16, 64], type=int, nargs="+")
    args = parser.parse_args()

    video_label_file = "dataset/benchmark_damaged/summary.csv"
    video_label_df = pd.read_csv(video_label_file)
    
    dataset = FrameVideoDataset(
        video_label_df,
        transforms = make_transforms(args, train=True),
        clip_stride=args.clip_stride,
        num_frames_per_clip=args.num_frames_per_clip,
        num_samples_per_clip=args.num_samples_per_clip,
        temporal_size=args.temporal_size
    )
    print(f"{len(dataset)} = ")

    tic = time.time()
    item = dataset[0]
    print(time.time() - tic)

    print(f"item['video_name'] = {item['video_name'] }")
    print(f"item['frame_folder'] ={item['frame_folder']} ")
    print(f"item['label'] = {item['label']} ")
    print(f"len(item['clips'])  = {len(item['clips']) } ")

    for i, clip in enumerate(item["clips"]):
        for j, t in enumerate(args.temporal_size):
            print(f'i= {i}, t ={t} , clip[j].shape ={clip[j].shape} ')

    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, drop_last=False)
    for batch in tqdm(dataloader):
        pass
        # print(f"{batch['video_name'] = }")
        # print(f"{batch['frame_folder'] = }")
        # print(f"{batch['label'] = }")
        # print(f"{batch['label'].dtype = }")
        # print(f'{len(batch["clips"]) = }')
        # print(f'{batch["clips"][0][0].shape = }')
        # print(f'{batch["clips"][0][1].shape = }')
