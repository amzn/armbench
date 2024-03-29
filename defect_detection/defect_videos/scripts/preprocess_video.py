import os
import time
import shutil
from collections import deque
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
import av; av.logging.set_level(av.logging.ERROR)
from tqdm import tqdm
import argparse


def uniform_sample(num_frames, num_samples):
    return np.round(np.linspace(0, num_frames - 1, num_samples)).astype(int)

def extract_frames_pyav(video_path, num_samples=None, cut_frames=0, max_frames=None):
    """
    extract num_samples frames from an av.container.input.InputContainter as evenly as possible
    """
    container = av.open(video_path)
    num_frames = container.streams.video[0].frames
    if max_frames is None:
        max_frames = num_frames - cut_frames
    max_frames = max(0, num_frames - cut_frames, max_frames)
    if num_samples is None:
        num_samples = max_frames
    frame_ids = deque(uniform_sample(max_frames, num_samples))
    imgs = []
    for idx, frame in enumerate(container.decode(video=0)):
        if not frame_ids:
            break
        while frame_ids and idx == frame_ids[0]:
            imgs.append(frame.to_ndarray(format='rgb24'))
            frame_ids.popleft()
    return imgs

def cv2_imshow(img, window_name="image"):
    cv2.imshow(window_name, img)
    key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return key

def crop_frames(imgs, pt1, pt2):
    return [img[pt1[1]:pt2[1], pt1[0]:pt2[0], :] for img in imgs]

def resize_frames(imgs, im_w=320, im_h=320, interpolation=cv2.INTER_CUBIC):
    return [cv2.resize(img, (im_w, im_h), interpolation=interpolation) for img in imgs]

def read_frames(input_folder):
    img_paths = [os.path.join(input_folder, file) for file in sorted(os.listdir(input_folder))]
    return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_paths]

def save_frames(output_folder, imgs, fmt="jpg"):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    for idx, img in enumerate(imgs):
        output_path = os.path.join(output_folder, f"{idx:04d}.{fmt}")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def processing_single_video(video_name, dataset, delete_video):
    video_path = os.path.join(dataset, 'videos', f"{video_name}.mp4")
    output_folder = os.path.join(dataset, 'cropped_frames', video_name)
    imgs = extract_frames_pyav(video_path, num_samples=None, cut_frames=0)
    imgs = crop_frames(imgs, pt1=(100, 80), pt2=(850, 650))
    imgs = resize_frames(imgs, im_w=320, im_h=320)
    save_frames(output_folder, imgs, fmt="jpg")
    if delete_video:
        os.remove(video_path)

def process_videos(file_list, dataset_path, delete_video):
    tic = time.time()
    with Pool(processes=8) as pool:
        f = partial(processing_single_video, dataset=dataset_path, delete_video=delete_video)
        pool.map(f, file_list)
    print(f"Processing {len(file_list)} videos takes {time.time() - tic}s")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", required=True)
    
    parser.add_argument("--delete_videos", action='store_true', default=False)

    args = parser.parse_args()


    output_folder = os.path.join(args.dataset_path, "cropped_frames")
    os.makedirs(output_folder, exist_ok=True)

    train_csv = os.path.join(args.dataset_path, "train.csv")
    test_csv = os.path.join(args.dataset_path, "test.csv")

    
    df_train = pd.read_csv(train_csv, header=None)
    df_test = pd.read_csv(test_csv, header=None)
    
    video_files_train = df_train[0].to_list()
    video_files_test = df_test[0].to_list()

    process_videos(video_files_train, args.dataset_path, args.delete_videos)
    process_videos(video_files_test, args.dataset_path, args.delete_videos)
