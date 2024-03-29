import os
import json
import argparse
import logging

import sys
sys.path.append('..')
sys.path.append('.')

from os import path
sys.path.append('path.dirname(path.dirname(path.abspath(__file__)))')

import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from .model import VideoClassificationLightningModule
from .dataset import FrameVideoDataset, make_transforms

def test(args):
        trainer: pl.Trainer = pl.Trainer.from_argparse_args(args)
        output_folder = os.path.join(args.dataset_path, "output")


        print("loading from checkpoint:", args.ckpt)
        model = VideoClassificationLightningModule.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)
        val_df = pd.read_csv(args.val_video_label_file, header=None)
        val_set = FrameVideoDataset(
            val_df,
            transforms=make_transforms(args, train=False),
            num_samples_per_clip=args.num_samples_per_clip,
            temporal_size=args.temporal_size,
            dataset_path=args.dataset_path
        )
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)
        

        trainer.test(model, dataloaders=val_loader)


        # predictions = trainer.predict(model, val_loader)
        # pred_list = list()
        # for preds in predictions:
        #     for p in preds:
        #         pred_list.append(p.cpu().detach().numpy().tolist())

        # result = {'predictions': pred_list, 
        #           'targets': val_loader.targets}
                    

        # results_location = os.path.join(output_folder, 'output.txt')

        # with open(results_location, 'w', encoding='UTF8', newline='') as f:
        #     json.dump(result, f)
        






def train(args):
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args)
    if args.ckpt is None:
        model = VideoClassificationLightningModule(args)
    else:
        print("loading from checkpoint:", args.ckpt)
        model = VideoClassificationLightningModule.load_from_checkpoint(checkpoint_path=args.ckpt, args=args)

    train_df = pd.read_csv(args.train_video_label_file, header=None)
    train_set = FrameVideoDataset(
        train_df,
        transforms=make_transforms(args, train=True),
        num_samples_per_clip=args.num_samples_per_clip,
        temporal_size=args.temporal_size,
        dataset_path=args.dataset_path
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    print(f"{len(train_set)} = ")

    val_df = pd.read_csv(args.val_video_label_file, header=None)
    val_set = FrameVideoDataset(
        val_df,
        transforms=make_transforms(args, train=False),
        num_samples_per_clip=args.num_samples_per_clip,
        temporal_size=args.temporal_size,
        dataset_path=args.dataset_path
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)
    print(f"{len(val_set)} = ")

    # trainer.test(model, dataloaders=val_loader)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt)
    trainer.test(model, dataloaders=val_loader)


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("damage_detection")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


def main():
    seed_everything(777)
    setup_logger()

    parser = argparse.ArgumentParser()

    # Model parameters.
    parser.add_argument("--model", default="SlowFast", type=str)
    parser.add_argument("--model_depth", default=50, type=int)
    parser.add_argument("--lr", "--learning-rate", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--ckpt", default=None, type=str)

    parser.add_argument("-m", "--mode", default="train",
                        help="Set model to 'train' or 'test'", required=False)


    # Data parameters.
    parser.add_argument("-d", "--dataset_path", required=True)

    parser.add_argument("--train_video_label_file", type=str, required=True)
    parser.add_argument("--val_video_label_file", type=str, required=True)

    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_samples_per_clip", default=64, type=int)
    parser.add_argument("--temporal_size", default=[16, 64], type=int, nargs="+")

    parser.add_argument("--train_spatial_size", default=224, type=int)
    parser.add_argument("--test_spatial_size", default=320, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    
    checkpoint_callback = ModelCheckpoint( 
        save_top_k=1,
        save_last=True,
        monitor="val_overall_acc",
        mode="max",
        every_n_epochs=5,
        filename="{epoch:02d}-{val_overall_acc:.2f}-{val_open_acc:.2f}-{val_deconstruction_acc:.2f}"
    )
    lr_monitor = LearningRateMonitor()

    # Trainer parameters.
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        accelerator="gpu",
        device=1,
        callbacks=[lr_monitor, checkpoint_callback],
        check_val_every_n_epoch=1
    )

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()
    args.spatial_size = args.train_spatial_size
    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
