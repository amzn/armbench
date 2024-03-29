import argparse
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from coco_utils import get_coco, collate_fn
from engine import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")

    parser.add_argument("-d", "--dataset_path", default="", required=True)

    args = parser.parse_args()
    return args


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def main():
    args = parse_args()
    dataset_path = args.dataset_path

    img_data_path = os.path.join(dataset_path, "mix-object-tote/images")
    train_ann_data_path = os.path.join(dataset_path, "mix-object-tote/train.json")
    val_ann_data_path = os.path.join(dataset_path, "mix-object-tote/val.json")

    coco_train_ds = get_coco(
        img_data_path, train_ann_data_path, mode="instances", train=True
    )
    coco_val_ds = get_coco(
        img_data_path, val_ann_data_path, mode="instances", train=False
    )

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        coco_train_ds, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        coco_val_ds, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 3

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    from torch.optim.lr_scheduler import StepLR

    num_epochs = 10
    s_epoch = 0
    print(args.resume_from)
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        s_epoch = checkpoint["epoch"] + 1

    for epoch in range(s_epoch, num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "latest.pt",
        )


if __name__ == "__main__":
    main()
