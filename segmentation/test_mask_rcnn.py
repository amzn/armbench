import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from PIL import Image, ImageOps
import PIL


from coco_utils import get_coco, collate_fn
from engine import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Test a detector")
    parser.add_argument(
        "--resume-from",
        help="the checkpoint file to load the model from",
        required=True,
    )

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


def save_result(img, int_img, prediction):
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    output = prediction[0]
    masks, boxes = output["masks"], output["boxes"]

    detection_threshold = 0.8
    pred_scores = output["scores"].detach().cpu().numpy()
    pred_classes = [str(i) for i in output["labels"].cpu().numpy()]
    pred_bboxes = output["boxes"].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    pred_classes = pred_classes[: len(boxes)]

    int_img = np.array(int_img)
    int_img = np.transpose(int_img, [2, 0, 1])
    int_img = torch.tensor(int_img, dtype=torch.uint8)

    colors = np.random.randint(0, 255, size=(len(pred_bboxes), 3))
    colors = [tuple(color) for color in colors]
    result_with_boxes = draw_bounding_boxes(
        int_img,
        boxes=torch.tensor(boxes),
        width=4,
        colors=colors,
        labels=pred_classes,
    )

    final_masks = masks > 0.5
    final_masks = final_masks.squeeze(1)
    seg_result = draw_segmentation_masks(
        result_with_boxes, final_masks, colors=colors, alpha=0.8
    )

    seg_img = Image.fromarray(seg_result.mul(255).permute(1, 2, 0).byte().numpy())

    imgs = [img, seg_img]
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])

    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save("result.jpg")


def main():
    args = parse_args()
    dataset_path = args.dataset_path

    img_data_path = os.path.join(dataset_path, "mix-object-tote/images")
    val_ann_data_path = os.path.join(dataset_path, "mix-object-tote/test.json")

    coco_val_ds = get_coco(
        img_data_path, val_ann_data_path, mode="instances", train=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        coco_val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 3

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])

    evaluate(model, data_loader_val, device)

    # pick one image from the test set
    sample_idx = torch.randint(len(coco_val_ds), size=(1,)).item()

    int_img, img, _ = coco_val_ds[sample_idx]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    save_result(img, int_img, prediction)


if __name__ == "__main__":
    main()
