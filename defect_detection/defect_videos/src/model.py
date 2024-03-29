import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorchvideo.models import slowfast, r2plus1d, vision_transformers


class VideoClassificationLightningModule(pl.LightningModule):

    def __init__(self, args):
        """
        This LightningModule implementation constructs a PyTorchVideo ResNet,
        defines the train and val loss to be trained with (cross_entropy), and
        configures the optimizer.
        """
        super().__init__()
        self.args = args

        self.train_overall_accuracy = torchmetrics.Accuracy(multiclass=True, subset_accuracy=True, num_classes=2)
        #self.train_overall_accuracy = torchmetrics.Accuracy(task="multiclass", subset_accuracy=True, num_classes=2)
        self.train_open_accuracy = torchmetrics.Accuracy()
        self.train_deconstruction_accuracy = torchmetrics.Accuracy()
        self.val_overall_accuracy = torchmetrics.Accuracy(multiclass=True, subset_accuracy=True, num_classes=2)
        #self.val_overall_accuracy = torchmetrics.Accuracy(task="multiclass", subset_accuracy=True, num_classes=2)
        self.val_open_accuracy = torchmetrics.Accuracy()
        self.val_deconstruction_accuracy = torchmetrics.Accuracy()

        self.key = "clips"
        self.criterion = nn.BCELoss(reduction='mean')

        if args.model == "SlowFast":
            self.model = slowfast.create_slowfast(
                model_num_class=174,
                # model_num_class=400,
                head_pool=nn.AdaptiveAvgPool3d,
                head_pool_kernel_sizes=((args.temporal_size[0], 7, 7),  # 7 = 224 / 32
                                        (args.temporal_size[1], 7, 7)),
                model_depth=args.model_depth,
                head_activation=None,
                head_output_with_global_average=True
            )
            pretrained_ckpt = torch.load("pretrained_ckpt/SLOWFAST_8x8_R50_ssv2.pyth")
            # pretrained_ckpt = torch.load("pretrained_ckpt/SLOWFAST_8x8_R50_kinetics400.pyth")
            self.model.load_state_dict(pretrained_ckpt["model_state"], strict=True)
            self.model.blocks[6].proj = nn.Linear(in_features=2304, out_features=args.num_classes, bias=True)
        elif args.model == "R(2+1)D":
            self.model = r2plus1d.create_r2plus1d( 
                model_num_class=400,
                head_pool=nn.AdaptiveAvgPool3d,
                head_pool_kernel_size=(None, 1, 1),
                # head_pool_kernel_size=(args.temporal_size[0] // 4, 7, 7),
                model_depth=args.model_depth,
                head_activation=None
            )
            pretrained_ckpt = torch.load("pretrained_ckpt/R2PLUS1D_16x4_R50.pyth")
            self.model.load_state_dict(pretrained_ckpt["model_state"], strict=True)
            self.model.blocks[5].proj = nn.Linear(in_features=2048, out_features=args.num_classes, bias=True)
        elif args.model == "MViT_B":
            self.model = vision_transformers.create_multiscale_vision_transformers( 
                spatial_size=args.spatial_size,
                temporal_size=args.temporal_size[0],
                embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
                atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
                pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
                pool_kv_stride_adaptive=[1, 8, 8],
                pool_kvq_kernel=[3, 3, 3],
                head_num_classes=400,
                head_activation=None
            )
            pretrained_ckpt = torch.load("pretrained_ckpt/MVIT_B_32x3.pyth")
            self.model.load_state_dict(pretrained_ckpt["model_state"], strict=True)
            self.model.head.proj = nn.Linear(in_features=768, out_features=args.num_classes, bias=True)
        # elif args.model == "MViTv2_B":
        #     cfg_file = "SlowFast/configs/Kinetics/MVITv2_B_32x3.yaml"
        #     # cfg_file = "SlowFast/configs/Kinetics/MVITv2_S_16x4.yaml"
        #     self.args.opts = None
        #     cfg = load_config(self.args, cfg_file)

        #     name = cfg.MODEL.MODEL_NAME
        #     self.model = MODEL_REGISTRY.get(name)(cfg)
        #     pretrained_ckpt = torch.load("pretrained_ckpt/MViTv2_B_32x3_k400.pyth")
        #     # pretrained_ckpt = torch.load("pretrained_ckpt/MViTv2_S_16x4_k400.pyth")
        #     self.model.load_state_dict(pretrained_ckpt['model_state'], strict=True)
        #     self.model.head.projection = nn.Linear(in_features=768, out_features=args.num_classes, bias=True)
        #     self.model.head.act = nn.Identity()
        else:
            raise NotImplementedError("args.model should be one of ['SlowFast', 'R(2+1)D', 'MViT_B']")

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        pass
        # epoch = self.trainer.current_epoch
        # if self.trainer.use_ddp:
        #     self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X = batch[self.key][0]
        if self.args.model not in ["SlowFast", "MViTv2_B"]:
            X = X[0]
        y_hat = torch.sigmoid(self.model(X))
        loss = self.criterion(y_hat, batch['label'].to(torch.float32))
        overall_acc = self.train_overall_accuracy(y_hat, batch["label"])
        open_acc = self.train_open_accuracy(y_hat[:, 0], batch["label"][:, 0])
        deconstruction_acc = self.train_deconstruction_accuracy(y_hat[:, 1], batch["label"][:, 1])
        self.log("train_loss", loss)
        self.log("train_overall_acc", overall_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_open_acc", open_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_deconstruction_acc", deconstruction_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch[self.key][0]
        if self.args.model not in ["SlowFast", "MViTv2_B"]:
            X = X[0]
        y_hat = torch.sigmoid(self.model(X))
        loss = self.criterion(y_hat, batch['label'].to(torch.float32))
        overall_acc = self.val_overall_accuracy(y_hat, batch["label"])
        open_acc = self.val_open_accuracy(y_hat[:, 0], batch["label"][:, 0])
        deconstruction_acc = self.val_deconstruction_accuracy(y_hat[:, 1], batch["label"][:, 1])
        self.log("val_loss", loss)
        self.log("val_overall_acc", overall_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_open_acc", open_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_deconstruction_acc", deconstruction_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # https://arxiv.org/pdf/2205.09113.pdf
        optimizer = torch.optim.AdamW( 
            self.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.max_epochs,
            eta_min=self.args.lr / 20,
            last_epoch=-1
        )
        return [optimizer], [scheduler]
