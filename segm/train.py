#!/usr/bin/env python3

# supervised

import sys
import os
from pathlib import Path
import yaml
import torch
import click
from types import SimpleNamespace

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import suppress
from timm.utils import NativeScaler

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config
from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params
import segm.engine as engine
from segm.engine import evaluate
import segm.utils.logging_config as log_cfg
from segm.utils.logging_config import plot_eval_metrics_from_csv


engine.LOSS_HISTORY = log_cfg.init_history()

IGNORE_LABEL = 255


@click.command(help="")
@click.option("--log-dir", type=str)
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int)
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=1, type=int)
@click.option("--amp/--no-amp", default=False)
@click.option("--resume/--no-resume", default=True)
def main(
    log_dir, dataset, im_size, crop_size, window_size, window_stride,
    backbone, decoder, optimizer, scheduler, weight_decay,
    dropout, drop_path, batch_size, epochs, learning_rate,
    normalization, eval_freq, amp, resume
):

    # --------------------
    # Distributed
    # --------------------
    ptu.set_gpu_mode(True)
    distributed.init_process()

    # --------------------
    # Config
    # --------------------
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    decoder_cfg = cfg["decoder"]["mask_transformer"] if "mask_transformer" in decoder else cfg["decoder"][decoder]

    im_size = im_size or dataset_cfg["im_size"]
    crop_size = crop_size or dataset_cfg.get("crop_size", im_size)
    window_size = window_size or dataset_cfg.get("window_size", im_size)
    window_stride = window_stride or dataset_cfg.get("window_stride", im_size)

    model_cfg.update({
        "image_size": (crop_size, crop_size),
        "backbone": backbone,
        "dropout": dropout,
        "drop_path_rate": drop_path,
    })

    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    world_batch_size = batch_size or dataset_cfg["batch_size"]
    num_epochs = epochs or dataset_cfg["epochs"]
    lr = learning_rate or dataset_cfg["learning_rate"]

    batch_size = max(1, world_batch_size // max(1, ptu.world_size))

    # --------------------
    # Dataset
    # --------------------
    dataset_kwargs = dict(
        dataset=dataset,
        image_size=im_size,
        crop_size=crop_size,
        batch_size=batch_size,
        normalization=model_cfg.get("normalization", "vit"),
        split="train",
        num_workers=10,
        root=dataset_cfg.get("data_root", dataset_cfg.get("root", None)),
    )

    train_dataset = create_dataset(dataset_kwargs)


    # --------------------
    # Validation dataset
    # --------------------
    val_dataset = None
    val_split = None
    
    for s in ["validation", "val"]:
        try:
            vkw = dataset_kwargs.copy()
            vkw["split"] = s
            vkw["batch_size"] = 1
            val_dataset = create_dataset(vkw)
            val_split = s
            print(f"Detected validation split: {s}")
            break
        except:
            continue
    
    if val_dataset is None:
        raise RuntimeError("No validation split found")
    
    
    # --------------------
    # FIXED: Build GT dict (DATASET AGNOSTIC)
    # --------------------
    # --------------------
    # BUILD GT MAP (FIXED + COMPATIBLE WITH ENGINE)
    # --------------------
    val_seg_gt = {}
    
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
    
        mask = item.get("mask", item.get("segmentation"))
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
    
        file_id = item.get("id", f"img_{idx}")
        file_id = os.path.basename(file_id)
        file_id = os.path.splitext(file_id)[0]
    
        val_seg_gt[file_id] = mask
    
    
    # --------------------
    # CRITICAL FIX: ENGINE STILL NEEDS THIS
    # --------------------
    val_dataset.dataset_gt = val_seg_gt
    train_dataset.dataset_gt = val_seg_gt


    # --------------------
    # Loaders
    # --------------------
    def make_loader(ds, shuffle):
        sampler = DistributedSampler(ds, shuffle=shuffle) if ptu.distributed else None
        return DataLoader(
            ds,
            batch_size=batch_size if shuffle else 1,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=10,
            pin_memory=True,
        )

    train_loader = make_loader(train_dataset, True)
    val_loader = make_loader(val_dataset, False)

    n_cls = train_dataset.n_cls

    # --------------------
    # Model
    # --------------------
    model_cfg["n_cls"] = n_cls
    model = create_segmenter(model_cfg)
    model.to(ptu.device)

    print(f"Model params: {num_params(model)}")

    # --------------------
    # Optimizer + Scheduler FIX
    # --------------------
    opt_args = SimpleNamespace()

    opt_args.opt = optimizer
    opt_args.lr = lr
    opt_args.weight_decay = weight_decay
    opt_args.momentum = 0.9

    opt_args.sched = scheduler
    opt_args.epochs = num_epochs
    opt_args.iter_max = len(train_loader) * num_epochs
    opt_args.iter_warmup = 0.0

    # ?? REQUIRED FOR PolynomialLR (your original crash)
    opt_args.poly_step_size = 1
    opt_args.poly_power = 0.9
    opt_args.min_lr = 1e-5

    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)

    # --------------------
    # AMP
    # --------------------
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # --------------------
    # Resume
    # --------------------
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt = log_dir / "checkpoint.pth"
    if resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # --------------------
    # Training loop
    # --------------------
    for epoch in range(num_epochs):

        if hasattr(train_loader, "sampler_obj") and train_loader.sampler_obj:
            train_loader.sampler_obj.set_epoch(epoch)

        train_logger = engine.train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            log_dir=str(log_dir),
            val_loader=val_loader,   
        )

        print(f"[Epoch {epoch+1}/{num_epochs}] {train_logger}")

        # --------------------
        # Evaluation (SAFE VERSION)
        # --------------------
        if epoch % eval_freq == 0 or epoch == num_epochs - 1:
            eval_logger = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
                log_dir=str(log_dir),
                epoch=epoch,
            )

            print(f"[Eval Epoch {epoch}] {eval_logger}")
            
            if ptu.dist_rank == 0:
                plot_eval_metrics_from_csv(str(log_dir))

        # --------------------
        # Save checkpoint
        # --------------------
        if ptu.dist_rank == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }, ckpt)

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(0)


if __name__ == "__main__":
    main()
    
    