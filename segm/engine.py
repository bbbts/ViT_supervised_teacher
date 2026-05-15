#FULLY SUPERVISED

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
from segm.metrics import gather_data
import segm.utils.torch as ptu



# ---------------------------------------------------------
# LOSS TRACKING
# ---------------------------------------------------------
from segm.utils.logging_config import (
    init_history,
    append_history,
    write_csv,
    plot_losses,
    plot_metrics
)

LOSS_HISTORY = init_history()
IGNORE_LABEL = 255
EPS = 1e-6

# ---------------------------------------------------------
# HELPER: REMAP MASK LABELS
# ---------------------------------------------------------
def remap_mask(mask, label_map=None):
    """
    Remap mask values to 0..n_cls-1
    label_map: dict mapping original labels to [0..n_cls-1]
    """
    if label_map is None:
        label_map = {0: 0, 1: 1, 2: 2, 3: 3}  # update if needed
    remapped = np.full_like(mask, fill_value=IGNORE_LABEL, dtype=np.int64)
    for k, v in label_map.items():
        remapped[mask == k] = v
    return remapped

# ---------------------------------------------------------
# LOSS FUNCTIONS
# ---------------------------------------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# ---------------------------------------------------------
# PLOT LOSSES
# ---------------------------------------------------------


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train_one_epoch(model, data_loader, optimizer, lr_scheduler, epoch, amp_autocast,
                    loss_scaler=None, log_dir=None, class_weights=None, val_loader=None):

    model.train()
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    if class_weights is not None:
        weighted_ce_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(ptu.device),
            ignore_index=IGNORE_LABEL
        )
    else:
        weighted_ce_fn = ce_loss_fn

    n_cls = getattr(data_loader.dataset, "n_cls", 4)

    #ce_epoch, weighted_ce_epoch, dice_epoch, total_epoch = 0.0, 0.0, 0.0, 0.0
    ce_epoch, dice_epoch, total_epoch = 0.0, 0.0, 0.0

    for batch_idx, batch in enumerate(data_loader):
        images = batch["image"].to(ptu.device)
        masks = batch["mask"].to(ptu.device).long()

        optimizer.zero_grad()
        with amp_autocast():
            outputs = model(images)

            if batch_idx == 0:
                print("DEBUG: Model output shape:", outputs.shape)
                if outputs.shape[1] != n_cls:
                    raise ValueError(
                        f"Model output channels ({outputs.shape[1]}) != dataset classes ({n_cls})"
                    )

            ce_loss = ce_loss_fn(outputs, masks)
            weighted_ce_loss = weighted_ce_fn(outputs, masks)

            probs = torch.softmax(outputs, dim=1)
            if probs.shape[1] > 1:
                dice = torch.mean(torch.stack([
                    dice_loss(probs[:, c, :, :], (masks == c).float())
                    for c in range(probs.shape[1])
                ]))
            else:
                dice = dice_loss(probs[:, 0, :, :], masks.float())

            total_loss = ce_loss + dice

        if loss_scaler is not None:
            loss_scaler(total_loss, optimizer)
        else:
            total_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        ce_epoch += ce_loss.item()
        #weighted_ce_epoch += weighted_ce_loss.item()
        dice_epoch += dice.item()
        total_epoch += total_loss.item()

    n_batches = len(data_loader)
    ce_epoch /= n_batches
    #weighted_ce_epoch /= n_batches
    dice_epoch /= n_batches
    total_epoch /= n_batches

    val_loss_epoch = None
    if val_loader is not None:
        val_loss_epoch = compute_validation_loss(
            model, val_loader, ce_loss_fn, weighted_ce_fn, amp_autocast
        )
        
        

    # -----------------------------------
    # Compute dataset-level metrics
    # -----------------------------------
    if not hasattr(data_loader.dataset, "dataset_gt"):
        raise ValueError(
            "Dataset must define dataset_gt for evaluation."
        )
            
    metrics = evaluate(
        model,
        data_loader,
        data_loader.dataset.dataset_gt,
        amp_autocast=amp_autocast,
        epoch=epoch
    )
    
    def safe_append(key, value):
        append_history(LOSS_HISTORY, key, value)
        
        
    if metrics is not None:
    
        safe_append("CE", ce_epoch)
    
        safe_append("Dice_Loss", dice_epoch)
    
        safe_append("Total", total_epoch)
    
        safe_append("Validation", val_loss_epoch)
    
        safe_append("PixelAcc", metrics["PixelAcc"])
        safe_append("MeanIoU", metrics["MeanIoU"])
        safe_append("FWIoU", metrics["FWIoU"])
        safe_append("DiceMetric", np.mean(metrics["PerClassDice"]))
    
    if log_dir and ptu.dist_rank == 0:
        #write_csv(log_dir, epoch, LOSS_HISTORY)
        write_csv(log_dir, epoch, LOSS_HISTORY, filename="losses.csv")
        plot_losses(log_dir, LOSS_HISTORY)
        plot_metrics(log_dir, LOSS_HISTORY)

    return {
        "CE": ce_epoch,
        #"Weighted_CE": weighted_ce_epoch,
        "Dice": dice_epoch,
        "Validation": val_loss_epoch,
        "Total": total_epoch,
    }

# ---------------------------------------------------------
# VALIDATION LOSS
# ---------------------------------------------------------
def compute_validation_loss(model, val_loader, ce_fn, weighted_ce_fn, amp_autocast):
    model.eval()
    ce_sum, weighted_ce_sum, dice_sum, total_sum = 0.0, 0.0, 0.0, 0.0
    n_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(ptu.device)
            masks = batch["mask"].to(ptu.device).long()

            with amp_autocast():
                outputs = model(images)

                ce_loss = ce_fn(outputs, masks)
                weighted_ce_loss = weighted_ce_fn(outputs, masks)

                probs = torch.softmax(outputs, dim=1)
                if probs.shape[1] > 1:
                    dice = torch.mean(torch.stack([
                        dice_loss(probs[:, c, :, :], (masks == c).float())
                        for c in range(probs.shape[1])
                    ]))
                else:
                    dice = dice_loss(probs[:, 0, :, :], masks.float())

                total_loss = ce_loss + dice

            ce_sum += ce_loss.item()
            weighted_ce_sum += weighted_ce_loss.item()
            dice_sum += dice.item()
            total_sum += total_loss.item()

    val_loss = total_sum / n_batches
    return val_loss

# ---------------------------------------------------------
# EVALUATION WITH FULL DATASET DIAGNOSTICS
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_loader, val_seg_gt_raw, window_size=None, window_stride=None,
             amp_autocast=None, log_dir=None, epoch=None):

    
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()
    
    seg_pred = {}

    n_cls = getattr(data_loader.dataset, "n_cls", 4)
    gt_pixel_count = np.zeros(n_cls, dtype=np.int64)
    pred_pixel_count = np.zeros(n_cls, dtype=np.int64)
    class_image_count = np.zeros(n_cls, dtype=np.int64)
    total_images = 0

    dataset_label_summary = {c: [] for c in range(n_cls)}  # track image IDs per class

    for batch in data_loader:
        images = batch["image"].to(ptu.device)
        ids = batch["id"]

        with amp_autocast():
            outputs = model_eval(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for i, file_id in enumerate(ids):
            pred = preds[i]
            #gt_raw = val_seg_gt_raw[file_id]
            gt_raw = val_seg_gt_raw.get(file_id, None)

            if gt_raw is None:
                continue
            
            gt_raw = remap_mask(gt_raw)

            unique_labels = np.unique(gt_raw)
            for lbl in unique_labels:
                if lbl != IGNORE_LABEL:
                    dataset_label_summary[lbl].append(file_id)

            if pred.shape != gt_raw.shape:
                import cv2
                pred = cv2.resize(
                    pred.astype(np.uint8),
                    (gt_raw.shape[1], gt_raw.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            seg_pred[file_id] = pred

            gt_flat = gt_raw.flatten()
            pred_flat = pred.flatten()

            for c in range(n_cls):
                gt_pixel_count[c] += np.sum(gt_flat == c)
                pred_pixel_count[c] += np.sum(pred_flat == c)
                if np.any(gt_flat == c):
                    class_image_count[c] += 1

            total_images += 1

    # Per-class image occurrence debug
    print("\n" + "="*70)
    print(f"[DEBUG] Epoch {epoch} - True Image Occurrence per Class")
    print("-"*70)
    print("Total images evaluated:", total_images)
    for c in range(n_cls):
        print(f"Class {c} appears in {class_image_count[c]} images (sample: {dataset_label_summary[c][:5]})")
    print("="*70 + "\n")

    # Per-class pixel statistics debug
    print("\n" + "="*70)
    print(f"[DEBUG] Epoch {epoch} - Per-Class Pixel Statistics")
    print("-"*70)
    for c in range(n_cls):
        print("Class {}: GT pixels = {:<12} | Predicted pixels = {}".format(
            c, gt_pixel_count[c], pred_pixel_count[c]
        ))
    print("="*70 + "\n")

    # Compute metrics
    seg_pred = gather_data(seg_pred)
    
    val_seg_gt_filtered = {
        k: np.asarray(remap_mask(val_seg_gt_raw[k]), dtype=np.int64)
        for k in seg_pred.keys()
    }

    metrics = compute_segmentation_metrics(seg_pred, val_seg_gt_filtered, n_cls)

    # CSV logging
    if log_dir and epoch is not None:
        csv_path = os.path.join(log_dir, "evaluation_metrics.csv")
        header = ["epoch"] + list(metrics.keys())
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            row = {"epoch": epoch}
            for k, v in metrics.items():
                row[k] = list(v) if isinstance(v, np.ndarray) else v
            writer.writerow(row)

    #model.train()
    return metrics

# ---------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------
def compute_segmentation_metrics(preds, gts, n_cls):
    eps = 1e-6

    TP = np.zeros(n_cls, dtype=np.float64)
    FP = np.zeros(n_cls, dtype=np.float64)
    FN = np.zeros(n_cls, dtype=np.float64)

    GT = np.zeros(n_cls, dtype=np.float64)
    PRED = np.zeros(n_cls, dtype=np.float64)

    total_valid_pixels = 0
    total_correct_pixels = 0

    for k in preds.keys():

        pred = np.asarray(preds[k], dtype=np.int64).flatten()
        gt   = np.asarray(gts[k], dtype=np.int64).flatten()

        valid = (gt != IGNORE_LABEL)

        if valid.sum() == 0:
            continue

        pred_v = pred[valid]
        gt_v = gt[valid]

        total_valid_pixels += int(valid.sum())
        total_correct_pixels += int((pred_v == gt_v).sum())

        for c in range(n_cls):

            pred_c = (pred_v == c)
            gt_c = (gt_v == c)

            TP[c] += np.sum(pred_c & gt_c)
            FP[c] += np.sum(pred_c & (~gt_c))
            FN[c] += np.sum((~pred_c) & gt_c)

            GT[c] += np.sum(gt_c)
            PRED[c] += np.sum(pred_c)

    PerClassIoU = TP / (TP + FP + FN + eps)

    PerClassDice = 2 * TP / (2 * TP + FP + FN + eps)

    Precision = TP / (PRED + eps)

    Recall = TP / (GT + eps)

    F1 = 2 * (Precision * Recall) / (Precision + Recall + eps)

    PixelAcc = total_correct_pixels / (total_valid_pixels + eps)

    MeanIoU = float(np.mean(PerClassIoU))

    FWIoU = float(
        np.sum(
            (GT / (total_valid_pixels + eps)) * PerClassIoU
        )
    )

    return {
        "PixelAcc": PixelAcc,
        "MeanIoU": MeanIoU,
        "IoU": PerClassIoU.astype(np.float32),
        "FWIoU": FWIoU,
        "PerClassDice": PerClassDice.astype(np.float32),
        "Precision": Precision.astype(np.float32),
        "Recall": Recall.astype(np.float32),
        "F1": F1.astype(np.float32),
        "GT_pixels": GT.astype(np.int64),
        "Pred_pixels": PRED.astype(np.int64),
    }


