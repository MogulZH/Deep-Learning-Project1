"""Local performance helpers; model, augmentation and optimizer stay unchanged."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Cache decoded uint8 pixels, but apply the original PIL transforms per access.

    No augmented samples are cached. Caching consumes no random numbers, so the
    original RandomCrop/RandomHorizontalFlip sequence is retained with workers=0.
    Targets and paths are prepared once instead of querying pandas per image.
    """

    def __init__(self, frame, folder, transform, labeled=True, cache_images=True):
        self.paths = [Path(folder) / name for name in frame.filename.tolist()]
        self.targets = frame["label" if labeled else "id"].to_numpy(dtype=np.int64, copy=True)
        self.transform = transform
        self.images = None
        if cache_images:
            started = perf_counter()
            self.images = np.empty((len(self.paths), 32, 32, 3), dtype=np.uint8)
            for index, path in enumerate(self.paths):
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if image.size != (32, 32):
                        raise ValueError(f"Expected a 32x32 image: {path}")
                    self.images[index] = np.asarray(image)
            print(
                f"Cached {len(self.paths):,} images "
                f"({self.images.nbytes / 2**20:.1f} MiB) in "
                f"{perf_counter() - started:.1f}s; random augmentation remains per access.",
                flush=True,
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.images is None:
            with Image.open(self.paths[index]) as image:
                x = self.transform(image.convert("RGB"))
        else:
            x = self.transform(Image.fromarray(self.images[index]))
        return x, int(self.targets[index])


class EpochMetrics:
    """Accumulate detached statistics on the device, transferring only at epoch end."""

    def __init__(self, device):
        self.loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.correct = torch.zeros((), dtype=torch.int64, device=device)
        self.count = 0

    def update(self, loss, logits, labels):
        self.loss_sum.add_(loss.detach().to(torch.float64) * len(labels))
        self.correct.add_((logits.detach().argmax(1) == labels).sum())
        self.count += len(labels)

    def result(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty loader")
        loss_sum, correct = torch.stack((self.loss_sum, self.correct.to(torch.float64))).cpu().tolist()
        return loss_sum / self.count, correct / self.count


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, amp):
    model.train()
    metrics = EpochMetrics(device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metrics.update(loss, logits, y)
    return metrics.result()


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, amp, collect_predictions=True):
    model.eval()
    metrics = EpochMetrics(device)
    labels, predictions = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        metrics.update(loss, logits, y)
        if collect_predictions:
            labels.append(y)
            predictions.append(logits.argmax(1))
    loss, accuracy = metrics.result()
    ys = torch.cat(labels).cpu().tolist() if collect_predictions else []
    preds = torch.cat(predictions).cpu().tolist() if collect_predictions else []
    return loss, accuracy, ys, preds
