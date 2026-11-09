"""Create an ID-aligned submission from your own trained checkpoint, with optional flip TTA."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cifar_runtime import ImageDataset
from model import SmallResNet

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@torch.no_grad()
def predictions(model, loader, device, tta=False):
    model.eval()
    result = {}
    for x, ids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            scores = model(x).float().softmax(dim=1)
            if tta:
                flipped = model(torch.flip(x, dims=[3])).float().softmax(dim=1)
                scores = (scores + flipped) / 2
        result.update(zip(ids.tolist(), scores.argmax(1).cpu().tolist()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('outputs/submission.csv'))
    parser.add_argument('--tta', action='store_true', help='Average softmax probabilities of original/flip views')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Only load checkpoints you trained or otherwise trust.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SmallResNet(checkpoint['config']['width']).to(device)
    model.load_state_dict(checkpoint['model'])
    frame = pd.read_csv(args.data_dir/'test.csv')
    sample = pd.read_csv(args.data_dir/'sample_submission.csv')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = ImageDataset(frame, args.data_dir/'test_images', transform, labeled=False)
    loader = DataLoader(dataset, batch_size=256, num_workers=0, pin_memory=device.type == 'cuda')
    by_id = predictions(model, loader, device, tta=args.tta)
    submission = sample[['id']].copy()
    submission['label'] = submission.id.map(by_id)
    assert submission.label.notna().all() and len(submission) == len(frame)
    submission['label'] = submission.label.astype(int)
    assert submission.label.between(0, 9).all()
    assert set(frame.id) == set(submission.id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'Saved {len(submission)} predictions; horizontal-flip probability TTA: {args.tta}')
    print('Test accuracy: unavailable locally. Kaggle computes it from hidden labels.')


if __name__ == '__main__':
    main()
