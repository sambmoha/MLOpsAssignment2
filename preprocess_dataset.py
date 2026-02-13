import argparse
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageEnhance


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def split_counts(n, train_frac, val_frac, test_frac):
    train_n = int(math.floor(n * train_frac))
    val_n = int(math.floor(n * val_frac))
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def ensure_dirs(root: Path, splits, classes):
    for split in splits:
        for cls in classes:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def safe_open_rgb(path: Path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


def resize_save(img: Image.Image, out_path: Path, size: int):
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=95)


def augment_image(img: Image.Image, rng: random.Random):
    # Random horizontal flip
    if rng.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # Random rotation (-15 to 15 degrees)
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

    # Random brightness/contrast
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(rng.uniform(0.8, 1.2))

    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(rng.uniform(0.8, 1.2))

    return img


def process_class(
    cls_name,
    files,
    out_root,
    size,
    splits,
    train_frac,
    val_frac,
    test_frac,
    augment,
    aug_per_image,
    seed,
):
    rng = random.Random(seed)
    rng.shuffle(files)

    train_n, val_n, test_n = split_counts(len(files), train_frac, val_frac, test_frac)
    split_files = {
        "train": files[:train_n],
        "val": files[train_n : train_n + val_n],
        "test": files[train_n + val_n :],
    }

    stats = {"kept": 0, "skipped": 0, "augmented": 0}

    for split, split_list in split_files.items():
        for src in split_list:
            img = safe_open_rgb(src)
            if img is None:
                stats["skipped"] += 1
                continue

            base = src.stem
            out_path = out_root / split / cls_name / f"{base}.jpg"
            resize_save(img, out_path, size)
            stats["kept"] += 1

            if split == "train" and augment:
                for i in range(aug_per_image):
                    aug_img = augment_image(img, rng)
                    aug_path = out_root / split / cls_name / f"{base}_aug{i+1}.jpg"
                    resize_save(aug_img, aug_path, size)
                    stats["augmented"] += 1

    return stats, (train_n, val_n, test_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/dog-and-cat-classification-dataset/PetImages")
    parser.add_argument("--output", default="data/processed_224")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-per-image", type=int, default=1)
    args = parser.parse_args()

    total_frac = args.train_split + args.val_split + args.test_split
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test splits must sum to 1.0")

    source = Path(args.source)
    out_root = Path(args.output)

    classes = ["Cat", "Dog"]
    for cls in classes:
        if not (source / cls).exists():
            raise FileNotFoundError(f"Missing class folder: {source/cls}")

    ensure_dirs(out_root, ["train", "val", "test"], classes)

    summary = {}
    for cls in classes:
        files = list_images(source / cls)
        stats, split_sizes = process_class(
            cls,
            files,
            out_root,
            args.img_size,
            ["train", "val", "test"],
            args.train_split,
            args.val_split,
            args.test_split,
            args.augment,
            args.aug_per_image,
            args.seed,
        )
        summary[cls] = {"files": len(files), "split": split_sizes, **stats}

    print("Output:", out_root)
    for cls, info in summary.items():
        train_n, val_n, test_n = info["split"]
        print(
            f"{cls}: total={info['files']} train={train_n} val={val_n} test={test_n} "
            f"kept={info['kept']} skipped={info['skipped']} augmented={info['augmented']}"
        )


if __name__ == "__main__":
    main()
