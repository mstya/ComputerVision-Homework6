import os
import sys
from PIL import Image

ROTATIONS = [90, 180, 270]
SUFFIXES = tuple(f'_rot{r}' for r in ROTATIONS)
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def already_augmented(stem):
    return any(stem.endswith(s) for s in SUFFIXES)


def augment_dataset(src_dir, dst_dir):
    total_original = 0
    total_generated = 0

    for class_name in sorted(os.listdir(src_dir)):
        class_src = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_src):
            continue

        class_dst = os.path.join(dst_dir, class_name)
        os.makedirs(class_dst, exist_ok=True)

        for fname in sorted(os.listdir(class_src)):
            if not fname.lower().endswith(IMG_EXTENSIONS):
                continue
            stem, ext = os.path.splitext(fname)
            if already_augmented(stem):
                continue

            src_path = os.path.join(class_src, fname)
            img = Image.open(src_path).convert('RGB')

            dst_original = os.path.join(class_dst, fname)
            if not os.path.exists(dst_original):
                img.save(dst_original)

            for angle in ROTATIONS:
                out_name = f"{stem}_rot{angle}{ext}"
                out_path = os.path.join(class_dst, out_name)
                if not os.path.exists(out_path):
                    img.rotate(angle, expand=False).save(out_path)
                    total_generated += 1

            total_original += 1

        print(f"  {class_name}: done")

    print(f"\nOriginal images: {total_original}")
    print(f"Generated images: {total_generated}")
    print(f"Total in output: {total_original + total_generated}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        src = sys.argv[1]
        dst = sys.argv[2]
    elif len(sys.argv) == 2:
        src = sys.argv[1]
        dst = sys.argv[1]
    else:
        src = './Aerial_Landscapes'
        dst = './Aerial_Landscapes'

    print(f"Source:      {src}")
    print(f"Destination: {dst}\n")
    augment_dataset(src, dst)
