import os
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from helper_functions.utils import detect_and_crop


SRC         = os.path.join('data', 'IUSTPersonReID')
DST         = os.path.join('data', 'IUSTPersonReID_crops')
SPLITS      = ['bounding_box_train', 'bounding_box_test', 'query']
YOLO_WEIGHTS = 'yolov8n.pt'
CONF_THRESH  = 0.4
PADDING      = 0.05


def process_split(split, detector):
    src_dir = os.path.join(SRC, split)
    dst_dir = os.path.join(DST, split)
    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    skipped = 0
    for fname in tqdm(files, desc=split):
        out_path = os.path.join(dst_dir, fname)
        if os.path.exists(out_path):
            skipped += 1
            continue
        img  = Image.open(os.path.join(src_dir, fname)).convert('RGB')
        crop = detect_and_crop(img, detector,
                                conf_thresh=CONF_THRESH, padding=PADDING)
        crop.save(out_path, quality=95)

    print(f'  {split}: wrote {len(files) - skipped} / skipped {skipped} existing')


def main():
    if not os.path.isdir(SRC):
        raise FileNotFoundError(f'Source dataset not found: {SRC}')

    print(f'Loading {YOLO_WEIGHTS}...')
    detector = YOLO(YOLO_WEIGHTS)

    print(f'Source:      {SRC}')
    print(f'Destination: {DST}')
    for split in SPLITS:
        if os.path.isdir(os.path.join(SRC, split)):
            process_split(split, detector)
        else:
            print(f'  {split}: missing in source, skipped')

    print('Done.')


if __name__ == '__main__':
    main()
