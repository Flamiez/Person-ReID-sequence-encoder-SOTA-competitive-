import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .object_models import ParsedImage
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt

class PersonReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True, max_frame_gap=50, min_seq_len=2):
        self.root_dir = root_dir
        self.is_training = is_training
        self.parsed_images = []
        self.max_frame_gap = max_frame_gap
        self.min_seq_len = min_seq_len
        self.sequences_by_person = {}
        
        # Filename example 0005_c4s1_002993_01.jpg

        for filename in os.listdir(root_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                parts = filename.split(sep="_")

                person_id = int(parts[0])
                if person_id == -1:
                    continue
                parsed_image = ParsedImage(image_path=os.path.join(root_dir, filename),
                                           person_id=person_id,
                                           camera_id=parts[1],
                                           frame_id=int(parts[2]))
                self.parsed_images.append(parsed_image)
        
        self.parsed_images = sorted(self.parsed_images,
                                    key=lambda x: (x.person_id, x.camera_id, x.frame_id))
        self.parsed_images = np.array(self.parsed_images)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        self._split_sequences_by_frame_gap()
    
    def _split_sequences_by_frame_gap(self):
        sequences = []
        current_seq = []

        for img in self.parsed_images:
            if not current_seq:
                current_seq.append(img)
                continue

            last = current_seq[-1]
            same_person_cam = (img.person_id == last.person_id and img.camera_id == last.camera_id)
            frame_gap = img.frame_id - last.frame_id

            if same_person_cam and frame_gap <= self.max_frame_gap:
                current_seq.append(img)
            else:
                if len(current_seq) >= self.min_seq_len:
                    sequences.append(current_seq)
                current_seq = [img]

        if current_seq and len(current_seq) >= self.min_seq_len:
            sequences.append(current_seq)

        self.parsed_images = [img for seq in sequences for img in seq]

        for seq in sequences:
            pid = seq[0].person_id
            cam = seq[0].camera_id
            self.sequences_by_person.setdefault(pid, {}).setdefault(cam, []).append(seq)

        
    def __len__(self):
        return len(self.parsed_images)
    
    def __getitem__(self, idx):
        parsed_image = self.parsed_images[idx]
        image = Image.open(parsed_image.image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'person_id': parsed_image.person_id,
            'camera_id': parsed_image.camera_id,
            'frame_id': parsed_image.frame_id,
            'image_path': parsed_image.image_path
        }


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    person_ids = torch.tensor([item['person_id'] for item in batch])
    camera_ids = [item['camera_id'] for item in batch]
    frame_ids = torch.tensor([item['frame_id'] for item in batch])
    
    return {
        'image': images,
        'person_id': person_ids,
        'camera_id': camera_ids,
        'frame_id': frame_ids
    }

def extract_features(dataset, model, device='cuda'):
    batch_size = 32
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    features = []
    labels = []
    camera_ids = []
    frame_ids = []
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch['image'].to(device)
            batch_features = model(images)
            batch_features = batch_features.cpu().numpy()
            
            features.append(batch_features)
            labels.extend(batch['person_id'].numpy())
            camera_ids.extend(batch['camera_id'])
            frame_ids.extend(batch['frame_id'].numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    frame_ids = np.array(frame_ids)
    
    return features, labels, camera_ids, frame_ids

def prepare_data(data_dir, model, is_training=False):

    dataset = PersonReIDDataset(data_dir, is_training=is_training)
    features, labels, camera_ids, frame_ids = extract_features(dataset, model)
    return features, labels, camera_ids, frame_ids


class Market1501Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.jpg')]
        
        self.labels = []
        for img in os.listdir(data_dir):
            if img.endswith('.jpg'):
                person_id = img.split('_')[0]
                self.labels.append(int(person_id) if person_id != '-1' else -1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class SequenceReIDDataset(Dataset):
    """
    Each sample is one complete temporal sequence (variable length).
    Built on top of PersonReIDDataset's sequence-splitting logic.

    Returns per sample:
        images:    (T, C, H, W)  float tensor
        length:    int            number of frames
        person_id: int
        camera_id: str
    """
    def __init__(self, root_dir, transform=None, max_seq_len=None, max_frame_gap=50):
        self.max_seq_len = max_seq_len

        # Reuse existing parsing + sequence-splitting logic
        _base = PersonReIDDataset(root_dir, transform=transform,
                                  max_frame_gap=max_frame_gap)
        self.transform = _base.transform

        # Flatten sequences_by_person → list of sequences
        self.sequences = []
        for pid, cams in _base.sequences_by_person.items():
            for cam, seqs in cams.items():
                for seq in seqs:
                    self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.max_seq_len is not None:
            seq = seq[:self.max_seq_len]

        frames = []
        for parsed_img in seq:
            img = Image.open(parsed_img.image_path).convert('RGB')
            frames.append(self.transform(img))

        return {
            'images': torch.stack(frames),   # (T, C, H, W)
            'length': len(frames),
            'person_id': seq[0].person_id,
            'camera_id': seq[0].camera_id,
        }


def sequence_collate_fn(batch):
    """Collate variable-length sequences by zero-padding to the longest in the batch."""
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    max_len = lengths.max().item()

    T, C, H, W = batch[0]['images'].shape[0], *batch[0]['images'].shape[1:]
    padded = torch.zeros(len(batch), max_len, C, H, W)
    for i, item in enumerate(batch):
        t = item['length']
        padded[i, :t] = item['images']

    return {
        'images': padded,                                                      # (B, T_max, C, H, W)
        'lengths': lengths,                                                    # (B,)
        'person_id': torch.tensor([item['person_id'] for item in batch]),
        'camera_id': [item['camera_id'] for item in batch],
    }


def display_sequences(sequences):
    for i, seq in enumerate(sequences):
        fig, axes = plt.subplots(1, len(seq), figsize=(3 * len(seq), 3))

        for j, parsed_img in enumerate(seq):
            img = Image.open(parsed_img.image_path).convert('RGB')
            img = np.array(img)
            axes[j].imshow(img)
            axes[j].set_title(f"PID {parsed_img.person_id}\n{parsed_img.camera_id}\nF{parsed_img.frame_id}")
            axes[j].axis("off")
        plt.show()

def get_image_palette(image, mask, n_colors=5):
    masked_pixels = image[mask > 0]
    if len(masked_pixels) == 0:
        return np.zeros((n_colors, 3), dtype=np.uint8)
    
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(masked_pixels)
    colors = np.clip(kmeans.cluster_centers_.astype(np.uint8), 0, 255)
    return colors

def plot_palette(colors, ax):
    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    step = 300 // len(colors)
    for i, color in enumerate(colors):
        palette[:, i * step:(i + 1) * step, :] = color
    ax.imshow(palette)
    ax.axis("off")

def display_img_mask_palette(images_list, masks_list):
    for i, (images, masks) in enumerate(zip(images_list, masks_list)):
        fig, axes = plt.subplots(3, len(images), figsize=(3 * len(images), 8))

        for j, (image, mask) in enumerate(zip(images, masks)):
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            green_overlay = np.zeros_like(mask_rgb)
            green_overlay[:, :, 1] = mask
            colors = get_image_palette(image, mask, n_colors=5)
            axes[0, j].imshow(image)
            axes[0, j].set_title(f"{j+1}")
            axes[1, j].imshow(cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0))
            plot_palette(colors, axes[2, j])

            for k in range(3):
                axes[k, j].axis("off")

        plt.tight_layout()
        plt.show()

def display_masks(images_list, masks_list):
    for i, (images, masks) in enumerate(zip(images_list, masks_list)):
        fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
        for j, (image, mask) in enumerate(zip(images, masks)):
            axes[j].imshow(cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0))
            axes[j].axis("off")
        plt.show()

def get_person_masks_for_sequence(sequence, model):
    images = []
    masks = []

    for item in sequence:
        image = cv2.cvtColor(cv2.imread(item.image_path), cv2.COLOR_BGR2RGB)
        images.append(image)
        mask_combined = get_image_mask(image, model)
        masks.append(mask_combined)

    return images, masks

def get_image_mask(image, model):
    results = model(image)
    result = results[0]

    if result.masks is not None:
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for m in result.masks.data.cpu().numpy():
            mask_resized = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_combined = np.maximum(mask_combined, mask_resized.astype(np.uint8))
    else:
        mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    return mask_combined


def detect_and_crop(pil_image, detector, conf_thresh=0.4, padding=0.05):
    """
    Run YOLOv8 person detection on a PIL image and return the highest-confidence
    person crop. Falls back to the original image if no person is detected.

    Args:
        pil_image  : PIL.Image (RGB)
        detector   : ultralytics YOLO model
        conf_thresh: minimum detection confidence
        padding    : fractional padding added around the bounding box

    Returns:
        PIL.Image: cropped person region, or the original image as fallback
    """
    img_np = np.array(pil_image)
    results = detector(img_np, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes):
        cls  = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        person_mask = (cls == 0) & (conf >= conf_thresh)
        if person_mask.any():
            best = xyxy[person_mask][conf[person_mask].argmax()]
            x1, y1, x2, y2 = best
            w, h = pil_image.size
            pad_x = (x2 - x1) * padding
            pad_y = (y2 - y1) * padding
            x1 = max(0.0, x1 - pad_x)
            y1 = max(0.0, y1 - pad_y)
            x2 = min(float(w), x2 + pad_x)
            y2 = min(float(h), y2 + pad_y)
            return pil_image.crop((x1, y1, x2, y2))

    return pil_image  # fallback: full frame


class MARSDataset:
    """
    Loads the MARS video ReID dataset.

    MARS structure:
        bounding_box_train/{pid}/{pid}C{cam}T{tracklet}F{frame}.jpg
        bounding_box_test/{pid}/  (includes '00-1' distractors and '0000' background — skipped)

    Filename format: 0001C1T0001F001.jpg
        pid      = first 4 chars
        cam      = char after 'C' (single digit)
        tracklet = 4 chars after 'T'
        frame    = 3 chars after 'F'

    Produces the same sequences_by_person dict as PersonReIDDataset:
        {pid: {cam_id: [[ParsedImage, ...], ...]}}
    so the entire training/eval pipeline works unchanged.

    Args:
        root_dir    : path to bounding_box_train or bounding_box_test
        min_seq_len : discard tracklets shorter than this (default 2)
    """
    def __init__(self, root_dir, min_seq_len=2):
        self.root_dir  = root_dir
        self.sequences_by_person = {}

        # Each person has its own subdirectory
        for pid_str in sorted(os.listdir(root_dir)):
            # Skip distractors (00-1) and background (0000)
            if pid_str in ('00-1', '0000'):
                continue
            pid_dir = os.path.join(root_dir, pid_str)
            if not os.path.isdir(pid_dir):
                continue

            try:
                pid = int(pid_str)
            except ValueError:
                continue

            # Group frames by (cam, tracklet)
            tracklets = {}   # (cam, tracklet) -> [ParsedImage]
            for fname in sorted(os.listdir(pid_dir)):
                if not fname.lower().endswith('.jpg'):
                    continue
                try:
                    # e.g. 0001C1T0001F001.jpg
                    c_idx  = fname.index('C')
                    t_idx  = fname.index('T')
                    f_idx  = fname.index('F')
                    cam    = fname[c_idx + 1 : t_idx]
                    trk    = fname[t_idx + 1 : f_idx]
                    frame  = int(fname[f_idx + 1 : fname.index('.')])
                except (ValueError, IndexError):
                    continue

                key = (cam, trk)
                img = ParsedImage(
                    image_path=os.path.join(pid_dir, fname),
                    person_id=pid,
                    camera_id=cam,
                    frame_id=frame,
                )
                tracklets.setdefault(key, []).append(img)

            for (cam, _), frames in tracklets.items():
                if len(frames) < min_seq_len:
                    continue
                frames.sort(key=lambda x: x.frame_id)
                self.sequences_by_person \
                    .setdefault(pid, {}) \
                    .setdefault(cam, []) \
                    .append(frames)


class DetectionCropTransform:
    """
    Drop-in transform that detects and crops the dominant person before any
    other transforms run.  Place it first in a transforms.Compose pipeline
    when working with full-frame images (e.g. IUSTPersonReID).

    Example::

        transform = transforms.Compose([
            DetectionCropTransform(yolo_model),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    """
    def __init__(self, detector, conf_thresh=0.4, padding=0.05):
        self.detector   = detector
        self.conf_thresh = conf_thresh
        self.padding    = padding

    def __call__(self, pil_image):
        return detect_and_crop(pil_image, self.detector,
                               conf_thresh=self.conf_thresh,
                               padding=self.padding)



# Find person in images sequence and then rank them based by most recent sequences found, If many sequences which are found lets say when using 4 consecutive images, sort them by most recent ones and by similarity assigning weights of importance to both



