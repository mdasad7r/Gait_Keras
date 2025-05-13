import os
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image

class CASIABSequenceLoader(tf.keras.utils.Sequence):
    def __init__(self, root_dir, sequence_len=None, batch_size=4, image_size=(64, 64), shuffle=True):
        self.root_dir = root_dir
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

        # Scan dataset
        self.sequence_paths = glob(os.path.join(root_dir, "*/*"))
        self.sequence_paths = [p for p in self.sequence_paths if os.path.isdir(p)]
        self.label_map = self._create_label_map()
        self.indexes = np.arange(len(self.sequence_paths))

        print(f"📂 Found {len(self.sequence_paths)} sequences in {root_dir}")
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _create_label_map(self):
        subjects = sorted(set(os.path.basename(os.path.normpath(p).split(os.sep)[-2]) for p in self.sequence_paths))
        return {sid: idx for idx, sid in enumerate(subjects)}

    def __len__(self):
        return int(np.ceil(len(self.sequence_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.sequence_paths[i] for i in batch_indexes]

        batch_x, batch_y = [], []
        for seq_path in batch_paths:
            frames = self._load_sequence(seq_path)
            label_name = os.path.basename(os.path.normpath(seq_path).split(os.sep)[-2])
            label = self.label_map[label_name]
            batch_x.append(frames)
            batch_y.append(label)

        batch_x = np.stack(batch_x)  # [B, T, H, W, 1]
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=len(self.label_map))
        return batch_x, batch_y

    def _load_sequence(self, seq_path):
        frame_paths = sorted(glob(os.path.join(seq_path, "*.png")))
        frames = []

        for img_path in frame_paths:
            img = Image.open(img_path).convert("L").resize(self.image_size)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
            frames.append(arr[..., np.newaxis])  # Add channel dim

        sequence = np.stack(frames, axis=0)  # [T, H, W, 1]

        # Pad or truncate
        T = sequence.shape[0]
        if self.sequence_len:
            if T < self.sequence_len:
                pad = np.zeros((self.sequence_len - T, *sequence.shape[1:]), dtype=np.float32)
                sequence = np.concatenate([sequence, pad], axis=0)
            elif T > self.sequence_len:
                sequence = sequence[:self.sequence_len]

        return sequence

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
