# dataloader.py
# refactored dataloader with train/val separation
# NOTE:
# We assume shards are pre-shuffled and pre-encoded into `.npy` files.
# This loader is designed for single-process / single-GPU runs.

import os
import torch
import numpy as np


class DataLoader:
    def __init__(self, data_dir, config):
        """
        A data loader that reads sharded data sequentially.

        Note: We use the term "shard", but in practice it is simply a list of
        paths to split `.npy` files.
        """
        self.config = config
        self.data_dir = data_dir

#=========================================
        # shards (= paths to split data files)
        # =========================================

        self.train_shard_paths = [
            os.path.join(data_dir, f"/content/data/perfect_mixed_tokens_20b/perfect_mixed_{i:04d}_gpt2.npy") for i in range(1, 100)
        ]

        self.val_shard_paths = [
            os.path.join(data_dir, "/content/data/perfect_mixed_tokens_20b/perfect_mixed_0600_gpt2.npy")
        ]

        # =========================================
        # Read state for training
        # =========================================

        self.train_shard_index = 0
        self.train_shard_tokens = self.load_shard(self.train_shard_paths[self.train_shard_index])
        self.train_read_position = 0
        

        # =========================================
        # Read state for validation
        # =========================================

        self.val_shard_index = 0
        self.val_shard_tokens = self.load_shard(self.val_shard_paths[self.val_shard_index])
        self.val_read_position = 0
        

    def load_shard(self, shard_path):
        """
        Load a shard (= one `.npy` file) and convert it to a `torch.Tensor`.
        """
        tokens_np = np.load(shard_path).astype(np.int32)
        return torch.tensor(tokens_np, dtype=torch.long)

    def get_batch(self, split):
        """
        Return the next batch sequentially from the specified split ('train' or 'val').
        """
        batch_size = self.config.batch_size
        sequence_length = self.config.input_sequence_length

        # -----------------------------------------
        # train
        # -----------------------------------------
        if split == "train":
            chunk = self.train_shard_tokens[
                self.train_read_position :
                self.train_read_position + batch_size * sequence_length + 1
            ]

            input_sequences = chunk[:-1].view(batch_size, sequence_length)
            target_sequences = chunk[1:].view(batch_size, sequence_length)

            self.train_read_position += batch_size * sequence_length
            

            # If the current shard does not have enough room for the next batch, move to the next shard.
            if (self.train_read_position + batch_size * sequence_length + 1
                > len(self.train_shard_tokens)):
                self.train_shard_index = (self.train_shard_index + 1) % len(self.train_shard_paths)
                self.train_shard_tokens = self.load_shard(self.train_shard_paths[self.train_shard_index])
                self.train_read_position = 0
                print("[INFO] Switched to train shard:", self.train_shard_paths[self.train_shard_index])
                

        # -----------------------------------------
        # validation
        # -----------------------------------------
        elif split == "val":
            chunk = self.val_shard_tokens[
                self.val_read_position :
                self.val_read_position + batch_size * sequence_length + 1
            ]

            input_sequences = chunk[:-1].view(batch_size, sequence_length)
            target_sequences = chunk[1:].view(batch_size, sequence_length)

            self.val_read_position += batch_size * sequence_length
            

            # If the current shard does not have enough room for the next batch, move to the next shard.
            if (self.val_read_position + batch_size * sequence_length + 1
                > len(self.val_shard_tokens)):
                self.val_shard_index = (self.val_shard_index + 1) % len(self.val_shard_paths)
                self.val_shard_tokens = self.load_shard(self.val_shard_paths[self.val_shard_index])
                self.val_read_position = 0
                print("[INFO] Switched to val shard:", self.val_shard_paths[self.val_shard_index])
                

        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # -----------------------------------------
        # Return batch
        # -----------------------------------------
        return (
            input_sequences.to(self.config.device_type),
            target_sequences.to(self.config.device_type),
        )
