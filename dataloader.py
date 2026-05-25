# dataloader.py
# refactored dataloader with train/val separation and DDP support

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

        
        import torch.distributed as dist
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        

        #=========================================
        # shards (= paths to split data files)
        # =========================================

        self.train_shard_paths = [
            os.path.join(data_dir, f"/content/data/token_for_AIkenGPT/perfect_mixed_tokens_20b/perfect_mixed_000{i}_gpt2.npy") for i in range(1, 600)
        ]

        self.val_shard_paths = [
            os.path.join(data_dir, "/content/data/token_for_AIkenGPT/perfect_mixed_tokens_20b/perfect_mixed_0600_gpt2.npy")
        ]

        # =========================================
        # Read state for training
        # =========================================

        self.train_shard_index = 0
        self.train_shard_tokens = self.load_shard(self.train_shard_paths[self.train_shard_index])

        """ DELETE code
        self.train_read_position = 0
        """
        
        self.train_read_position = (
            self.rank
            * self.config.batch_size
            * self.config.input_sequence_length
        )
        

        # =========================================
        # Read state for validation
        # =========================================

        self.val_shard_index = 0
        self.val_shard_tokens = self.load_shard(self.val_shard_paths[self.val_shard_index])

        """ DELETE code
        self.val_read_position = 0
        """
        
        self.val_read_position = (
            self.rank
            * self.config.batch_size
            * self.config.input_sequence_length
        )
        

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

            """ DELETE code
            self.train_read_position += batch_size * sequence_length
            """
            
            self.train_read_position += (batch_size * sequence_length * self.world_size)
            

            # If the current shard does not have enough room for the next batch, move to the next shard.
            if (self.train_read_position + batch_size * sequence_length * self.world_size + 1 
                > len(self.train_shard_tokens)):
                self.train_shard_index = (self.train_shard_index + 1) % len(self.train_shard_paths)
                self.train_shard_tokens = self.load_shard(self.train_shard_paths[self.train_shard_index])

                
                self.train_read_position = (self.rank * batch_size * sequence_length)
                

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

            """ DELETE code
            self.val_read_position += batch_size * sequence_length
            """
            
            self.val_read_position += (batch_size * sequence_length * self.world_size)
            

            # If the current shard does not have enough room for the next batch, move to the next shard.
            if (self.val_read_position + batch_size * sequence_length * self.world_size + 1
                > len(self.val_shard_tokens)):
                self.val_shard_index = (self.val_shard_index + 1) % len(self.val_shard_paths)
                self.val_shard_tokens = self.load_shard(self.val_shard_paths[self.val_shard_index])

                
                self.val_read_position = (self.rank * batch_size * sequence_length)
                

        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # -----------------------------------------
        # Return batch
        # -----------------------------------------
        return (
            input_sequences.to(self.config.device_type),
            target_sequences.to(self.config.device_type),
        )
