# added torch.compile
import math

def get_learning_rate(current_step, config):
    max_learning_rate = config.max_learning_rate
    min_learning_rate = config.min_learning_rate
    warmup_steps = config.warmup_steps
    total_training_steps = config.total_training_steps

    if current_step < warmup_steps:
        # --- Linear Warmup ---
        warmup_progress_ratio = current_step / warmup_steps
        learning_rate = max_learning_rate * warmup_progress_ratio

    else:
        # --- Cosine Decay ---
        decay_step_index = current_step - warmup_steps
        decay_total_steps = total_training_steps - warmup_steps
        decay_progress_ratio = decay_step_index / decay_total_steps

        cosine_decay_value = math.cos(math.pi * decay_progress_ratio)
        cosine_decay_ratio = 0.5 * (1.0 + cosine_decay_value)

        learning_rate_range = max_learning_rate - min_learning_rate
        learning_rate = min_learning_rate + cosine_decay_ratio * learning_rate_range

    return learning_rate


import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, data_loader, config, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config
        self.start_step = 0
        self.checkpoint_dir = checkpoint_dir
        
        self.steps = []
        self.learning_rates = []
        self.train_losses = []
        self.val_losses = []
        self.tokens_per_second_list = []
        self.total_seen_tokens_list = []
        self.total_train_time_list = []

    def save_checkpoint(self, current_step):
        checkpoint_data = {
            "current_step": current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.config),
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{current_step:06d}.pt"
        )

        torch.save(checkpoint_data, checkpoint_path)
        print(f"[INFO] Successfully saved checkpoint at step {current_step:06d}")


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location=self.config.device_type)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        self.start_step = checkpoint_data["current_step"] + 1
        print(f"[INFO] Resume. Loaded checkpoint from step {checkpoint_data['current_step']}")


    def update_learning_rate(self, current_step):
        learning_rate = get_learning_rate(current_step, self.config)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate


    def train_step(self):
        input_batch, target_batch = self.data_loader.get_batch('train')
        self.optimizer.zero_grad()

        ### NEW ###
        with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
        ### NEW ###
            logits, loss = self.model(input_batch, target_batch)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self):
        ### NEW ###
        with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
        ### NEW ###
            self.model.eval()  # switch to eval mode
            losses = {"train": [], "val": []} # compute losses for both train/val splits
            with torch.no_grad():
                for split in ['train', 'val']:
                    for _ in range(self.config.evaluation_loops):
                        input_batch, target_batch = self.data_loader.get_batch(split)
                        _, loss = self.model(input_batch, target_batch)
                        losses[split].append(loss.item())
            self.model.train()  # switch back to train mode

        # Return mean loss for each dataset (train, val)
        return {split: sum(values) / len(values) for split, values in losses.items()}

    def train(self):
        last_eval_end_time = None
        total_train_time = 0.0
        # Run train_step for (total_training_steps + 1) iterations.
        for step in range(self.start_step, self.config.total_training_steps + 1):
            # One training step (main work done every iteration)
            # update lr
            self.update_learning_rate(step)
            train_loss = self.train_step()
            
            if step > 0 and step % self.config.checkpoint_save_frequency == 0:
              self.save_checkpoint(step)

            # Evaluate every evaluation_frequency steps.
            if step % self.config.evaluation_frequency == 0:
                if last_eval_end_time is None: # step==0 and checkpoint resume
                    tokens_per_second = None
                else: # compute tokens/s if last_eval_end_time is available
                    current_eval_start_time = time.time()
                    evaluation_interval = current_eval_start_time - last_eval_end_time
                    total_train_time += evaluation_interval
                    tokens_per_evaluation_interval = self.config.batch_size * self.config.input_sequence_length * self.config.evaluation_frequency
                    tokens_per_second = tokens_per_evaluation_interval / evaluation_interval

                eval_loss = self.evaluate()
                total_seen_tokens = self.config.batch_size * self.config.input_sequence_length * step        
                current_learning_rate = self.optimizer.param_groups[0]["lr"]
    

                print(
                    f"step {step:05d} | "
                    f"lr {current_learning_rate:.6e} | "
                    f"train loss {eval_loss['train']:.4f} | "
                    f"val loss {eval_loss['val']:.4f} | "
                    f"tok/s {int(tokens_per_second) if tokens_per_second is not None else 'None'} | "
                    f"tokens {total_seen_tokens:,} | "
                    f"time {total_train_time:.2f}s"
                )

                self.steps.append(step)
                self.learning_rates.append(current_learning_rate)
                self.train_losses.append(eval_loss['train'])
                self.val_losses.append(eval_loss['val'])
                self.tokens_per_second_list.append(tokens_per_second)
                self.total_seen_tokens_list.append(total_seen_tokens)
                self.total_train_time_list.append(total_train_time)

                # Record when this evaluation finishes. The delta to the next evaluation start
                # becomes `evaluation_interval`.
                last_eval_end_time = time.time()

        # Save the final model if training completes successfully
        self.save_checkpoint(self.config.total_training_steps)
