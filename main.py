# main.py
import os
import torch

from config import ModelConfig
from dataloader import DataLoader
from model import GPT
from train import Trainer


def main():
    config = ModelConfig()

    # (Optional) common settings for accuracy and speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Model/optimization
    model = GPT(config=config)
    device = torch.device(config.device_type if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Call torch.compile only once
    model = torch.compile(model)

    # close to nanoGPT setting
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # DataLoader
    data_dir = os.environ.get("DATA_DIR", "/content/data") # ex: /home/ubuntu/virginia-filesystem
    data_loader = DataLoader(data_dir=data_dir, config=config)

    checkpoint_dir = os.environ.get("CKPT_DIR", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    from huggingface_hub import create_repo
    repo_id = "aoUTlum/AIkenGPT-checkpoints-test" # TODO: FILL example: "HayatoHongo/EveryonesGPT-checkpoints"
    create_repo(repo_id=repo_id, private=False, exist_ok=True)

    #import os
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
    )
    
# 🛠️ 【修正】自動ダウンロード機能付きのロード処理
    from huggingface_hub import hf_hub_download

    resume_checkpoint_name = "checkpoint_097000.pt"  # 指定するチェックポイント名
    resume_path = os.path.join(checkpoint_dir, resume_checkpoint_name)

    # 1. ローカルにファイルが存在しない場合、Hugging Faceからダウンロードを実行
    if not os.path.exists(resume_path):
        print(f"[INFO] Local checkpoint not found. Downloading {resume_checkpoint_name} from Hugging Face...")
        try:
            # local_dirを指定することで、隠しキャッシュではなく ./checkpoints 直下に直接配置される
            hf_hub_download(
                repo_id=repo_id,
                filename=resume_checkpoint_name,
                local_dir=checkpoint_dir
            )
            print("[INFO] Download completed successfully.")
        except Exception as e:
            print(f"[WARN] Failed to download {resume_checkpoint_name}: {e}")

    # 2. ファイルの存在確認（ダウンロード成功、または既にローカルに存在した場合に発火）
    if os.path.exists(resume_path):
        print(f"[INFO] Found checkpoint: {resume_path}. Initializing resume process...")
        trainer.load_checkpoint(resume_path)
    else:
        print("[INFO] No specific checkpoint found. Starting training from scratch (step 0).")

    trainer.train()

if __name__ == "__main__":
    main()
