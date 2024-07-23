import os
import copy
import wandb
import torch
import argparse
import pandas as pd
from pointe import PointE
from shapenet import ShapeNet
from datetime import datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=7e-5 * 0.4)
    parser.add_argument("--grad_acc_steps", type=int, default=11)
    parser.add_argument("--num_val_samples", type=int, default=20)
    parser.add_argument("--cond_drop_prob", type=float, default=0.5)
    parser.add_argument("--wandb_project", type=str, default="PointE")
    parser.add_argument("--train_data", type=str, default="chair.csv")
    parser.add_argument("--outputs_dir", type=str, default="/scratch/noam/pointe")
    parser.add_argument("--prompt_key", type=str, default="llama3_target_utterance")
    args = parser.parse_args()
    return args


def build_name(args):
    name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name += f"_{args.train_data.replace('.csv', '')}"
    if args.val_data is not None:
        name += f"_{args.val_data.replace('/', '_')}"
    if args.subset_size is not None:
        name += f"_subset_{args.subset_size}"
    name += f"_{args.prompt_key}"
    return name


def load_df(data_csv, subset_size):
    df = pd.read_csv(os.path.join(data_csv))
    if subset_size is not None:
        df = df.head(subset_size)
    return df


def main(args):
    name = build_name(args)
    wandb.init(project=args.wandb_project, name=name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = load_df(args.train_data, args.subset_size)
    train_dataset = ShapeNet(
        df=train_df,
        device=device,
        num_points=args.num_points,
        batch_size=args.batch_size,
        prompt_key=args.prompt_key,
    )
    print(f"Train dataset length: {len(train_dataset)}")
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    if args.val_data:
        val_df = load_df(args.val_data, args.num_val_samples)
        val_dataset = ShapeNet(
            df=val_df,
            device=device,
            prompt_key=args.prompt_key,
            num_points=args.num_points,
            batch_size=args.num_val_samples,
        )
    else:
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.set_length(args.num_val_samples, args.num_val_samples)
    print(f"Val dataset length: {len(val_dataset)}")
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.num_val_samples)
    model = PointE(
        lr=args.lr,
        dev=device,
        timesteps=args.timesteps,
        num_points=args.num_points,
        batch_size=args.batch_size,
        val_data_loader=val_data_loader,
        cond_drop_prob=args.cond_drop_prob,
    )
    wandb.watch(model)
    os.makedirs(os.path.join(args.outputs_dir, name), exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_weights_only=True,
        every_n_epochs=args.val_freq,
        dirpath=os.path.join(args.outputs_dir, name),
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.val_freq,
        accumulate_grad_batches=args.grad_acc_steps,
    )
    trainer.fit(model, train_data_loader, val_data_loader)
    torch.save(
        model.model.state_dict(), os.path.join(args.outputs_dir, "model_weights.pt")
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
