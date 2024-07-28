import os
import copy
import wandb
import torch
import argparse
import pandas as pd
from datetime import datetime
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from control_point_e import ControlPointE
from control_shapenet import ControlShapeNet
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
torch.set_float32_matmul_precision("high")

DATA_DIR = "data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=7e-5 * 0.4)
    parser.add_argument("--cd_filter", type=str, default="10th")
    parser.add_argument("--switch_prob", type=float, default=0.5)
    parser.add_argument("--grad_acc_steps", type=int, default=11)
    parser.add_argument("--num_val_samples", type=int, default=20)
    parser.add_argument("--cond_drop_prob", type=float, default=0.5)
    parser.add_argument("--val_data", type=str, default="chair/val")
    parser.add_argument("--train_data", type=str, default="chair/train")
    parser.add_argument("--wandb_project", type=str, default="ControlPointE")
    parser.add_argument("--outputs_dir", type=str, default="/scratch/noam/cntrl_pointe")
    args = parser.parse_args()
    return args


def build_name(args):
    name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name += f"_{args.train_data.replace('/', '_')}"
    if args.val_data is not None:
        name += f"_{args.val_data.replace('/', '_')}"
    if args.cd_filter is not None:
        name += f"_cd_filter_{args.cd_filter}"
    if args.subset_size is not None:
        name += f"_subset_{args.subset_size}"
    if args.switch_prob is not None:
        name += f"_switch_prob_{args.switch_prob}"
    return name


def load_df(data_csv, subset_size, cd_filter=None):
    df = pd.read_csv(os.path.join(DATA_DIR, data_csv))
    if cd_filter == "mean":
        df = df[df["chamfer_distance"] <= df["chamfer_distance"].mean()]
    elif cd_filter == "median":
        df = df[df["chamfer_distance"] <= df["chamfer_distance"].median()]
    elif cd_filter == "10th":
        df = df[df["chamfer_distance"] <= df["chamfer_distance"].quantile(0.1)]
    if subset_size is not None:
        df = df.head(subset_size)
    return df


def main(args):
    name = build_name(args)
    wandb.init(project=args.wandb_project, name=name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = load_df(args.train_data + ".csv", args.subset_size, args.cd_filter)
    train_dataset = ControlShapeNet(
        df=train_df,
        device=device,
        num_points=args.num_points,
        batch_size=args.batch_size,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    if args.val_data:
        val_df = load_df(args.val_data + ".csv", args.num_val_samples)
        val_dataset = ControlShapeNet(
            df=val_df,
            device=device,
            num_points=args.num_points,
            batch_size=args.num_val_samples,
        )
    else:
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.set_length(args.num_val_samples, args.num_val_samples)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.num_val_samples)
    model = ControlPointE(
        lr=args.lr,
        dev=device,
        timesteps=args.timesteps,
        num_points=args.num_points,
        batch_size=args.batch_size,
        switch_prob=args.switch_prob,
        val_data_loader=val_data_loader,
        cond_drop_prob=args.cond_drop_prob,
    )
    wandb.watch(model)
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
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
