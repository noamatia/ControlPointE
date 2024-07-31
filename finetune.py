import os
import py3nvml.py3nvml as nvml


def find_free_gpu():
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    max_free_memory = 0
    best_gpu_index = 0

    for i in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = memory_info.free

        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu_index = i

    nvml.nvmlShutdown()
    return str(best_gpu_index)


os.environ["CUDA_VISIBLE_DEVICES"] = find_free_gpu()

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
from pytorch_lightning.loggers import TensorBoardLogger


torch.set_float32_matmul_precision("high")

UTTERANCE = "utterance"
BASE_DIR = "/scratch/noam/control_point_e"
LLAMA3_WNLEMMA_UTTERANCE = "llama3_wnlemma_utterance"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--chamfer_percentile", type=float)
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=7e-5 * 0.4)
    parser.add_argument("--switch_prob", type=float, default=0.5)
    parser.add_argument("--num_val_samples", type=int, default=20)
    parser.add_argument("--val_dataset", type=str, default="chair")
    parser.add_argument("--train_dataset", type=str, default="chair")
    parser.add_argument("--wandb_project", type=str, default="ControlPointE")
    parser.add_argument("--empty_prompt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prompt_key", type=str, default=LLAMA3_WNLEMMA_UTTERANCE)
    args = parser.parse_args()
    return args


def build_name(args):
    name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name += f"_{args.prompt_key}"
    name += f"_train_{args.train_dataset}"
    if args.subset_size is not None:
        name += f"_subset_{args.subset_size}"
    if args.val_dataset:
        name += f"_val_{args.val_dataset}"
    if args.switch_prob is not None:
        name += f"_switch_{str(args.switch_prob).replace('.', '_')}"
        if args.empty_prompt:
            name += "_empty"
    if args.chamfer_percentile is not None:
        name += f"_chamfer_{str(args.chamfer_percentile).replace('.', '_')}"
    return name


def load_df(data_csv, subset_size, prompt_key, chamfer_percentile=None):
    df = pd.read_csv(os.path.join(BASE_DIR, "datasets", data_csv))
    if prompt_key == LLAMA3_WNLEMMA_UTTERANCE:
        df = df[df[LLAMA3_WNLEMMA_UTTERANCE] != "Unknown"]
    if chamfer_percentile is not None:
        df = df[
            df["chamfer_distance"] < df["chamfer_distance"].quantile(chamfer_percentile)
        ]
    if subset_size is not None and subset_size < len(df):
        df = df.sample(subset_size)
    return df


def main(args):
    name = build_name(args)
    output_dir = os.path.join(BASE_DIR, "executions", name)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.join(output_dir)
    os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
    wandb.init(project=args.wandb_project, name=name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = load_df(
        os.path.join(args.train_dataset, "train.csv"),
        args.subset_size,
        args.prompt_key,
        args.chamfer_percentile,
    )
    train_dataset = ControlShapeNet(
        df=train_df,
        device=device,
        prompt_key=args.prompt_key,
        num_points=args.num_points,
        batch_size=args.batch_size,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    if args.val_dataset:
        val_df = load_df(
            os.path.join(args.val_dataset, "val.csv"),
            args.num_val_samples,
            args.prompt_key,
        )
        val_dataset = ControlShapeNet(
            df=val_df,
            device=device,
            prompt_key=args.prompt_key,
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
        num_points=args.num_points,
        batch_size=args.batch_size,
        switch_prob=args.switch_prob,
        empty_prompt=args.empty_prompt,
        val_data_loader=val_data_loader,
    )
    wandb.watch(model)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_weights_only=True,
        every_n_epochs=args.val_freq,
        dirpath=os.path.join(output_dir, "checkpoints"),
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accumulate_grad_batches=10,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.val_freq,
        logger=TensorBoardLogger(output_dir),
    )
    trainer.fit(model, train_data_loader, val_data_loader)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
