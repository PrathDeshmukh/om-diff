import argparse
import os
import warnings

import dotenv
import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.base_datamodule import BaseDataModule
from src.data.utils import read_json
from src.models.regression.lit_module import RegressorLitModule

dotenv.load_dotenv(override=True)

MCS = {n: s for n, s in zip([77], ["Ir"])}


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for pre from a checkpoint.")
    parser.add_argument(
        "--predictor_dir_path", type=str, required=True, help="Path to the model's directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--target_key",
        type=str,
        default="barrier",
    )
    parser.add_argument("--use_last_ckpt", action="store_true")

    return parser.parse_args()


def setup_paths(args: argparse.Namespace):
    ckpt_dir_path = os.path.join(args.predictor_dir_path, "checkpoints")
    if args.use_last_ckpt:
        ckpt_path = os.path.join(ckpt_dir_path, "last.ckpt")
    else:
        ckpt_paths = [
            os.path.join(ckpt_dir_path, path)
            for path in sorted(os.listdir(ckpt_dir_path))
            if path != "last.ckpt"
        ]
        assert len(ckpt_paths) > 0
        ckpt_path = ckpt_paths[-1]

    return {
        "config": os.path.join(args.predictor_dir_path, ".hydra", "config.yaml"),
        "ckpt": ckpt_path,
        "evaluation": os.path.join(
            args.predictor_dir_path,
            "evaluation.csv",
        ),
    }


if __name__ == "__main__":
    L.seed_everything(42)

    cmd_args = parse_cmd()

    paths = setup_paths(cmd_args)

    cfg = OmegaConf.load(paths["config"])

    device = torch.device(cmd_args.device)

    print(f"Read data from checkpoint.")
    print(f"> Instantiating datamodule <{cfg.data._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")

    print(f"> Instantiating model <{cfg.model._target_}>")
    model: RegressorLitModule = hydra.utils.instantiate(cfg.model, dm=datamodule)
    model = model.to(device)

    print("> Loading checkpoint from ", paths["ckpt"])
    ckpt = torch.load(paths["ckpt"], map_location=device)

    (missing_keys, unexpected_keys) = model.load_state_dict(ckpt["state_dict"], strict=False)
    if len(missing_keys) > 0:
        warnings.warn(
            f"Some keys were missing from the 'state_dict' ({missing_keys}), this might lead to unexpected results."
        )

    if len(unexpected_keys) > 0:
        warnings.warn(
            f"Some keys were unexpected in 'state_dict' ({unexpected_keys}), this might lead to unexpected results."
        )

    test_dataset = datamodule.splits["test"]

    split_file = datamodule.hparams.split_file
    split_dict = read_json(split_file)
    fname = os.path.basename(split_file)
    fold_idx = int(fname.split("split_")[1].split(".")[0])

    test_i = 0
    records = []
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            batch = model.transfer_batch_to_device(batch, device=device, dataloader_idx=0)
            n = batch.num_data
            out = model.forward(batch=batch)[cmd_args.target_key]

            for batch_i in range(n):
                item = test_dataset.__getitem__(test_i, transform=False)
                number = [n for n in item.numbers if n in MCS]
                assert len(number) == 1
                number = number[0]

                record = {
                    "fold_idx": fold_idx,
                    "test_idx": split_dict["test"][test_i],
                    "target": getattr(item, cmd_args.target_key),
                    "mc": number,
                    "pred": out[batch_i].item(),
                }
                records.append(record)
                test_i += 1

    df = pd.DataFrame.from_records(records)
    df.to_csv(paths["evaluation"], index=False)