import argparse
import copy
import os
import warnings

import dotenv
import hydra
import lightning as L
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.base_datamodule import BaseDataModule
from src.models.diffusion.lit_module import OMDiffLitModule
from src.models import ops
from src.models.diffusion.loss import DiffusionL2Loss
from src.models.diffusion.model import OMDiff

dotenv.load_dotenv(override=True)
MCS = {77: "Ir"}


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for pre from a checkpoint.")
    parser.add_argument(
        "--denoiser_dir_path", type=str, required=True, help="Path to the model's directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--n_per_timestep", type=int, default=3)
    parser.add_argument("--use_last_ckpt", action="store_true")

    return parser.parse_args()


def setup_paths(args: argparse.Namespace):
    ckpt_dir_path = os.path.join(args.denoiser_dir_path, "checkpoints")
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
        "config": os.path.join(args.denoiser_dir_path, ".hydra", "config.yaml"),
        "ckpt": ckpt_path,
        "evaluation": os.path.join(
            args.denoiser_dir_path,
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
    model: OMDiffLitModule = hydra.utils.instantiate(cfg.model, dm=datamodule)
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

    records = []

    with torch.inference_mode():
        ts = torch.arange(
            0, model.om_diff.timesteps + 1, step=cmd_args.step, device=device, dtype=torch.long
        )
        ts[-1] = model.om_diff.timesteps - 1
        for t in ts:
            for t_i in range(cmd_args.n_per_timestep):
                test_i = 0
                for batch in datamodule.test_dataloader():
                    batch = model.transfer_batch_to_device(batch, device=device, dataloader_idx=0)
                    n = batch.num_data
                    ts = torch.full(
                        (n, 1),
                        fill_value=t.item(),
                        device=batch.node_features.device,
                    )

                    if not isinstance(model.om_diff, OMDiff):
                        batch.node_positions = ops.center_splits(
                            batch.node_positions, batch.num_nodes
                        )

                    batch_copy = copy.deepcopy(batch)

                    noisy_inputs, target_noise, gamma_t = model.om_diff.forward_diffusion(batch, ts)
                    (
                        alpha_t,
                        sigma_t,
                    ) = model.om_diff.noise_model.noise_schedule.get_alpha_sigma_from_gamma(gamma_t)
                    alpha_t = torch.repeat_interleave(alpha_t, batch.num_nodes, dim=0)
                    sigma_t = torch.repeat_interleave(sigma_t, batch.num_nodes, dim=0)

                    noisy_batch_copy = copy.deepcopy(noisy_inputs)

                    predicted_noise = model.om_diff.forward(noisy_inputs, ts)

                    target_x = {key: getattr(batch_copy, key) for key in target_noise}
                    predicted_x = {
                        key: (getattr(noisy_batch_copy, key) - sigma_t * predicted_noise[key])
                        / alpha_t
                        for key in target_noise
                    }

                    num_nodes = batch.num_nodes
                    if isinstance(model.om_diff, OMDiff):
                        print(f"> Doing masking ...")
                        # remove masked nodes to avoid biasing the loss
                        mask = ~noisy_inputs.node_mask.view(-1)  # OBS: not
                        for key in target_noise:
                            target_x[key] = target_x[key][mask]
                            predicted_x[key] = predicted_x[key][mask]

                            target_noise[key] = target_noise[key][mask]
                            predicted_noise[key] = predicted_noise[key][mask]
                        num_nodes = num_nodes - ops.sum_splits(
                            noisy_inputs.node_mask.long(), num_nodes
                        ).view(-1)

                    loss_eps, loss_eps_dict = DiffusionL2Loss.l2_loss(
                        predicted_noise,
                        target_noise,
                        splits=num_nodes,
                    )

                    loss_x, loss_x_dict = DiffusionL2Loss.l2_loss(
                        predicted_x,
                        target_x,
                        splits=num_nodes,
                    )

                    for batch_i in range(n):
                        item = test_dataset.__getitem__(test_i, transform=False)
                        number = [n for n in item.numbers if n in MCS]
                        assert len(number) == 1
                        number = number[0]

                        record = {
                            "t": t.item(),
                            "sample_idx": t_i,
                            "mc": number,
                            **{
                                f"eps_{key}": loss_eps_dict[key][batch_i].item()
                                for key in loss_eps_dict
                            },
                            **{
                                f"x_{key}": loss_x_dict[key][batch_i].item() for key in loss_x_dict
                            },
                        }
                        records.append(record)
                        test_i += 1

    df = pd.DataFrame.from_records(records)
    df.to_csv(paths["evaluation"], index=False)