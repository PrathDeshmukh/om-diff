import os.path
from typing import Optional

from src.data.ase_datamodule import ASEDataModule
from src.data.components.datasets
class VaskasDataModule(ASEDataModule):
  dataset_cls = VaskasDataset
  