import os.path
from typing import Optional

from src.data.ase_datamodule import ASEDataModule
from src.data.components.datasets.vaskas import VaskasDataset

class VaskasDataModule(ASEDataModule):
  dataset_cls = VaskasDataset
  dataset: Optional[VaskasDataset]
  splits: Optional[dict[str, VaskasDataset]]
  
  def prepare_data(self) -> None:
    if not os.path.exists(self.hparams.db_path):
      VaskasDataset(self.hparams.db_path, download=True, in_memory=False)
  