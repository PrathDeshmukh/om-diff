import os
import shutil
import urllib.request as requests

import ase
import ase.io
import numpy as np
import pandas as pd
import tqdm
from ase.db import connect

from src.data.components.datasets.ase_dataset import ASEDBDataset
from src.data.components.transforms.base import Transform

class VaskasDataset(ASEDBDataset):
  distant_url = "https://github.com/pascalfriederich/vaskas-space/archive/refs/heads/master.zip"
  structures_path = "data/coordinates_complex/"
  energies_path = "data/vaskas_features_properties_smiles_filenames.csv"
  
  usecols = ["barrier", "smiles", "filename"]
  index_col = "filename"
  
  expected_length = 1_947
  
  def __init__(
      self,
      db_path,
      transform: Transform = lambda _x: _x,
      download: bool = False,
      idx_subset: np.ndarray = None,
      in_memory: bool = True,
      **kwargs,
  ):
    if download:
      dir_path = os.path.dirname(db_path)
      db_name = os.path.basename(db_path)
      self.download_dataset(dir_path, db_name)
    super(VaskasDataset, self).__init__(
      db_path=db_path,
      transform=transform,
      idx_subset=idx_subset,
      in_memory=in_memory,
      **kwargs,
    )
  
  @classmethod
  def download_dataset(
      cls, dir_path, db_name: str = "suzuki.db", force_download: bool = True, clean: bool = True
  ):
    db_path = os.path.join(dir_path, db_name)
    
    uncompressed_path = os.path.join(dir_path, "vaskas-space-master")
    energies_path = os.path.join(uncompressed_path, cls.energies_path)
    structures_path = os.path.join(uncompressed_path, cls.structures_path)
    
    archive_name = os.path.basename(cls.distant_url)
    archive_path = os.path.join(dir_path, archive_name)
    
    if not os.path.exists(archive_path) or force_download:
      with tqdm.tqdm(
          unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=archive_path
      ) as t:
        requests.urlretrieve(cls.distant_url, filename=archive_path)
        
        shutil.unpack_archive(archive_path, extract_dir=dir_path)
        
        if clean:
          os.remove(archive_path)
    
    df = pd.read_csv(
      energies_path,
      sep=",",
      usecols=cls.usecols,
      index_col=cls.index_col,
    )
    with connect(db_path, append=False) as db:
      for filename, row in tqdm.tqdm(df.iterrows()):
        structure_path = f"{filename}_min.xyz"
        kv_pairs = row.to_dict()
        atoms = ase.io.read(os.path.join(structures_path, structure_path))
        
        db.write(atoms, key_value_pairs=kv_pairs)
    
    if clean:
      shutil.rmtree(uncompressed_path)
    
    print(f"Done downloading Vaska's Complex Dataset, now located at {db_path}.")


if __name__ == "__main__":
  from src.data.components.transforms.ase import AtomsRowToAtomsDataTransform
  
  db_path = "/Home/scratch3/s222491/vaskas/vaskas.db"
  
  dataset = VaskasDataset(
    db_path=db_path,
    download=False,
    transform=AtomsRowToAtomsDataTransform(extract_properties=["barrier"]),
  )
  assert len(dataset) == dataset.expected_length
  print(dataset.node_label_count)
  print(list(dataset.node_label_count.keys()))
  print(dataset.node_count)