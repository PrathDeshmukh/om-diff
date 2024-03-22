from typing import Optional, Iterable, Tuple

import ase
import ase.io
import numpy as np
from scipy.spatial.distance import squareform


class DistanceFilter:
  def __init__(self, min_distance: float = 0.9, max_distance: float = 7.5):
    self.min_distance, self.max_distance = min_distance, max_distance
  
  def __call__(
      self,
      valid_so_far: Optional[np.ndarray] = None,
      atoms_itr: Optional[Iterable[ase.Atoms]] = None,
      filename: Optional[str] = None,
  ) -> np.ndarray:
    min_distances, max_distances = self.gather_distances(valid_so_far, atoms_itr, filename)
    
    valid = np.bitwise_and(
      min_distances > self.min_distance, max_distances < self.max_distance
    )
    
    return valid
  
  def gather_distances(
      self,
      valid_so_far: Optional[np.ndarray] = None,
      atoms_itr: Optional[Iterable[ase.Atoms]] = None,
      filename: Optional[str] = None,
  ) -> Tuple[np.ndarray, np.ndarray]:
    if atoms_itr is None:
      assert filename is not None
      atoms_itr = ase.io.iread(filename, index=":")
    
    min_distances, max_distances = [], []
    for i, atoms in enumerate(atoms_itr):
      if valid_so_far is None or valid_so_far[i]:
        ind1, ind2 = np.triu_indices(len(atoms), k=1)
        D = squareform(
          np.linalg.norm(atoms.positions[ind2] - atoms.positions[ind1], axis=-1)
        )
        D[np.diag_indices(len(D))] = self.max_distance
        D_min = D.min(axis=1)
        min_d, max_d = np.min(D_min), np.max(D_min)
        min_distances.append(min_d)
        max_distances.append(max_d)
      else:
        min_distances.append(0.0)
        max_distances.append(100 * self.max_distance)
    min_distances, max_distances = np.array(min_distances), np.array(max_distances)
    return min_distances, max_distances