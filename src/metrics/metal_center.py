from typing import Optional, Iterable, Tuple, Union

import ase
import ase.io
import numpy as np

from src.metrics.molecular import RDKitMetrics


class MetalCenterMetrics:
  METAL_CENTER_NUMBERS = {
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
  }
  
  @classmethod
  def has_exactly_one_mc(cls, atoms: ase.Atoms) -> int:
    current_mc_atom = None
    for atom in atoms:
      if atom.number in cls.METAL_CENTER_NUMBERS:
        if current_mc_atom is None:
          current_mc_atom = atom
        else:
          current_mc_atom = None
          break
    return current_mc_atom


class OneMetalCenterFilter(MetalCenterMetrics):
  def __call__(
      self,
      valid_so_far: Optional[np.ndarray] = None,
      atoms_itr: Optional[Iterable[ase.Atoms]] = None,
      filename: Optional[str] = None,
      return_mcs: bool = False,
  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if atoms_itr is None:
      assert filename is not None
      atoms_itr = ase.io.iread(filename, index=":")
    
    mcs, valid = [], []
    for i, atoms in enumerate(atoms_itr):
      if valid_so_far is None or valid_so_far[i]:
        current_mc_atom = self.has_exactly_one_mc(atoms)
        if current_mc_atom:
          mcs.append(current_mc_atom.number)
          valid.append(True)
        else:
          mcs.append(0)
          valid.append(False)
      else:
        mcs.append(0)
        valid.append(False)
    
    valid = np.array(valid)
    if return_mcs:
      mcs = np.array(mcs)
      return valid, mcs
    
    return valid


class RDKitFilter(MetalCenterMetrics):
  def __call__(
      self,
      valid_so_far: Optional[np.ndarray] = None,
      atoms_itr: Optional[Iterable[ase.Atoms]] = None,
      filename: Optional[str] = None,
  ) -> np.ndarray:
    if atoms_itr is None:
      assert filename is not None
      atoms_itr = ase.io.iread(filename, index=":")
    
    valid = []
    for i, atoms in enumerate(atoms_itr):
      if valid_so_far is None or valid_so_far[i]:
        # pop metal center
        natoms = len(atoms)
        for j, number in enumerate(reversed(atoms.numbers)):  # OBS: reversed
          if number in self.METAL_CENTER_NUMBERS:
            atoms.pop(natoms - j - 1)
        mol = RDKitMetrics.mol_with_inferred_bonds(atoms)
        valid.append(mol is not None)
      else:
        valid.append(False)
    
    valid = np.array(valid)
    return valid