from typing import Optional, Iterable, Union

import ase
from ase.io import iread
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from src.metrics.utils import atoms_to_xyz_str


class RDKitMetrics:
  def __call__(
      self, atoms_itr: Optional[Iterable[ase.Atoms]] = None, filename: Optional[str] = None
  ) -> float:
    if atoms_itr is None:
      assert filename is not None
      atoms_itr = iread(filename, index=":")
    count, invalid = 0, 0
    for atoms in atoms_itr:
      count += 1
      mol = self.mol_with_inferred_bonds(atoms)
      if mol is None:
        invalid += 1
    
    valid = count - invalid
    return valid / count
  
  @staticmethod
  def mol_with_inferred_bonds(
      atoms: ase.Atoms, try_charges: Optional[list[int]] = None
  ) -> Union[Chem.Mol, None]:
    if try_charges is None:
      try_charges = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    xyz_str = atoms_to_xyz_str(atoms)
    return_mol = None
    
    raw_mol = Chem.MolFromXYZBlock(xyz_str)
    if raw_mol is not None:
      for charge in try_charges:
        try:
          mol = Chem.Mol(raw_mol)
          rdDetermineBonds.DetermineBonds(
            mol,
            charge=charge,
          )
          mol.UpdatePropertyCache(strict=False)
          Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_CLEANUP,
          )
          if mol:
            return mol
          else:
            continue
        except ValueError:
          continue
    return return_mol
