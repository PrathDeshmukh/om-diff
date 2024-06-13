from io import StringIO
import xyz2mol as xm
import ase
from ase import Atoms
from os.path import join, dirname
from ase.io import read, write

def atoms_to_xyz_str(atoms: ase.Atoms):
  f = StringIO()
  atoms.write(f, format="xyz")
  return f.getvalue()


def remove_h2(sample):
  at_list = [int(at) for at in sample.get_atomic_numbers()]
  xyz = sample.get_positions()
  
  ase_atom_H2 = sample
  ase_atom = ase_atom_H2.copy()
  
  AC, mol = xm.xyz2AC(at_list, xyz, charge=0)
  
  metal_connectivity = AC[at_list.index(77)]
  H_pos = [list(xyz[i]) for i in range(len(metal_connectivity)) if metal_connectivity[i] == 1 and at_list[i] == 1]
  H_index = [atom.index for atom in ase_atom if atom.position.tolist() in H_pos]
  
  assert len(H_pos) == 2, f"No. of H's is {len(H_pos)}"
  del ase_atom[H_index]
  
  return ase_atom, H_index


def write_basisfile(atoms, basisfile, label, save_path, freeze:bool, addsec=None):
  at_copy = atoms.copy()
  at_types = at_copy.symbols.species()
  at_types.remove('Ir')
  at_list = list(at_types)
  
  with open(basisfile, 'r') as file:
    lines = file.readlines()
  if freeze:
    lines[0] = addsec
    lines[5] = " ".join(at_list) + " " + "0" + "\n"
    path_to_file = join(save_path, "basisfiles", f"basisfile_{label}_freeze.gbs")
  else:
     lines[3] = " ".join(at_list) + " " + "0" + "\n"
     path_to_file = join(save_path, "basisfiles", f"basisfile_{label}.gbs")
  
  with open(path_to_file, 'w') as f:
    f.writelines(lines)
  return path_to_file
