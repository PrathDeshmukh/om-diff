from io import StringIO
import xyz2mol.xyz2mol as xm
import ase
from ase import Atoms
from os.path import join
from ase.io import read, write

def atoms_to_xyz_str(atoms: ase.Atoms):
  f = StringIO()
  atoms.write(f, format="xyz")
  return f.getvalue()



def remove_h2(self) -> None:
  atoms, charge, xyz = xm.read_xyz_file(self.sample)
  
  ase_atom_H2 = Atoms(numbers=atoms, positions=xyz, pbc=False)
  ase_atom = ase_atom_H2.copy()
  
  AC, mol = xm.xyz2AC(atoms, xyz, charge)
  
  metal_connectivity = AC[atoms.index(77)]
  H_pos = [xyz[i] for i in range(len(metal_connectivity)) if metal_connectivity[i] == 1 and atoms[i] == 1]
  assert len(H_pos) == 2, f"No. of H's is {len(H_pos)}"
  del ase_atom[[atom.index for atom in ase_atom if atom.position.tolist() in H_pos]]
  
  return ase_atom


def write_basisfile(atoms, basisfile):
  at_types = atoms.symbols.species()
  at_types.remove('Ir')
  at_list = list(at_types)
  print(at_list)
  
  with open(basisfile, 'r') as file:
    lines = file.readlines()
  
  lines[3] = " ".join(at_list) + " " + "0" + "\n"
  
  with open(basisfile, 'w') as f:
    f.writelines(lines)